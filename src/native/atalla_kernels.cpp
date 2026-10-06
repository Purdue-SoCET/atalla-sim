// Atalla-Sim native compute kernels.
//
// Accelerates the two data-parallel hot paths of the simulator:
//   1. SystolicArrayTPU.tick() -- the per-cycle grouped-MAC / psum datapath.
//   2. dtype casting -- previously a numpy scalar round-trip costing ~900ns
//      per element, called O(G*S) times per simulated cycle.
//
// Three code paths are compiled and selected at load time via CPUID:
// AVX-512F, AVX2, and portable scalar. The .so therefore runs unmodified on
// any x86-64 host; build once, dispatch per machine.
//
// BIT-EXACTNESS
// -------------
// The vector paths must reproduce the pure-Python reference exactly -- the
// simulator reports saturation/overflow counts and psum values that tests
// pin to specific numbers. Two rules follow:
//
//   * All arithmetic is double precision, because Python floats are doubles.
//   * Products are rounded before they are accumulated, so FMA is NEVER used
//     (fma() keeps the product at infinite precision, which changes results).
//     Every dot product is an explicit mul followed by an add.
//   * Lane reductions keep Python's strict left-to-right order. Vectorisation
//     is therefore across the *column* index j -- whose cells are mutually
//     independent -- never across the lane index being reduced.
//
// Build: make native

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <immintrin.h>
#include <cpuid.h>

#define ATALLA_EXPORT extern "C" __attribute__((visibility("default")))

enum IsaLevel { ISA_SCALAR = 0, ISA_AVX2 = 1, ISA_AVX512 = 2 };

static int g_isa = -1;

// GCC 8 does not accept "f16c" in __builtin_cpu_supports, so query CPUID
// leaf 1 directly: ECX bit 29 is F16C.
static bool cpu_has_f16c() {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    return (ecx & (1u << 29)) != 0;
}

static int detect_isa() {
    __builtin_cpu_init();
    // AVX-512F drives the 8-wide double ops; f16c supplies the half
    // conversion used by the same path.
    if (__builtin_cpu_supports("avx512f") && cpu_has_f16c()) return ISA_AVX512;
    // f16c is required for the half-precision conversion path.
    if (__builtin_cpu_supports("avx2") && cpu_has_f16c()) return ISA_AVX2;
    return ISA_SCALAR;
}

static inline int isa() {
    if (__builtin_expect(g_isa < 0, 0)) g_isa = detect_isa();
    return g_isa;
}

ATALLA_EXPORT int atalla_isa(void) { return isa(); }

ATALLA_EXPORT const char* atalla_isa_name(void) {
    switch (isa()) {
        case ISA_AVX512: return "avx512f";
        case ISA_AVX2:   return "avx2+f16c";
        default:         return "scalar";
    }
}

// ---------------------------------------------------------------------------
// Half-precision (FP16) conversion
// ---------------------------------------------------------------------------
//
// Reference semantics: numpy's  np.asarray(x, dtype=np.float16).item().
// That is round-to-nearest-even, overflow to +/-inf, gradual underflow.
//
// A plain double -> float -> half chain is WRONG: it double-rounds. The
// "innocuous double rounding" bound (p1 >= 2*p2 + 2) applies to results of
// arithmetic on p2-bit operands, not to arbitrary reals, and our inputs are
// arbitrary doubles. Empirically it misrounds roughly 1 value in 20000 --
// every case being a double that sits just off a half-way point and gets
// pulled exactly onto it by the first rounding.
//
// The fix is to make the intermediate rounding round-to-odd. Boldo and
// Melquiond proved RN_p2(RO_p1(x)) == RN_p2(x) whenever p1 >= p2 + 2; here
// p1 = 24 (float) and p2 = 11 (half), so the bound holds with room to spare.
//
// Round-to-odd to float is cheap on the double's bit pattern: truncate the 29
// mantissa bits below float's LSB, and if any of them were set, force that LSB
// to 1. Truncating cannot carry, so no exponent fixup is needed. The result
// has a 24-bit significand and converts to float exactly; f16c then does the
// single correctly-rounded step to half.
//
// Specials need no branches: infinities have a zero mantissa so they are
// untouched, and NaNs stay NaN under a mantissa OR. Doubles too small to be
// normal floats are far below half's smallest subnormal and flush to zero
// either way.
//
// Verified exhaustively against numpy by tests/native/test_native_kernels.py.

static const uint64_t RO_LOW_MASK = 0x000000001FFFFFFFULL;  // below float's LSB
static const uint64_t RO_LSB      = 0x0000000020000000ULL;  // float's LSB

static inline double round_to_odd_float(double v) {
    uint64_t b;
    std::memcpy(&b, &v, 8);
    uint64_t t = b & ~RO_LOW_MASK;
    if (b & RO_LOW_MASK) t |= RO_LSB;
    double d;
    std::memcpy(&d, &t, 8);
    return d;
}

// Portable scalar half conversion, used when f16c is unavailable.
static inline double cast_half_scalar(double v) {
    const float f = (float)round_to_odd_float(v);   // exact: 24-bit significand
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const uint32_t sign = b & 0x80000000u;
    const int32_t  exp  = (int32_t)((b >> 23) & 0xFF);
    uint32_t mant = b & 0x007FFFFFu;

    if (exp == 0xFF) {                      // inf / nan propagate
        if (mant) return (double)f;         // nan stays nan
        return sign ? -INFINITY : INFINITY;
    }

    const int32_t unbiased = exp - 127;
    if (unbiased > 15) {                    // magnitude above half's range
        return sign ? -INFINITY : INFINITY;
    }
    if (unbiased >= -14) {                  // normal half
        // Round mantissa from 23 bits to 10, ties to even.
        const uint32_t round_bit = 1u << 12;
        const uint32_t lsb       = 1u << 13;
        uint32_t rounded = mant + (round_bit - 1) + ((mant & lsb) ? 1 : 0);
        int32_t  e = unbiased;
        if (rounded & 0x00800000u) {        // mantissa overflowed into exponent
            rounded = 0;
            e += 1;
            if (e > 15) return sign ? -INFINITY : INFINITY;
        }
        const uint32_t h_mant = (rounded & 0x007FFFFFu) >> 13;
        // Rebuild as a float (half values are all exactly representable).
        const uint32_t out_bits = sign | ((uint32_t)(e + 127) << 23) | (h_mant << 13);
        float out;
        std::memcpy(&out, &out_bits, 4);
        return (double)out;
    }
    if (unbiased < -25) {                   // rounds to zero
        return sign ? -0.0 : 0.0;
    }
    // Subnormal half: shift the implicit 1 back in and round at the
    // fixed exponent of 2^-24.
    mant |= 0x00800000u;
    const int32_t shift = -14 - unbiased;   // 1 .. 11
    const int32_t total = 13 + shift;       // bits discarded
    const uint32_t round_bit = 1u << (total - 1);
    const uint32_t lsb       = 1u << total;
    const uint32_t sticky    = mant & (lsb - 1);
    uint32_t q = mant >> total;
    if ((sticky > round_bit) || (sticky == round_bit && (q & 1))) q += 1;
    if (q == 0) return sign ? -0.0 : 0.0;
    const double mag = (double)q * 5.9604644775390625e-08;  // 2^-24
    return sign ? -mag : mag;
}

__attribute__((target("f16c")))
static inline double cast_half_f16c(double v) {
    const float f = (float)round_to_odd_float(v);
    __m128 s = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return (double)_mm_cvtss_f32(_mm_cvtph_ps(h));
}

// ---------------------------------------------------------------------------
// bfloat16 conversion
// ---------------------------------------------------------------------------
//
// Reference semantics: base.dtype.bf16_round, which is the functional sim's
// bf16_round -- round to FP32 first (an ordinary nearest-even conversion, so
// this one DOES double-round, on purpose: it is what the hardware does), then
// nearest-even on the top 16 bits: add 0x7FFF plus bit 16 and clear the low
// half. A carry out of the mantissa bumps the exponent, so the largest finite
// floats round up to infinity. NaNs are quieted instead of rounded, since the
// add would carry a NaN with all-ones low bits into the sign bit.

static inline float bf16_round_f32(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu)) {
        b |= 0x00400000u;
    } else {
        b += 0x7FFFu + ((b >> 16) & 1u);
    }
    b &= 0xFFFF0000u;
    std::memcpy(&f, &b, 4);
    return f;
}

static inline double cast_bf16_scalar(double v) {
    return (double)bf16_round_f32((float)v);
}

// ---------------------------------------------------------------------------
// Batched casts
// ---------------------------------------------------------------------------
// mode: 0 = half (FP16)
//       2 = bf16
//       1 = int8 (truncate toward zero; out-of-range is reported, not clamped,
//                 matching numpy's OverflowError on Python scalars)
//
// count_sat / count_ovf are only accumulated when track != 0, mirroring
// SystolicArrayTPU._note_cast which only counts for DType.FP16.

struct CastStats {
    int64_t sat;        // |input| > 65504
    int64_t ovf;        // output is not finite
    int32_t int8_range; // an int8 value fell outside [-128, 127]
};

__attribute__((target("avx2,f16c")))
static void cast_half_array_avx2(const double* in, double* out, int64_t n,
                                 int track, CastStats* st) {
    int64_t i = 0;
    int64_t sat = 0, ovf = 0;
    const __m256d lim = _mm256_set1_pd(65504.0);
    const __m256d absmask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    const __m256i lowm = _mm256_set1_epi64x((long long)RO_LOW_MASK);
    const __m256i lsb  = _mm256_set1_epi64x((long long)RO_LSB);
    const __m256i izero = _mm256_setzero_si256();
    for (; i + 4 <= n; i += 4) {
        __m256d d = _mm256_loadu_pd(in + i);
        // round-to-odd to float precision, then one exact step to half
        __m256i bi   = _mm256_castpd_si256(d);
        __m256i low  = _mm256_and_si256(bi, lowm);
        __m256i keep = _mm256_andnot_si256(lowm, bi);
        __m256i isz  = _mm256_cmpeq_epi64(low, izero);      // low bits all zero
        __m256i odd  = _mm256_or_si256(keep, _mm256_andnot_si256(isz, lsb));
        __m128  f = _mm256_cvtpd_ps(_mm256_castsi256_pd(odd));
        __m128i h = _mm_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128  b = _mm_cvtph_ps(h);
        __m256d r = _mm256_cvtps_pd(b);
        _mm256_storeu_pd(out + i, r);
        if (track) {
            __m256d a = _mm256_and_pd(d, absmask);
            sat += __builtin_popcount((unsigned)_mm256_movemask_pd(_mm256_cmp_pd(a, lim, _CMP_GT_OQ)));
            // non-finite == (r != r) or |r| == inf; compare against itself and inf
            __m256d ra = _mm256_and_pd(r, absmask);
            __m256d inf = _mm256_set1_pd(INFINITY);
            __m256d isnan = _mm256_cmp_pd(r, r, _CMP_UNORD_Q);
            __m256d isinf = _mm256_cmp_pd(ra, inf, _CMP_EQ_OQ);
            ovf += __builtin_popcount((unsigned)_mm256_movemask_pd(_mm256_or_pd(isnan, isinf)));
        }
    }
    for (; i < n; ++i) {
        double v = in[i];
        double r = cast_half_f16c(v);
        out[i] = r;
        if (track) {
            if (std::fabs(v) > 65504.0) sat++;
            if (!std::isfinite(r)) ovf++;
        }
    }
    st->sat += sat;
    st->ovf += ovf;
}

__attribute__((target("avx512f,f16c")))
static void cast_half_array_avx512(const double* in, double* out, int64_t n,
                                   int track, CastStats* st) {
    int64_t i = 0;
    int64_t sat = 0, ovf = 0;
    const __m512d lim = _mm512_set1_pd(65504.0);
    const __m512d inf = _mm512_set1_pd(INFINITY);
    const __m512i lowm = _mm512_set1_epi64((long long)RO_LOW_MASK);
    const __m512i lsb  = _mm512_set1_epi64((long long)RO_LSB);
    const __m512i izero = _mm512_setzero_si512();
    for (; i + 8 <= n; i += 8) {
        __m512d d = _mm512_loadu_pd(in + i);
        // round-to-odd to float precision, then one exact step to half
        __m512i bi   = _mm512_castpd_si512(d);
        __m512i low  = _mm512_and_si512(bi, lowm);
        __m512i keep = _mm512_andnot_si512(lowm, bi);
        __mmask8 nz  = _mm512_cmpneq_epi64_mask(low, izero);
        __m512i odd  = _mm512_mask_or_epi64(keep, nz, keep, lsb);
        __m256  f = _mm512_cvtpd_ps(_mm512_castsi512_pd(odd));
        __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256  b = _mm256_cvtph_ps(h);
        __m512d r = _mm512_cvtps_pd(b);
        _mm512_storeu_pd(out + i, r);
        if (track) {
            __m512d a = _mm512_abs_pd(d);
            sat += __builtin_popcount((unsigned)_mm512_cmp_pd_mask(a, lim, _CMP_GT_OQ));
            __m512d ra = _mm512_abs_pd(r);
            unsigned m = (unsigned)_mm512_cmp_pd_mask(r, r, _CMP_UNORD_Q)
                       | (unsigned)_mm512_cmp_pd_mask(ra, inf, _CMP_EQ_OQ);
            ovf += __builtin_popcount(m);
        }
    }
    for (; i < n; ++i) {
        double v = in[i];
        double r = cast_half_f16c(v);
        out[i] = r;
        if (track) {
            if (std::fabs(v) > 65504.0) sat++;
            if (!std::isfinite(r)) ovf++;
        }
    }
    st->sat += sat;
    st->ovf += ovf;
}

static void cast_half_array_scalar(const double* in, double* out, int64_t n,
                                   int track, CastStats* st) {
    int64_t sat = 0, ovf = 0;
    for (int64_t i = 0; i < n; ++i) {
        double v = in[i];
        double r = cast_half_scalar(v);
        out[i] = r;
        if (track) {
            if (std::fabs(v) > 65504.0) sat++;
            if (!std::isfinite(r)) ovf++;
        }
    }
    st->sat += sat;
    st->ovf += ovf;
}

// Plain loop; the compiler vectorises it, and the cast is cheap next to the
// ctypes call that reaches it.
static void cast_bf16_array(const double* in, double* out, int64_t n,
                            int track, CastStats* st) {
    int64_t sat = 0, ovf = 0;
    for (int64_t i = 0; i < n; ++i) {
        double v = in[i];
        double r = cast_bf16_scalar(v);
        out[i] = r;
        if (track) {
            if (std::fabs(v) > 3.3895313892515355e38) sat++;  // bf16 max
            if (!std::isfinite(r)) ovf++;
        }
    }
    st->sat += sat;
    st->ovf += ovf;
}

static void cast_int8_array(const double* in, double* out, int64_t n, CastStats* st) {
    for (int64_t i = 0; i < n; ++i) {
        double v = in[i];
        if (!std::isfinite(v)) { st->int8_range = 1; out[i] = v; continue; }
        double t = std::trunc(v);
        if (t < -128.0 || t > 127.0) { st->int8_range = 1; out[i] = t; continue; }
        out[i] = t;
    }
}

// Dispatching entry point used for whole-array casts.
static void cast_array(const double* in, double* out, int64_t n, int mode,
                       int track, CastStats* st) {
    if (mode == 1) { cast_int8_array(in, out, n, st); return; }
    if (mode == 2) { cast_bf16_array(in, out, n, track, st); return; }
    switch (isa()) {
        case ISA_AVX512: cast_half_array_avx512(in, out, n, track, st); break;
        case ISA_AVX2:   cast_half_array_avx2(in, out, n, track, st);   break;
        default:         cast_half_array_scalar(in, out, n, track, st); break;
    }
}

ATALLA_EXPORT void atalla_cast_array(const double* in, double* out, int64_t n,
                                     int mode, int track, int64_t* stats_out) {
    CastStats st = {0, 0, 0};
    cast_array(in, out, n, mode, track, &st);
    if (stats_out) {
        stats_out[0] += st.sat;
        stats_out[1] += st.ovf;
        stats_out[2] |= st.int8_range;
    }
}

ATALLA_EXPORT double atalla_cast_scalar(double v, int mode, int32_t* int8_range) {
    if (mode == 1) {
        if (!std::isfinite(v)) { if (int8_range) *int8_range = 1; return v; }
        double t = std::trunc(v);
        if (t < -128.0 || t > 127.0) { if (int8_range) *int8_range = 1; }
        return t;
    }
    if (mode == 2) return cast_bf16_scalar(v);
    return (isa() == ISA_SCALAR) ? cast_half_scalar(v) : cast_half_f16c(v);
}

// ---------------------------------------------------------------------------
// Systolic array tick
// ---------------------------------------------------------------------------
//
// State layout is LANE-MAJOR: act[g][lane][j] and wgt[g][lane][j], so that the
// column index j -- the axis we vectorise over -- is contiguous, and the
// per-column shift becomes one memmove per (g, lane) row.
//
// Phase order below is chosen so that no snapshot copies are needed. The
// Python reference snapshots act/wgt/mul/acc up front because its shift runs
// before its compute; here compute runs first and the shift last, and the psum
// pass walks g downward so acc[g-1] is still the previous cycle's value when
// it is read. The observable result is identical.

struct SaState {
    int32_t G, S, GS;
    double* act;   // G * GS * S
    double* wgt;   // G * GS * S
    double* mul;   // G * S
    double* acc;   // G * S
};

// Metric slots returned to Python.
enum {
    M_ACTIVE_PES = 0,
    M_PSUM_NNZ   = 1,
    M_SAT        = 2,
    M_OVF        = 3,
    M_SHIFT_NNZ  = 4,
    M_INT8_RANGE = 5,   // an int8 cast fell outside [-128, 127]
    M_COUNT      = 6
};

// ---- active PE count: cells where activation and weight are both non-zero --

static int64_t count_active_scalar(const double* act, const double* wgt, int64_t n) {
    int64_t c = 0;
    for (int64_t i = 0; i < n; ++i) c += (act[i] != 0.0) && (wgt[i] != 0.0);
    return c;
}

__attribute__((target("avx2")))
static int64_t count_active_avx2(const double* act, const double* wgt, int64_t n) {
    int64_t c = 0, i = 0;
    const __m256d z = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        __m256d a = _mm256_loadu_pd(act + i);
        __m256d w = _mm256_loadu_pd(wgt + i);
        __m256d m = _mm256_and_pd(_mm256_cmp_pd(a, z, _CMP_NEQ_UQ),
                                  _mm256_cmp_pd(w, z, _CMP_NEQ_UQ));
        c += __builtin_popcount((unsigned)_mm256_movemask_pd(m));
    }
    for (; i < n; ++i) c += (act[i] != 0.0) && (wgt[i] != 0.0);
    return c;
}

__attribute__((target("avx512f")))
static int64_t count_active_avx512(const double* act, const double* wgt, int64_t n) {
    int64_t c = 0, i = 0;
    const __m512d z = _mm512_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        __m512d a = _mm512_loadu_pd(act + i);
        __m512d w = _mm512_loadu_pd(wgt + i);
        unsigned m = (unsigned)_mm512_cmp_pd_mask(a, z, _CMP_NEQ_UQ)
                   & (unsigned)_mm512_cmp_pd_mask(w, z, _CMP_NEQ_UQ);
        c += __builtin_popcount(m);
    }
    for (; i < n; ++i) c += (act[i] != 0.0) && (wgt[i] != 0.0);
    return c;
}

static int64_t count_nonzero_scalar(const double* p, int64_t n) {
    int64_t c = 0;
    for (int64_t i = 0; i < n; ++i) c += (p[i] != 0.0);
    return c;
}

__attribute__((target("avx2")))
static int64_t count_nonzero_avx2(const double* p, int64_t n) {
    int64_t c = 0, i = 0;
    const __m256d z = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(p + i);
        c += __builtin_popcount((unsigned)_mm256_movemask_pd(_mm256_cmp_pd(v, z, _CMP_NEQ_UQ)));
    }
    for (; i < n; ++i) c += (p[i] != 0.0);
    return c;
}

__attribute__((target("avx512f")))
static int64_t count_nonzero_avx512(const double* p, int64_t n) {
    int64_t c = 0, i = 0;
    const __m512d z = _mm512_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(p + i);
        c += __builtin_popcount((unsigned)_mm512_cmp_pd_mask(v, z, _CMP_NEQ_UQ));
    }
    for (; i < n; ++i) c += (p[i] != 0.0);
    return c;
}

static inline int64_t count_active(const double* a, const double* w, int64_t n) {
    switch (isa()) {
        case ISA_AVX512: return count_active_avx512(a, w, n);
        case ISA_AVX2:   return count_active_avx2(a, w, n);
        default:         return count_active_scalar(a, w, n);
    }
}

static inline int64_t count_nonzero(const double* p, int64_t n) {
    switch (isa()) {
        case ISA_AVX512: return count_nonzero_avx512(p, n);
        case ISA_AVX2:   return count_nonzero_avx2(p, n);
        default:         return count_nonzero_scalar(p, n);
    }
}

// ---- accumulate: dst[j] += a[j] * b[j], strictly mul-then-add (no FMA) -----

static void mul_add_scalar(double* dst, const double* a, const double* b, int64_t n) {
    for (int64_t j = 0; j < n; ++j) dst[j] += a[j] * b[j];
}

__attribute__((target("avx2")))
static void mul_add_avx2(double* dst, const double* a, const double* b, int64_t n) {
    int64_t j = 0;
    for (; j + 4 <= n; j += 4) {
        __m256d p = _mm256_mul_pd(_mm256_loadu_pd(a + j), _mm256_loadu_pd(b + j));
        _mm256_storeu_pd(dst + j, _mm256_add_pd(_mm256_loadu_pd(dst + j), p));
    }
    for (; j < n; ++j) dst[j] += a[j] * b[j];
}

__attribute__((target("avx512f")))
static void mul_add_avx512(double* dst, const double* a, const double* b, int64_t n) {
    int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
        __m512d p = _mm512_mul_pd(_mm512_loadu_pd(a + j), _mm512_loadu_pd(b + j));
        _mm512_storeu_pd(dst + j, _mm512_add_pd(_mm512_loadu_pd(dst + j), p));
    }
    for (; j < n; ++j) dst[j] += a[j] * b[j];
}

static inline void mul_add(double* dst, const double* a, const double* b, int64_t n) {
    switch (isa()) {
        case ISA_AVX512: mul_add_avx512(dst, a, b, n); break;
        case ISA_AVX2:   mul_add_avx2(dst, a, b, n);   break;
        default:         mul_add_scalar(dst, a, b, n); break;
    }
}

// ---- accumulate with a broadcast scalar: dst[j] += s * b[j] ----------------

static void scal_add_scalar(double* dst, double s, const double* b, int64_t n) {
    for (int64_t j = 0; j < n; ++j) dst[j] += s * b[j];
}

__attribute__((target("avx2")))
static void scal_add_avx2(double* dst, double s, const double* b, int64_t n) {
    int64_t j = 0;
    const __m256d vs = _mm256_set1_pd(s);
    for (; j + 4 <= n; j += 4) {
        __m256d p = _mm256_mul_pd(vs, _mm256_loadu_pd(b + j));
        _mm256_storeu_pd(dst + j, _mm256_add_pd(_mm256_loadu_pd(dst + j), p));
    }
    for (; j < n; ++j) dst[j] += s * b[j];
}

__attribute__((target("avx512f")))
static void scal_add_avx512(double* dst, double s, const double* b, int64_t n) {
    int64_t j = 0;
    const __m512d vs = _mm512_set1_pd(s);
    for (; j + 8 <= n; j += 8) {
        __m512d p = _mm512_mul_pd(vs, _mm512_loadu_pd(b + j));
        _mm512_storeu_pd(dst + j, _mm512_add_pd(_mm512_loadu_pd(dst + j), p));
    }
    for (; j < n; ++j) dst[j] += s * b[j];
}

static inline void scal_add(double* dst, double s, const double* b, int64_t n) {
    switch (isa()) {
        case ISA_AVX512: scal_add_avx512(dst, s, b, n); break;
        case ISA_AVX2:   scal_add_avx2(dst, s, b, n);   break;
        default:         scal_add_scalar(dst, s, b, n); break;
    }
}

// Scratch buffers grow on demand; the tick is called millions of times so the
// allocation must not be per-call.
static double*  g_scratch = nullptr;
static int64_t  g_scratch_n = 0;

// All vector loads/stores are unaligned, so plain malloc is sufficient.
static double* scratch(int64_t n) {
    if (n > g_scratch_n) {
        free(g_scratch);
        g_scratch = (double*)malloc((size_t)n * sizeof(double));
        g_scratch_n = g_scratch ? n : 0;
    }
    return g_scratch;
}

ATALLA_EXPORT void atalla_sa_free_scratch(void) {
    free(g_scratch);
    g_scratch = nullptr;
    g_scratch_n = 0;
}

// One simulated cycle of the systolic array datapath.
//
//   start / weight_en / mac_shift : control signals, as in Python
//   has_dtype                     : whether _current_dtype is set
//   cast_mode                     : 0 = half, 1 = int8, 2 = bf16
//   track_sat                     : count saturation/overflow (FP16 only)
//   psum_top[S], psum_top_valid[S]: values dequeued for the g == 0 boundary
//   shift_in[G*GS]                : boundary vector entering column 0
//   issued[G*GS]                  : activations issued this cycle; all zero
//                                   unless the activation shift ran, matching
//                                   Python's issued_groups
//   issued_out[S]                 : output row, written only when start
//   metrics[M_COUNT]              : cleared on entry, so every slot reports
//                                   this cycle's value only
ATALLA_EXPORT void atalla_sa_tick(
        SaState* st,
        int start, int weight_en, int mac_shift,
        int has_dtype, int cast_mode, int track_sat,
        const double* psum_top, const uint8_t* psum_top_valid,
        const double* shift_in, const double* issued,
        double* issued_out,
        int64_t* metrics) {

    const int32_t G = st->G, S = st->S, GS = st->GS;
    const int64_t plane = (int64_t)GS * S;   // per-group stride in act/wgt
    CastStats cs = {0, 0, 0};

    // Sized once for the largest consumer (phase D) so no phase reallocates.
    double* const work = scratch(2 * S);
    if (!work) return;

    for (int i = 0; i < M_COUNT; ++i) metrics[i] = 0;

    // -- Phase A: active PE count (reads pre-shift act/wgt) ------------------
    if (start) {
        metrics[M_ACTIVE_PES] = count_active(st->act, st->wgt, (int64_t)G * plane);
    }

    // -- Phase B: psum accumulation ------------------------------------------
    // acc[g][j] = cast(mul[g][j] + psum_in[g][j]) where psum_in is the top
    // boundary for g == 0 and the *previous cycle's* acc[g-1][j] otherwise.
    // Walking g downward keeps acc[g-1] unmodified when it is read.
    {
        double* buf = work;
        int64_t nnz = 0;
        for (int32_t g = G - 1; g >= 0; --g) {
            double* accg = st->acc + (int64_t)g * S;
            const double* mulg = st->mul + (int64_t)g * S;
            if (g == 0) {
                for (int32_t j = 0; j < S; ++j) {
                    buf[j] = psum_top_valid[j] ? psum_top[j] : 0.0;
                }
            } else {
                std::memcpy(buf, st->acc + (int64_t)(g - 1) * S, (size_t)S * 8);
            }
            nnz += count_nonzero(buf, S);
            for (int32_t j = 0; j < S; ++j) buf[j] = mulg[j] + buf[j];
            if (has_dtype) {
                cast_array(buf, accg, S, cast_mode, track_sat, &cs);
            } else {
                std::memcpy(accg, buf, (size_t)S * 8);
            }
        }
        metrics[M_PSUM_NNZ] += nnz;
    }

    // -- Phase C: grouped MAC (reads pre-shift act/wgt) ----------------------
    // dot[g][j] = sum over lane of act[g][lane][j] * wgt[g][lane][j], summed
    // left-to-right over lane exactly as Python's sum() does.
    if (start) {
        double* buf = work;
        for (int32_t g = 0; g < G; ++g) {
            std::memset(buf, 0, (size_t)S * 8);
            const double* ag = st->act + (int64_t)g * plane;
            const double* wg = st->wgt + (int64_t)g * plane;
            for (int32_t l = 0; l < GS; ++l) {
                mul_add(buf, ag + (int64_t)l * S, wg + (int64_t)l * S, S);
            }
            double* mulg = st->mul + (int64_t)g * S;
            if (has_dtype) {
                cast_array(buf, mulg, S, cast_mode, track_sat, &cs);
            } else {
                std::memcpy(mulg, buf, (size_t)S * 8);
            }
        }
    }

    // -- Phase D: output row (reads pre-shift wgt) ---------------------------
    // issued_out[j] = sum over g of ( sum over lane of issued[g][lane] *
    // wgt[g][lane][j] ), preserving Python's per-group then per-lane order.
    if (start) {
        double* buf = work;                 // one allocation, split in two
        double* grp = buf + S;
        std::memset(buf, 0, (size_t)S * 8);
        for (int32_t g = 0; g < G; ++g) {
            std::memset(grp, 0, (size_t)S * 8);
            const double* wg = st->wgt + (int64_t)g * plane;
            for (int32_t l = 0; l < GS; ++l) {
                scal_add(grp, issued[(int64_t)g * GS + l], wg + (int64_t)l * S, S);
            }
            for (int32_t j = 0; j < S; ++j) buf[j] += grp[j];
        }
        if (has_dtype) {
            cast_array(buf, issued_out, S, cast_mode, track_sat, &cs);
        } else {
            std::memcpy(issued_out, buf, (size_t)S * 8);
        }
    }

    // -- Phase E: shift ------------------------------------------------------
    // Column j receives column j-1; column 0 receives the boundary vector.
    // Lane-major layout makes this one memmove per (g, lane).
    if (weight_en || mac_shift) {
        double* base = weight_en ? st->wgt : st->act;
        for (int32_t g = 0; g < G; ++g) {
            double* bg = base + (int64_t)g * plane;
            for (int32_t l = 0; l < GS; ++l) {
                double* row = bg + (int64_t)l * S;
                std::memmove(row + 1, row, (size_t)(S - 1) * 8);
                row[0] = shift_in[(int64_t)g * GS + l];
            }
        }
        // Python counts the non-zeros of every value that lands in a cell,
        // which is exactly the non-zero count of the post-shift array.
        metrics[M_SHIFT_NNZ] += count_nonzero(base, (int64_t)G * plane);
    }

    metrics[M_SAT] += cs.sat;
    metrics[M_OVF] += cs.ovf;
    metrics[M_INT8_RANGE] += cs.int8_range;
}

// ---------------------------------------------------------------------------
// Note on the vector datapath
// ---------------------------------------------------------------------------
// VectorLane models a sequencer that retires one element per functional unit
// per simulated cycle, so its arithmetic is scalar by construction -- there is
// nothing there for SIMD to widen without changing the timing being modelled.
// Its measured cost is queue and pipeline bookkeeping, not floating point.
//
// The vector work that IS data-parallel is the whole-vector dtype cast in
// VectorDatapath.enqueue, and that runs through atalla_cast_array above.

// ===========================================================================
// MEISSA systolic array
// ===========================================================================
//
// Per cycle the array shifts its activation grid one column right, forms N*N
// products, and reduces each column's N products through a pipelined adder
// tree. The columns are mutually independent and the tree is identical for
// each, so the vectorisation axis is the COLUMN index j -- never the term
// index i, which is the axis being reduced and whose order must be preserved.
// Laying the grid out with j contiguous turns every tree level into a plain
// elementwise vector add over N floats.
//
// BIT-EXACTNESS, as for the TPU kernel above, but note the different type:
// MEISSA carries products and tree sums in FLOAT32 (the RTL's FP32 datapath),
// not double. Two consequences:
//   * a 2-input level is a float32 add, rounding once per level;
//   * a 4-input level accumulates in double and rounds ONCE on the way out,
//     which is what a real 4-input FP adder does and what makes it differ
//     from three cascaded 2-input adds. The double adds run left to right so
//     they match the reference's a+b+c+d evaluation order exactly.
// No FMA anywhere: products are rounded to float32 before they are summed.

typedef struct {
    int32_t  N, depth, out_depth, head;
    int32_t  levels4, levels2, do_reduce, collect_stats;
    float*   act;          // N*N, act[i*N + j] -- j contiguous
    float*   wgt;          // N*N
    int64_t* col_seq;      // N, -1 marks a bubble
    float*   tree_vals;    // depth*N ring buffer
    int64_t* tree_seq;     // depth*N
    float*   bank_vals;    // N*out_depth, one bank per column
    int64_t* bank_seq;     // N*out_depth
    int32_t* bank_head;    // N
    int32_t* bank_count;   // N
    int32_t  next_seq, credits, max_credits;
} MeissaState;

enum { MM_ACTIVE_PES = 0, MM_MUL_OPS = 1, MM_ADD_OPS = 2, MM_OUT_ROWS = 3,
       MM_ACT_SHIFT_NNZ = 4,   // nonzero activations that moved a column
       MM_WGT_SHIFT_NNZ = 5,   // nonzero weights that moved a column
       MM_OUT_NNZ = 6,         // nonzero values written into the banks
       MM_OVERFLOW = 7,        // finite tree sums the reducer sent to infinity
       MM_MAX_ACTIVE = 8,      // peak active PEs in any one cycle
       MM_LIVE_CYCLES = 9,     // cycles with at least one live column
       MM_COUNT = 10 };

// -- elementwise product, one tree level, over n contiguous columns ----------

static void mul_f32_scalar(float* d, const float* a, const float* b, int n) {
    for (int k = 0; k < n; ++k) d[k] = a[k] * b[k];
}
__attribute__((target("avx2")))
static void mul_f32_avx2(float* d, const float* a, const float* b, int n) {
    int k = 0;
    for (; k + 8 <= n; k += 8)
        _mm256_storeu_ps(d + k, _mm256_mul_ps(_mm256_loadu_ps(a + k),
                                              _mm256_loadu_ps(b + k)));
    for (; k < n; ++k) d[k] = a[k] * b[k];
}
__attribute__((target("avx512f")))
static void mul_f32_avx512(float* d, const float* a, const float* b, int n) {
    int k = 0;
    for (; k + 16 <= n; k += 16)
        _mm512_storeu_ps(d + k, _mm512_mul_ps(_mm512_loadu_ps(a + k),
                                              _mm512_loadu_ps(b + k)));
    for (; k < n; ++k) d[k] = a[k] * b[k];
}
static inline void mul_f32(float* d, const float* a, const float* b, int n) {
    switch (isa()) {
        case ISA_AVX512: mul_f32_avx512(d, a, b, n); return;
        case ISA_AVX2:   mul_f32_avx2(d, a, b, n);   return;
        default:         mul_f32_scalar(d, a, b, n);
    }
}

static void add2_f32_scalar(float* d, const float* a, const float* b, int n) {
    for (int k = 0; k < n; ++k) d[k] = a[k] + b[k];
}
__attribute__((target("avx2")))
static void add2_f32_avx2(float* d, const float* a, const float* b, int n) {
    int k = 0;
    for (; k + 8 <= n; k += 8)
        _mm256_storeu_ps(d + k, _mm256_add_ps(_mm256_loadu_ps(a + k),
                                              _mm256_loadu_ps(b + k)));
    for (; k < n; ++k) d[k] = a[k] + b[k];
}
__attribute__((target("avx512f")))
static void add2_f32_avx512(float* d, const float* a, const float* b, int n) {
    int k = 0;
    for (; k + 16 <= n; k += 16)
        _mm512_storeu_ps(d + k, _mm512_add_ps(_mm512_loadu_ps(a + k),
                                              _mm512_loadu_ps(b + k)));
    for (; k < n; ++k) d[k] = a[k] + b[k];
}
static inline void add2_f32(float* d, const float* a, const float* b, int n) {
    switch (isa()) {
        case ISA_AVX512: add2_f32_avx512(d, a, b, n); return;
        case ISA_AVX2:   add2_f32_avx2(d, a, b, n);   return;
        default:         add2_f32_scalar(d, a, b, n);
    }
}

// A 4-input adder: accumulate in double, round to float32 exactly once.
static void add4_f32_scalar(float* d, const float* a, const float* b,
                            const float* c, const float* e, int n) {
    for (int k = 0; k < n; ++k)
        d[k] = (float)((((double)a[k] + (double)b[k]) + (double)c[k])
                       + (double)e[k]);
}
__attribute__((target("avx2")))
static void add4_f32_avx2(float* d, const float* a, const float* b,
                          const float* c, const float* e, int n) {
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        __m256d s = _mm256_cvtps_pd(_mm_loadu_ps(a + k));
        s = _mm256_add_pd(s, _mm256_cvtps_pd(_mm_loadu_ps(b + k)));
        s = _mm256_add_pd(s, _mm256_cvtps_pd(_mm_loadu_ps(c + k)));
        s = _mm256_add_pd(s, _mm256_cvtps_pd(_mm_loadu_ps(e + k)));
        _mm_storeu_ps(d + k, _mm256_cvtpd_ps(s));
    }
    for (; k < n; ++k)
        d[k] = (float)((((double)a[k] + (double)b[k]) + (double)c[k])
                       + (double)e[k]);
}
__attribute__((target("avx512f")))
static void add4_f32_avx512(float* d, const float* a, const float* b,
                            const float* c, const float* e, int n) {
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m512d s = _mm512_cvtps_pd(_mm256_loadu_ps(a + k));
        s = _mm512_add_pd(s, _mm512_cvtps_pd(_mm256_loadu_ps(b + k)));
        s = _mm512_add_pd(s, _mm512_cvtps_pd(_mm256_loadu_ps(c + k)));
        s = _mm512_add_pd(s, _mm512_cvtps_pd(_mm256_loadu_ps(e + k)));
        _mm256_storeu_ps(d + k, _mm512_cvtpd_ps(s));
    }
    for (; k < n; ++k)
        d[k] = (float)((((double)a[k] + (double)b[k]) + (double)c[k])
                       + (double)e[k]);
}
static inline void add4_f32(float* d, const float* a, const float* b,
                            const float* c, const float* e, int n) {
    switch (isa()) {
        case ISA_AVX512: add4_f32_avx512(d, a, b, c, e, n); return;
        case ISA_AVX2:   add4_f32_avx2(d, a, b, c, e, n);   return;
        default:         add4_f32_scalar(d, a, b, c, e, n);
    }
}

// The reducer: round float32 to the storage type and back. do_reduce is
// 1 for IEEE binary16 (nearest-even) and 2 for bfloat16; either way it must
// match cast_vector bit for bit.
static void half_round_scalar(float* d, const float* s, int n) {
    for (int k = 0; k < n; ++k) {
        double v = (double)s[k];
        int32_t dummy = 0;
        d[k] = (float)atalla_cast_scalar(v, 0, &dummy);
    }
}
__attribute__((target("avx2,f16c")))
static void half_round_f16c(float* d, const float* s, int n) {
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(s + k),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_ps(d + k, _mm256_cvtph_ps(h));
    }
    for (; k < n; ++k) {
        __m128i h = _mm_cvtps_ph(_mm_set_ss(s[k]),
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        d[k] = _mm_cvtss_f32(_mm_cvtph_ps(h));
    }
}
static inline void half_round(float* d, const float* s, int n) {
    if (isa() >= ISA_AVX2) half_round_f16c(d, s, n);
    else                   half_round_scalar(d, s, n);
}
static void bf16_round_arr(float* d, const float* s, int n) {
    for (int k = 0; k < n; ++k) d[k] = bf16_round_f32(s[k]);
}

// -- scratch ----------------------------------------------------------------
static float* g_mscratch = nullptr;
static size_t g_mscratch_n = 0;

static float* mscratch(size_t need) {
    if (g_mscratch_n < need) {
        free(g_mscratch);
        g_mscratch = (float*)malloc(need * sizeof(float));
        g_mscratch_n = g_mscratch ? need : 0;
    }
    return g_mscratch;
}

// The Python side sizes the metrics buffer from its own copy of MM_COUNT.
// Export the kernel's value so a mismatch is a failing test rather than a
// heap overrun.
ATALLA_EXPORT int32_t atalla_meissa_metric_count(void) { return MM_COUNT; }

ATALLA_EXPORT void atalla_meissa_free_scratch(void) {
    free(g_mscratch);
    g_mscratch = nullptr;
    g_mscratch_n = 0;
}

// Reduce every column at once. `terms` is N*N with j contiguous; `out` is N.
// `buf` must hold 2*N*N floats. Taking it as an argument rather than calling
// mscratch() here is deliberate: one allocation per cycle, so a realloc can
// never invalidate a pointer another part of the cycle is still holding.
static void reduce_columns(const MeissaState* st, const float* terms, float* out,
                           float* buf, int64_t* metrics) {
    const int N = st->N;
    float* cur = buf;
    float* nxt = buf + (size_t)N * N;

    memcpy(cur, terms, (size_t)N * N * sizeof(float));
    int rows = N;                       // live terms per column

    for (int l = 0; l < st->levels4; ++l) {
        int pairs = rows / 4;
        for (int k = 0; k < pairs; ++k)
            add4_f32(nxt + (size_t)k * N,
                     cur + (size_t)(4 * k) * N, cur + (size_t)(4 * k + 1) * N,
                     cur + (size_t)(4 * k + 2) * N, cur + (size_t)(4 * k + 3) * N, N);
        metrics[MM_ADD_OPS] += (int64_t)pairs * N;
        rows = pairs;
        float* t = cur; cur = nxt; nxt = t;
        if (rows <= 1) break;
    }
    while (rows > 1) {
        int pairs = rows / 2;
        for (int k = 0; k < pairs; ++k)
            add2_f32(nxt + (size_t)k * N,
                     cur + (size_t)(2 * k) * N, cur + (size_t)(2 * k + 1) * N, N);
        metrics[MM_ADD_OPS] += (int64_t)pairs * N;
        rows = pairs;
        float* t = cur; cur = nxt; nxt = t;
    }
    memcpy(out, cur, (size_t)N * sizeof(float));
}

// -- one cycle --------------------------------------------------------------
static void meissa_cycle(MeissaState* st, const float* inject, int has_inject,
                         const float* wgt_push, int has_wgt,
                         float* out_row, int* emitted, int64_t* metrics) {
    const int N = st->N;
    if (N < 1) { *emitted = 0; return; }
    // Shifting moves N-1 columns; stated as a size_t the compiler can bound so
    // it need not assume N could be 0 and the count could wrap.
    const size_t shift_n = (size_t)(N - 1);
    const size_t NN = (size_t)N * N;
    *emitted = 0;

    // One allocation for the whole cycle, carved into disjoint regions:
    //   [0, 2NN)        adder-tree ping-pong workspace
    //   [2NN, 3NN)      this cycle's products
    //   [3NN, 3NN+N)    the reduced row on its way to the banks
    float* const work = mscratch(3 * NN + (size_t)N);
    if (!work) return;
    float* const tree_buf = work;
    float* const prod = work + 2 * NN;
    float* const red = work + 3 * NN;

    // weights shift only while weight_en is held
    if (has_wgt) {
        for (int i = 0; i < N; ++i) {
            float* row = st->wgt + (size_t)i * N;
            if (st->collect_stats)
                for (int j = 0; j < N - 1; ++j)
                    if (row[j] != 0.0f) metrics[MM_WGT_SHIFT_NNZ]++;
            memmove(row + 1, row, shift_n * sizeof(float));
            row[0] = wgt_push[i];
        }
    }

    // activations shift every cycle; column 0 takes a vector or a zero bubble
    for (int i = 0; i < N; ++i) {
        float* row = st->act + (size_t)i * N;
        if (st->collect_stats)
            for (int j = 0; j < N - 1; ++j)
                if (row[j] != 0.0f) metrics[MM_ACT_SHIFT_NNZ]++;
        memmove(row + 1, row, shift_n * sizeof(float));
        row[0] = has_inject ? inject[i] : 0.0f;
    }
    memmove(st->col_seq + 1, st->col_seq, shift_n * sizeof(int64_t));
    if (has_inject) {
        st->col_seq[0] = st->next_seq++;
        if (st->credits > 0) st->credits--;
    } else {
        st->col_seq[0] = -1;
    }

    // retire whatever falls out of the tree ring, then push this cycle in
    const int slot = st->head;
    int64_t* dseq = st->tree_seq + (size_t)slot * N;
    int any_done = 0;
    for (int j = 0; j < N; ++j) if (dseq[j] >= 0) { any_done = 1; break; }
    if (any_done) {
        float* dval = st->tree_vals + (size_t)slot * N;
        if (st->do_reduce == 2)  bf16_round_arr(red, dval, N);
        else if (st->do_reduce)  half_round(red, dval, N);
        else               memcpy(red, dval, (size_t)N * sizeof(float));
        if (st->collect_stats && st->do_reduce) {
            // The reducer narrows FP32 to 16 bits, so a finite sum can leave it
            // as an infinity. That is a modelled outcome worth counting.
            for (int j = 0; j < N; ++j)
                if (dseq[j] >= 0 && std::isfinite(dval[j]) && !std::isfinite(red[j]))
                    metrics[MM_OVERFLOW]++;
        }
        for (int j = 0; j < N; ++j) {
            if (dseq[j] < 0 || st->bank_count[j] >= st->out_depth) continue;
            int pos = (st->bank_head[j] + st->bank_count[j]) % st->out_depth;
            st->bank_vals[(size_t)j * st->out_depth + pos] = red[j];
            st->bank_seq[(size_t)j * st->out_depth + pos] = dseq[j];
            st->bank_count[j]++;
            if (st->collect_stats && red[j] != 0.0f) metrics[MM_OUT_NNZ]++;
        }
    }

    int live = 0;
    for (int j = 0; j < N; ++j) if (st->col_seq[j] >= 0) live++;
    float* sums = st->tree_vals + (size_t)slot * N;
    if (live) {
        mul_f32(prod, st->act, st->wgt, (int)NN);
        metrics[MM_MUL_OPS] += (int64_t)live * N;
        reduce_columns(st, prod, sums, tree_buf, metrics);
    } else {
        memset(sums, 0, (size_t)N * sizeof(float));
    }
    memcpy(st->tree_seq + (size_t)slot * N, st->col_seq, (size_t)N * sizeof(int64_t));
    st->head = (slot + 1) % st->depth;

    // a vector leaves only once every bank holds it -- the de-skew
    int ready = 1;
    int64_t seq0 = -1;
    for (int j = 0; j < N; ++j) {
        if (st->bank_count[j] == 0) { ready = 0; break; }
        int64_t s = st->bank_seq[(size_t)j * st->out_depth + st->bank_head[j]];
        if (j == 0) seq0 = s;
        else if (s != seq0) { ready = 0; break; }
    }
    if (ready) {
        for (int j = 0; j < N; ++j) {
            int h = st->bank_head[j];
            out_row[j] = st->bank_vals[(size_t)j * st->out_depth + h];
            st->bank_head[j] = (h + 1) % st->out_depth;
            st->bank_count[j]--;
        }
        *emitted = 1;
        metrics[MM_OUT_ROWS]++;
        if (st->credits < st->max_credits) st->credits++;
    }

    if (st->collect_stats) {
        int64_t active = 0;
        for (size_t k = 0; k < NN; ++k)
            if (st->act[k] != 0.0f && st->wgt[k] != 0.0f) active++;
        metrics[MM_ACTIVE_PES] += active;
        if (active > metrics[MM_MAX_ACTIVE]) metrics[MM_MAX_ACTIVE] = active;
        if (live) metrics[MM_LIVE_CYCLES]++;
    }
}

// Run `cycles` cycles without returning to Python. inject/wgt_push are
// cycles*N float32 planes; inject_valid/wgt_en are per-cycle flags. Completed
// result vectors land in out_rows (cycles*N) and out_count says how many.
ATALLA_EXPORT void atalla_meissa_run(
        MeissaState* st, int32_t cycles,
        const float* inject, const uint8_t* inject_valid,
        const float* wgt_push, const uint8_t* wgt_en,
        float* out_rows, int32_t* out_count, int64_t* metrics) {
    const int N = st->N;
    for (int i = 0; i < MM_COUNT; ++i) metrics[i] = 0;
    int produced = 0;
    for (int32_t c = 0; c < cycles; ++c) {
        int emitted = 0;
        meissa_cycle(st,
                     inject + (size_t)c * N, inject_valid ? inject_valid[c] : 0,
                     wgt_push + (size_t)c * N, wgt_en ? wgt_en[c] : 0,
                     out_rows + (size_t)produced * N, &emitted, metrics);
        produced += emitted;
    }
    *out_count = produced;
}
