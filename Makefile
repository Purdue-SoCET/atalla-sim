.PHONY: clean-logs native native-clean test

# ---------------------------------------------------------------------------
# Native SIMD kernels
# ---------------------------------------------------------------------------
# One shared object holds AVX-512F, AVX2 and scalar implementations; the ISA is
# chosen at load time via CPUID. Do NOT add -march=native or -ffast-math:
# the kernels must stay bit-identical to the pure-Python reference, which
# means strict IEEE semantics and no FMA contraction.

NATIVE_SRC := src/native/atalla_kernels.cpp
NATIVE_LIB := src/native/libatalla_kernels.so

CXX      ?= g++
CXXFLAGS ?= -O3 -fPIC -shared -std=c++14 -fvisibility=hidden \
            -ffp-contract=off -fno-fast-math -fno-associative-math \
            -Wall -Wextra

native: $(NATIVE_LIB)

# Build to a temp file and rename. Writing the .so in place would corrupt any
# simulation currently running against it, since the library is mmap'd; a
# rename swaps the inode and leaves existing mappings untouched.
$(NATIVE_LIB): $(NATIVE_SRC)
	$(CXX) $(CXXFLAGS) -o $@.tmp $<
	mv -f $@.tmp $@
	@echo "built $@ -> ISA available at runtime:" \
	  "$$(python3 -c 'import ctypes,sys; l=ctypes.CDLL("$@"); l.atalla_isa_name.restype=ctypes.c_char_p; print(l.atalla_isa_name().decode())' 2>/dev/null || echo '?')"

native-clean:
	@rm -f $(NATIVE_LIB)

test: native
	pytest tests/ -q

clean-logs:
	@rm -rf logs/*
