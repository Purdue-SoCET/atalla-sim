.PHONY: clean-logs native native-clean ramulator ramulator-clean test

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

# ---------------------------------------------------------------------------
# Ramulator 2 DRAM timing (optional; tools/run_kernel.py --dram ramulator)
# ---------------------------------------------------------------------------
# Builds the third_party/ramulator2 submodule's library and the small C
# interface memory/ramulator.py loads. Ramulator needs C++20 (g++ 12 or
# newer): the newest gcc-toolset under /opt/rh is used when there is one;
# set RAMULATOR_CXX to choose another compiler.

RAMULATOR_DIR := third_party/ramulator2
RAMULATOR_SRC := src/native/atalla_ramulator.cpp
RAMULATOR_LIB := src/native/libatalla_ramulator.so
RAMULATOR_CXX ?= $(or $(lastword $(sort $(wildcard /opt/rh/gcc-toolset-*/root/usr/bin/g++))),g++)

ramulator: $(RAMULATOR_LIB)

$(RAMULATOR_DIR)/libramulator.so:
	@test -f $(RAMULATOR_DIR)/CMakeLists.txt || \
	  { echo "$(RAMULATOR_DIR) is empty: git submodule update --init $(RAMULATOR_DIR)"; exit 1; }
	cmake -S $(RAMULATOR_DIR) -B $(RAMULATOR_DIR)/build -DCMAKE_BUILD_TYPE=Release \
	  -DRAMULATOR_PYTHON_BINDINGS=OFF -DCMAKE_CXX_COMPILER=$(RAMULATOR_CXX)
	cmake --build $(RAMULATOR_DIR)/build --parallel

$(RAMULATOR_LIB): $(RAMULATOR_SRC) $(RAMULATOR_DIR)/libramulator.so
	$(RAMULATOR_CXX) -O2 -fPIC -shared -std=c++20 -fvisibility=hidden -Wall -Wextra \
	  -isystem $(RAMULATOR_DIR)/src -isystem $(RAMULATOR_DIR)/ext/fmt/include \
	  -isystem $(RAMULATOR_DIR)/ext/yaml-cpp/include -o $@.tmp $< \
	  -L$(RAMULATOR_DIR) -lramulator -Wl,-rpath,$(abspath $(RAMULATOR_DIR))
	mv -f $@.tmp $@

ramulator-clean:
	@rm -f $(RAMULATOR_LIB)

test: native
	pytest tests/ -q

clean-logs:
	@rm -rf logs/*
