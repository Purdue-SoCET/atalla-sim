PYTHON ?= python3
COMPILER ?= ../aihw-ppci-compiler/atalla_cc
BUILD_DIR ?= build

.PHONY: all clean
all: $(BUILD_DIR)/elementwise.o $(BUILD_DIR)/matmul.o $(BUILD_DIR)/spatial.o

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/elementwise.o: src/elementwise.c kernels.h include/sdma.h | $(BUILD_DIR)
	PYTHONPATH=../aihw-ppci-compiler $(PYTHON) $(COMPILER) -O0 -p -c $< -o $@ > $@.log 2>&1

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR)/matmul.o: src/matmul.c kernels.h include/sdma.h | $(BUILD_DIR)
	PYTHONPATH=../aihw-ppci-compiler $(PYTHON) $(COMPILER) -p -c $< -o $@ > $@.log 2>&1

$(BUILD_DIR)/spatial.o: src/spatial.c kernels.h include/sdma.h | $(BUILD_DIR)
	PYTHONPATH=../aihw-ppci-compiler $(PYTHON) $(COMPILER) -O0 -p -c $< -o $@ > $@.log 2>&1
