PYTHON ?= python3
COMPILER ?= ../aihw-ppci-compiler/atalla_cc
BUILD_DIR ?= build

.PHONY: all clean
all: $(BUILD_DIR)/elementwise.o

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/elementwise.o: src/elementwise.c kernels.h include/sdma.h | $(BUILD_DIR)
	$(PYTHON) $(COMPILER) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
