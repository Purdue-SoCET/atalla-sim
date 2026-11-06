PYTHON ?= python3
BUILD_DIR ?= build
SOURCES := elementwise matmul spatial spatial_tiles
OBJECTS := $(addprefix $(BUILD_DIR)/,$(addsuffix .o,$(SOURCES)))

.PHONY: all clean
all: $(OBJECTS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: src/%.c kernels.h include/sdma.h include/spatial.h compile.py | $(BUILD_DIR)
	$(PYTHON) compile.py $< -O 2 -o $@ > $@.log 2>&1

clean:
	rm -rf $(BUILD_DIR)
