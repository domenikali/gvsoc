SHELL := /bin/bash

REDMULE_PATH ?= $(shell echo $$REDMULE_PATH)
PULP_RISCV_GCC_TOOLCHAIN ?= $(shell echo $$PULP_RISCV_GCC_TOOLCHAIN)
GVSOC_PATH ?= $(shell echo $$GVSOC_PATH)
PULP_SDK_PATH ?= $(shell echo $$PULP_SDK_PATH)
TEST_PATH ?= $(shell echo $$TEST_PATH)

run:
	export PULP_RISCV_GCC_TOOLCHAIN=$(PULP_RISCV_GCC_TOOLCHAIN)
	export GVSOC_PATH=$(GVSOC_PATH)

	source $(PULP_SDK_PATH)/configs/pulp-open.sh; $(MAKE) -C $(GVSOC_PATH) all TARGETS=pulp-open-ddr; $(MAKE) -C $(TEST_PATH) clean; $(MAKE) -C $(TEST_PATH)

	mkdir -p reports

	cd reports; $(GVSOC_PATH)/install/bin/gvsoc --target=pulp-open-ddr --binary $(TEST_PATH)/BUILD/PULP/GCC_RISCV/test/test image flash run --trace=redmule

diff:
	cd .. && \
	git diff ../../../add_dramsyslib_patches/redmule/ . | grep "new file mode " -B 1 | GREP_COLORS='ms=1;32' grep "diff " --color=always && \
	git diff ../../../add_dramsyslib_patches/redmule/ . | grep "delete file mode " -B 1 | GREP_COLORS='ms=1;31' grep "diff " --color=always && \
	git diff ../../../add_dramsyslib_patches/redmule/ . | grep "diff " -A 1 | GREP_COLORS='ms=1;34' grep -v "file mode" --color=always && \
	echo "Done"

clean:
	rm -rf reports
	rm -rf ../test/BUILD
