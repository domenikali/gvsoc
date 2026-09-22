include ../utils/rt.mk

BUILDDIR = $(CURDIR)/build
CLUSTER_BINARY = $(BUILDDIR)/cluster
CLUSTER_FLAGS = -march=rv32imfdv -mabi=ilp32 -O2 -Tcluster_link.ld -nostartfiles -nostdlib -Wl,--no-warn-rwx-segments

clean:
	rm -f *.o $(BUILDDIR)/test
	make -C ../../../.. TARGETS=my_system MODULES=$(CURDIR) clean

all:
	mkdir -p $(BUILDDIR)
	riscv64-unknown-elf-gcc -g -o $(BUILDDIR)/test main.c cluster_dma.c $(RT_SRCS) $(RT_FLAGS)
	riscv64-unknown-elf-gcc -g -o $(CLUSTER_BINARY) cluster_main.c cluster_start.S \
		$(CLUSTER_FLAGS)

gvsoc:
	mkdir -p ../../../build/core
	make -C ../../../.. TARGETS=my_system MODULES=$(CURDIR) build

prepare:
	cp solution/* .

run:
	../../../..//install/bin/gvsoc --target-dir=$(CURDIR) --target=my_system --work-dir=$(BUILDDIR) \
		--binary=$(BUILDDIR)/test --cluster-binary=$(CLUSTER_BINARY) run $(runner_args)