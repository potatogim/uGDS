KERNEL ?= /lib/modules/$(shell uname -r)/build
NVIDIA_DIR ?= $(shell find /usr/src -maxdepth 1 -type d -regextype posix-extended -regex '.*/nvidia-[0-9]+(\.[0-9]+)+' 2>/dev/null | sort -V | tail -1)
export NVIDIA_DIR

# Backend selection (mutually exclusive):
#   BUILD_HIP=1  → HIP/ROCm only (DMA-buf, no NVIDIA headers required)
#   default      → CUDA (requires NVIDIA driver source)
ifeq ($(BUILD_HIP),1)
export BUILD_HIP
endif

ifneq ($(KERNELRELEASE),)
	obj-m := ugds_drv.o
	ugds_drv-objs := pci.o list.o ctrl.o map.o

	# AMD HIP path — standard Linux DMA-buf, no external headers needed
 ifdef BUILD_HIP
	ccflags-y += -D_HIP
 else
	# NVIDIA CUDA path — fail loudly if headers missing
 ifndef NVIDIA_DIR
$(error NVIDIA driver source not found in /usr/src. Install nvidia-source or set BUILD_HIP=1 for AMD HIP backend)
 endif
	ccflags-y += -D_CUDA -I$(NVIDIA_DIR)/nvidia
	KBUILD_EXTRA_SYMBOLS += $(NVIDIA_DIR)/Module.symvers
 endif
else

.PHONY: default reload unload load clean

default:
	$(MAKE) -C $(KERNEL) M=$(CURDIR) NVIDIA_DIR=$(NVIDIA_DIR) modules

clean:
	$(MAKE) -C $(KERNEL) M=$(CURDIR) clean

reload: unload load

unload:
	-rmmod ugds_drv

load:
	insmod ugds_drv.ko max_num_ctrls=64

endif
