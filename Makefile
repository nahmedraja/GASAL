GPU_SM_ARCH=$(MAKECMDGOALS)
GPU_COMPUTE_ARCH=$(subst sm,compute,$(GPU_SM_ARCH))
NVCC=/usr/local/cuda/bin/nvcc
OBJ_DIR=./obj/
LIB_DIR=./lib/
LOBJS=  gasal.o
LOBJS_PATH=$(addprefix $(OBJ_DIR),$(LOBJS))
VPATH=src:obj:lib

ifeq ($(GPU_SM_ARCH),)
error:
	@echo "Must specify GPU architecture as SM_XX"
endif


ifneq ($(GPU_SM_ARCH),clean)

.SUFFIXES: .cu .c .o .cc .cpp
.cu.o:
	$(NVCC) -c -O3 -Xcompiler -Wall -Xptxas -Werror  --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) -lineinfo --ptxas-options=-v --default-stream per-thread $< -o $(OBJ_DIR)$@

$(GPU_SM_ARCH): makedir libgasal.a

makedir:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(LIB_DIR)
	 

libgasal.a: $(LOBJS)
	ar -csru $(LIB_DIR)$@ $(LOBJS_PATH)
	
endif
	
clean:
	rm -f -r $(OBJ_DIR) $(LIB_DIR) *~ *.exe *.o *.txt *~

gasal.o: gasal.h gasal_kernels_inl.h


