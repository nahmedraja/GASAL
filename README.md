# GASAL: A GPU Accelerated Sequnce Alignment Library for High-throughput NGS DATA

GASAL is an esay to use CUDA library for sequence alignment algorithms. Currently it the following sequence alignment functions.
- Local alignment without start position computation. Gives alignment score and end position of the alignment.
- Local alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Semi-global alignment without start position computation. Gives score and end position of the alignment.
- Semi-global alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Global alignment.

## Compiling GASAL
To compile the library first specify the compute capability using the *GPU_ARCH* variable in the Makefile. Its default value is *sm_35*. Run make. It will produce *libgasal.a*. Include *gasal.h* file in the code. While compling the code link *libgasal.a* in the linking step. Also link the CUDA runtime library by adding *-lcudart* flag. The path to the CUDA runtime library must also be specfied while linking as *-L <path to CUDA lib64 directory>*. In default CUDA installation on Linux machines the path is */usr/local/cuda/lib64*.
