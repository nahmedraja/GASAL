# GASAL: A GPU Accelerated Sequnce Alignment Library for High-throughput NGS DATA

GASAL2 is an easy to use CUDA library for DNA/RNA sequence alignment algorithms. Currently it the following sequence alignment functions.
- Local alignment without start position computation. Gives alignment score and end position of the alignment.
- Local alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Semi-global alignment without start position computation. Gives score and end position of the alignment.
- Semi-global alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Global alignment.
The GASAL alignment functions are blocking, i.e. the user has to wait for the alignment on the GPU. For non-blocking alignment to overlap CPU and GPU execution use GASAL2 (https://github.com/nahmedraja/GASAL2).

## Requirements
CUDA toolkit 8 or higher. May be 7 will also work, but not tested yet. 

## Compiling GASAL
To compile the library, run the following two commands:

```
$ ./configure.sh <path to cuda installation directory>
$ make GPU_SM_ARCH=<GPU SM architecture> MAX_LEN=<maximum sequence length> [N_SCORE=<penalty for aligning "N" against any other base>]
```

`N_SCORE` is optional and if it is not specified then GASAL2 considers "N" as an ordinary base having the same match/mismatch scores as for A, C, G or T. As a result of these commands, *include* and *lib* directories will be created containing `gasal.h` and `libgasal.a`, respectively. Include `gasal.h` in your code and link it with `libgasal.a` during compilation. Also link the CUDA runtime library by adding `-lcudart` flag. The path to the CUDA runtime library must also be specfied while linking as *-L <path to CUDA lib64 directory>*. In default CUDA installation on Linux machines the path is */usr/local/cuda/lib64*.

## Using GASAL
To use GASAL  alignment functions, first the match/mismatach scores and gap open/extension penalties need to be passed on to the GPU. Assign the values match/mismatach scores and gap open/extension penalties to the members of `gasal_subst_scores` struct

```
typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;e
	int32_t gap_extend;
}gasal_subst_scores;
```

The values are passed to the GPU by calling `gasal_copy_subst_scores()` function

```
void gasal_copy_subst_scores(gasal_subst_scores *subst);
```

Then memory is allocated on the GPU by using the following function:

```
void gasal_gpu_mem_alloc(gasal_gpu_storage_t *gpu_storage, int gpu_max_batch1_bytes, int gpu_max_batch2_bytes, int gpu_max_n_alns, int algo, int start);
```

In GASAL, the sequences to be alignned are conatined in two batches i.e. a sequence in batch1 is aligned to sequence in batch2. A *batch* is a concatenation of sequences. *The number of bases in each sequence must a multiple of 8*. At the same time the user Hence, if a sequence is not a multiple of 8, `N's` are added at the end of sequence. We call these redundant bases as *Pad bases*. With the help of `gpu_max_batch1_bytes` `gpu_max_batch2 _bytes` the user specifies the expected maxumum size (in bytes) of the two sequence batches. If the actual required GPU memory is more than the pre-allocated memory, GASAL automatically allocates more memory. The type of sequence alignment algorithm is specfied using `algo` parameter. Pass one of the follwing three values as the `algo` parameter:

```
LOCAL
GLOBAL
SEMI_GLOBAL
```

Similarly, to perform alignment with or without start position computation, pass one the following two values in the `start` parameter:

```
WITHOUT_START
WITH_START
```

To free up the allocated memory the following function is used:

```
void gasal_gpu_mem_free(gasal_gpu_storage_t *gpu_storage);
```

The `gasal_gpu_mem_alloc()` and `gasal_gpu_mem_free()` internally use `cudaMalloc()` and `cudaFree()` functions. These CUDA API functions are time expensive. Therefore, `gasal_gpu_mem_alloc()` and `gasal_gpu_mem_free()` should be preferably called only once in the program.

The alignment on the GPU can be performed by calling the following function:

```
void gasal_aln(gasal_gpu_storage_t *gpu_storage, const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens,   const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end,  int algo, int start);
```

where `batch1` and `batch2` contain the sequences. As described before, the length of each sequence in the batch must be a multiple of 8. `batch1_offsets` and `batch2_offsets` contain the starting point of sequences in the batch that are required to be aligned. These offset values include the pad bases, and hence always multiple of 8. `batch1_lens` and `batch2_lens` are the original length of sequences i.e. excluding pad bases. The `actual_batch1_bytes` and `actual_batch2_bytes` specify the size of the two batches (in bytes) including the pad bases. `actual_n_alns` is the number of alignments to be performed. The result of the alignment is in `host_*` arrays. The user allocates/de-allocates the memory for `host_*` arrays on the CPU. A `NULL` is passed for unused result arrays. From the performance prespective, if the average lengths of the sequences in *batch1* and *batch2* are not same, then the shorter sequences should be placed in *batch1*. Forexample, in case of read mappers the query sequences are conatined in batch1 and the genome sequences in batch2.

## Problems and suggestions
For any problems or suggestions contact Nauman Ahmed (n.ahmed@tudelft.nl)
