# GASAL: A GPU Accelerated Sequnce Alignment Library for High-throughput NGS DATA

GASAL is an easy to use CUDA library for DNA/RNA sequence alignment algorithms. Currently it the following sequence alignment functions.
- Local alignment without start position computation. Gives alignment score and end position of the alignment.
- Local alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Semi-global alignment without start position computation. Gives score and end position of the alignment.
- Semi-global alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Global alignment.

## Requirements
CUDA toolkit 8 or higher. May be 7 will also work, but not tested yet. 

## Compiling GASAL
To compile the library, run the following two commands following commands:

```
$ ./configure.sh <path to cuda installation directory>
$ make GPU_SM_ARCH=<GPU SM architecture> MAX_LEN=<maximum sequence length> N_CODE=<code for "N", e.g. 0x4E if the bases are represented by ASCII characters> [N_PENALTY=<penalty for aligning "N" against any other base>]
```

`N_PENALTY` is optional and if it is not specified then GASAL considers "N" as an ordinary base having the same match/mismatch scores as for A, C, G or T. As a result of these commands, *include* and *lib* directories will be created containing `gasal.h` and `libgasal.a`, respectively. Include `gasal.h` in your code. Link `libgasal.a` with your code. Also link the CUDA runtime library by adding `-lcudart` flag. The path to the CUDA runtime library must also be specfied while linking as *-L <path to CUDA lib64 directory>*. In default CUDA installation on Linux machines the path is */usr/local/cuda/lib64*.

## Using GASAL
To use GASAL  alignment functions, first the match/mismatach scores and gap open/extension penalties need to be passed on to the GPU. Assign the values match/mismatach scores and gap open/extension penalties to the members of `gasal_subst_scores` struct

```
typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
}gasal_subst_scores;
```

The values are passed to the GPU by calling `gasal_copy_subst_scores()` function

```
void gasal_copy_subst_scores(gasal_subst_scores *subst);
```

Then memory is allocated on the GPU by using the following function.

```
void gasal_gpu_mem_alloc(gasal_gpu_storage_t *gpu_storage, int max_query_batch_bytes, int max_target_batch_bytes, int max_n_alns, int algo, int start);
```

In GASAL, the sequences to be alignned are conatined in two batches i.e. a sequence in query_batch is aligned to sequence in target_batch. A *batch* is a concatenation of sequences. *The number of bases in each sequence must a multiple of 8*. Hence, if a sequence is not a multiple of 8, `N's` are added at the end of sequence. We call these redundant bases as *Pad bases*. Note that the pad bases are always "N's" irrespective of whether `N_PENALTY` is defined or not. With the help of `max_query_batch_bytes` `max_target_batch_bytes` the user specifies the expected maxumum size(in bytes) of sequences in the two batches. If the actual required GPU memory is more than the pre-allocated memory, GASAL automatically allocates more memory. The type of sequence alignment algorithm is specfied using `algo` parameter. Pass one of the follwing three values as the `algo` parameter:

```
LOCAL
GLOBAL
SEMI_GLOBAL
```

Similarly, to perform alignment with or without start position, computation is specfied by passing one the following two values in the `start` parameter:

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
void gasal_aln(gasal_gpu_storage_t *gpu_storage, const uint8_t *query_batch, const uint32_t *query_batch_offsets, const uint32_t *query_batch_lens, const uint8_t *target_batch, const uint32_t *target_batch_offsets, const uint32_t *target_batch_lens,   const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_query_batch_start, int32_t *host_target_batch_start, int32_t *host_query_batch_end, int32_t *host_target_batch_end,  int algo, int start);
```

where `query_batch` and `target_batch` conatin the sequences. `query_batch_offsets` and `target_batch_offsets` contain the starting point of sequences in the batch that are required to be aligned. These offset values include the pad bases, and hence always multiple of 8. `query_batch_lens` and `target_batch_lens` are the original length of sequences i.e. excluding pad bases. The `actual_query_batch_bytes` and `actual_target_batch_bytes` specify the size of the two batches (in bytes) including the pad bases. `actual_n_alns` is the number of alignments to be performed. The result of the alignment is in `host_*` arrays. The user allocates/de-allocates the memory for `host_*` arrays on the CPU. A `NULL` is passed for unused result arrays. From the performance prespective, if the average lengths of the sequences in *query_batch* and *target_batch* are not same, then the shorter sequences should be placed in *query_batch*. Forexample, in case of read mappers the read sequences are conatined in query_batch and the genome sequences in target_batch.
