# GASAL: A GPU Accelerated Sequnce Alignment Library for High-throughput NGS DATA

GASAL is an esay to use CUDA library for DNA/RNA sequence alignment algorithms. Currently it the following sequence alignment functions.
- Local alignment without start position computation. Gives alignment score and end position of the alignment.
- Local alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Semi-global alignment without start position computation. Gives score and end position of the alignment.
- Semi-global alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Global alignment.

## Compiling GASAL
To compile the library, specify the compute capability using the *GPU_ARCH* variable (default value *sm_35*) and then execute the following commands:

```
$ ./configure.sh <path to cuda installation directory>
$ make
```

*inclue* and *lib* directories will be crtaed containing `gasal.h` and `libgasal.a`, respectively. Include `gasal.h` in your code. Link `libgasal.a` with your code. Also link the CUDA runtime library by adding `-lcudart` flag. The path to the CUDA runtime library must also be specfied while linking as *-L <path to CUDA lib64 directory>*. In default CUDA installation on Linux machines the path is */usr/local/cuda/lib64*.

## Using GASAL
To use GASAL alignment functions, first the match/mismatach scores and gap open/extension penalties need to be passed on to the GPU. Assign the values match/mismatach scores and gap open/extension penalties to the members of `gasal_subst_scores` struct

```
typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
} gasal_subst_scores;
```
The values are passed to the GPU by calling `gasal_copy_subst_scores()` function

```
void gasal_copy_subst_scores(gasal_subst_scores *subst);
```

To align sequences with GASAL batches of sequences are passed to aliognment function. A batch is a concatenation of sequences. *The number of bases in each sequence must a multiple of 8*. To do this add redundant bases at the end of the seequnces, e.g. A's. We call these redundant bases as *Pad bases* Alignment can be performed by calling one of the follwing two functions:

```
void gasal_aln(const uint8_t *batch1, const uint32_t *batch1_lens, const uint32_t *batch1_offsets, const uint8_t *batch2, const uint32_t *batch2_lens, const uint32_t *batch2_offsets,  const uint32_t n_alns, const uint32_t batch1_bytes, const uint32_t batch2_bytes, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end, int algo, int start);

gasal_gpu_storage* gasal_aln_async(const uint8_t *batch1, const uint32_t *batch1_lens, const uint32_t *batch1_offsets, const uint8_t *batch2, const uint32_t *batch2_lens, const uint32_t *batch2_offsets,  const uint32_t n_alns, const uint32_t batch1_bytes, const uint32_t batch2_bytes, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end, int algo, int start);
```

`batch1` and `batch2` are the concatenation of sequences to be aligned. `batch1_offsets` and `batch2_offsets` contain the starting point of sequences in the batch that are required to be aligned. These offset values include the pad bases, and hence always multiple of 8. `batch1_lens` and `batch2_lens` are the original length of sequences i.e. excluding pad bases. `batch1_bytes` and `batch2_bytes` specify the size of the two batches (in bytes) including the pad bases. `n_alns` is the number of alignments to be performed. The type of sequence alignment algorithm is specfied using `algo` parameter. Pass one of the follwing three values as the `algo` parameter:

```
LOCAL
GLOBAL
SEMI_GLOBAL
```

Similarly, to perform alignment with or without start position computation is specfied by passing one the following two values in the `start` parameter:

```
WITHOUT_START
WITH_START
```


The result of alignments are stored in `host_\*` arrays. In cases where one or more results are not required, pass `NULL` as the parameter. Note that `n_alns = |batch1_offsets| = |batch2_offsets| = |batch1_lens| = |batch2_lens| = |host_\*|`.


The `void gasal_aln()` function returns only after the alignment on the GPU is finished and `host_\*` arrays contain valid result of the alignment. In contrast, the `gasal_aln_async()` function immediately returns control to the CPU after launching the alignment kernel on the GPU. This allows the user thread to do other useful work instead of waiting for the alignment kernel to finish. The *async* function returns the pointer to `gasal_gpu_strorage` struct. To test whether the alignment on GPU is finished and the  `host_\*` arrays contain valid results, a call to the following function is required to be made:

```
gasal_error_t is_gasal_aln_async_done(gasal_gpu_storage *gpu_storage);
```
If the function returns `0` the alignment on the GPU is finished and the  `host_\*` arrays contain valid results.


