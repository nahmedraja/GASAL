#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "/usr/local/cuda/include/cuda_runtime.h"

#ifndef MAX_BATCH1_LEN
#define MAX_BATCH1_LEN 304
#endif

#ifndef MAX_BATCH2_LEN
#define MAX_BATCH2_LEN 600
#endif

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN (MAX_BATCH1_LEN > MAX_BATCH2_LEN ? MAX_BATCH1_LEN : MAX_BATCH2_LEN)
#endif

typedef int32_t gasal_error_t;


typedef struct {
	uint8_t *unpacked1;
	uint8_t *unpacked2;
	uint32_t *packed1_4bit;
	uint32_t *packed2_4bit;
	uint32_t *offsets1;
	uint32_t *offsets2;
	uint32_t *lens1;
	uint32_t *lens2;
	int32_t *aln_score;
	int32_t *batch1_end;
	int32_t *batch2_end;
	int32_t *batch1_start;
	int32_t *batch2_start;
	cudaStream_t str;

} gasal_gpu_storage;

typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
} gasal_subst_scores;

enum comp_start{
	WITH_START,
	WITHOUT_START
};

enum algo_type{
	LOCAL,
	GLOBAL,
	SEMI_GLOBAL
};

#ifdef __cplusplus
extern "C" {
#endif

void gasal_aln(const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens, const uint32_t batch1_bytes, const uint32_t batch2_bytes, const uint32_t n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end, int algo, int start);

gasal_gpu_storage* gasal_aln_async(const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens,   const uint32_t batch1_bytes, const uint32_t batch2_bytes, const uint32_t n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end,  int algo, int start);

gasal_error_t gasal_is_aln_async_done(gasal_gpu_storage *gpu_storage);

void gasal_copy_subst_scores(gasal_subst_scores *subst);

void gasal_host_malloc(void *mem_ptr, uint32_t n_bytes);

void gasal_host_free(void *mem_ptr);

#ifdef __cplusplus
}
#endif

