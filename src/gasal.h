#ifndef __GASAL_H__
#define __GASAL_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "/usr/local/cuda-8.0/include/cuda_runtime.h"


#ifndef HOST_MALLOC_SAFETY_FACTOR
#define HOST_MALLOC_SAFETY_FACTOR 5
#endif


enum comp_start{
	WITH_START,
	WITHOUT_START
};

enum algo_type{
	LOCAL,
	GLOBAL,
	SEMI_GLOBAL
};


//kernel data
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
	uint32_t gpu_max_batch1_bytes;
	uint32_t gpu_max_batch2_bytes;
	uint32_t host_max_batch1_bytes;
	uint32_t host_max_batch2_bytes;
	uint32_t gpu_max_n_alns;

} gasal_gpu_storage_t;



//match/mismatch and gap penalties
typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
} gasal_subst_scores;



#ifdef __cplusplus
extern "C" {
#endif


void gasal_aln(gasal_gpu_storage_t *gpu_storage, const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens,   const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end,  int algo, int start);

void gasal_gpu_mem_alloc(gasal_gpu_storage_t *gpu_storage, int gpu_max_batch1_bytes, int gpu_max_batch2_bytes, int gpu_max_n_alns, int algo, int start);

void gasal_gpu_mem_free(gasal_gpu_storage_t *gpu_storage);

void gasal_copy_subst_scores(gasal_subst_scores *subst);



#ifdef __cplusplus
}
#endif

#endif
