#include "gasal.h"




#define CHECKCUDAERROR(error) \
		do{\
			err = error;\
			if (cudaSuccess != err ) { \
				fprintf(stderr, "[GASAL] CUDA error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
				exit(EXIT_FAILURE);\
			}\
		}while(0)\


inline int CudaCheckKernelLaunch()
{
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		return -1;

	}

	return 0;
}




#include "gasal_kernels_inl.h"





//GASAL2 blocking alignment function
void gasal_aln(gasal_gpu_storage_t *gpu_storage, const uint8_t *query_batch, const uint32_t *query_batch_offsets, const uint32_t *query_batch_lens, const uint8_t *target_batch, const uint32_t *target_batch_offsetss, const uint32_t *target_batch_lens, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_query_batch_start, int32_t *host_target_batch_start, int32_t *host_query_batch_end, int32_t *host_target_batch_end,  int algo, int start) {

	cudaError_t err;
	if (actual_n_alns <= 0) {
		fprintf(stderr, "[GASAL] Must perform at least 1 alignment (n_alns > 0)\n");
		exit(EXIT_FAILURE);
	}
	if (actual_query_batch_bytes <= 0) {
		fprintf(stderr, "[GASAL] Number of query_batch_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}
	if (actual_target_batch_bytes <= 0) {
		fprintf(stderr, "[GASAL] Number of target_batch_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}

	if (actual_query_batch_bytes % 8) {
		fprintf(stderr, "[GASAL] Number of query_batch_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (actual_target_batch_bytes % 8) {
		fprintf(stderr, "[GASAL] Number of target_batch_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);

	}
	//--------------if pre-allocated memory is less, allocate more--------------------------
	if (gpu_storage->max_query_batch_bytes < actual_query_batch_bytes) {
		fprintf(stderr, "[GASAL]  max_query_batch_bytes(%d) should be >= acutal_query_batch_bytes(%d)\n", gpu_storage->max_query_batch_bytes, actual_query_batch_bytes);

		int i = 2;
		while ( (gpu_storage->max_query_batch_bytes * i) < actual_query_batch_bytes) i++;

		fprintf(stderr, "[GASAL] Allocating %d bytes on GPU\n", gpu_storage->max_query_batch_bytes*i);
		gpu_storage->max_query_batch_bytes = gpu_storage->max_query_batch_bytes * i;

		if (gpu_storage->unpacked_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_query_batch));
		if (gpu_storage->packed_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_query_batch));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_query_batch), gpu_storage->max_query_batch_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_query_batch), (gpu_storage->max_query_batch_bytes/8) * sizeof(uint32_t)));




	}

	if (gpu_storage->max_target_batch_bytes < actual_target_batch_bytes) {
		fprintf(stderr, "max_target_batch_bytes(%d) should be >= acutal_target_batch_bytes(%d)\n", gpu_storage->max_target_batch_bytes, actual_target_batch_bytes);

		int i = 2;
		while ( (gpu_storage->max_target_batch_bytes * i) < actual_target_batch_bytes) i++;

		fprintf(stderr, "[GASAL] Allocating %d bytes on GPU\n", gpu_storage->max_target_batch_bytes*i);
		gpu_storage->max_target_batch_bytes = gpu_storage->max_target_batch_bytes * i;

		if (gpu_storage->unpacked_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_target_batch));
		if (gpu_storage->packed_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_target_batch));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_target_batch), gpu_storage->max_target_batch_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_target_batch), (gpu_storage->max_target_batch_bytes/8) * sizeof(uint32_t)));


	}

	if (gpu_storage->max_n_alns < actual_n_alns) {
		fprintf(stderr, "[GASAL] max_n_alns should >= acutal_n_alns(%d)\n", gpu_storage->max_n_alns, actual_n_alns);

		int i = 2;
		while ( (gpu_storage->max_n_alns * i) < actual_n_alns) i++;

		fprintf(stderr, "[GASAL] Allocating GPU memory for %d number of alignments\n", gpu_storage->max_n_alns*i);
		gpu_storage->max_n_alns = gpu_storage->max_n_alns * i;

		if (gpu_storage->query_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_offsets));
		if (gpu_storage->target_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_offsets));
		if (gpu_storage->query_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_lens));
		if (gpu_storage->target_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_lens));
		if (gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->aln_score));
		if (gpu_storage->query_batch_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_start));
		if (gpu_storage->target_batch_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_start));
		if (gpu_storage->query_batch_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_end));
		if (gpu_storage->target_batch_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_end));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_lens), gpu_storage->max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_lens), gpu_storage->max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_offsets), gpu_storage->max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_offsets), gpu_storage->max_n_alns * sizeof(uint32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->aln_score),gpu_storage->max_n_alns * sizeof(int32_t)));
		if (algo == GLOBAL) {
			gpu_storage->query_batch_start = NULL;
			gpu_storage->query_batch_end = NULL;
			gpu_storage->target_batch_start = NULL;
			gpu_storage->target_batch_end = NULL;
		} else {
			CHECKCUDAERROR(
					cudaMalloc(&(gpu_storage->target_batch_end),
							gpu_storage->max_n_alns * sizeof(uint32_t)));
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->target_batch_start),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
			} else
				gpu_storage->target_batch_start = NULL;
			if (algo == LOCAL) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->query_batch_end),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
				if (start == WITH_START) {
					CHECKCUDAERROR(
							cudaMalloc(&(gpu_storage->query_batch_start),
									gpu_storage->max_n_alns * sizeof(uint32_t)));
				} else
					gpu_storage->query_batch_start = NULL;
			} else {
				gpu_storage->query_batch_start = NULL;
				gpu_storage->query_batch_end = NULL;
			}
		}



	}
	//-------------------------------------------------------------------------------------------

	//------------------------copy sequence batches from CPU to GPU---------------------------
	CHECKCUDAERROR(cudaMemcpy(gpu_storage->unpacked_query_batch, query_batch, actual_query_batch_bytes, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(gpu_storage->unpacked_target_batch, target_batch, actual_target_batch_bytes, cudaMemcpyHostToDevice));
	//----------------------------------------------------------------------------------------

    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int query_batch_tasks_per_thread = (int)ceil((double)actual_query_batch_bytes/(8*BLOCKDIM*N_BLOCKS));
    int target_batch_tasks_per_thread = (int)ceil((double)actual_target_batch_bytes/(8*BLOCKDIM*N_BLOCKS));

    //launch packing kernel
    gasal_pack_kernel<<<N_BLOCKS, BLOCKDIM>>>((uint32_t*)(gpu_storage->unpacked_query_batch),
    						(uint32_t*)(gpu_storage->unpacked_target_batch), gpu_storage->packed_query_batch, gpu_storage->packed_target_batch,
    					    query_batch_tasks_per_thread, target_batch_tasks_per_thread, actual_query_batch_bytes/4, actual_target_batch_bytes/4);
    cudaError_t pack_kernel_err = cudaGetLastError();
    if ( cudaSuccess != pack_kernel_err )
    {
    	 fprintf(stderr, "[GASAL] CUDA error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
         exit(EXIT_FAILURE);
    }

    //----------------------copy sequence offsetss and lengths from CPU to GPU--------------------------------------
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->query_batch_lens, query_batch_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->target_batch_lens, target_batch_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->query_batch_offsets, query_batch_offsets, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->target_batch_offsets, target_batch_offsetss, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    //------------------------------------------------------------------------------------------------------------------------

    //--------------------------------------launch alignment kernels--------------------------------------------------------------
    if(algo == LOCAL) {
    	if (start == WITH_START) {
    		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,
    				gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score,
    				gpu_storage->query_batch_end, gpu_storage->target_batch_end, gpu_storage->query_batch_start,
    				gpu_storage->target_batch_start, actual_n_alns);
    	} else {
    		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,
    				gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score,
    				gpu_storage->query_batch_end, gpu_storage->target_batch_end, actual_n_alns);
    	}
    } else if (algo == SEMI_GLOBAL) {
    	if (start == WITH_START) {
    		gasal_semi_global_with_start_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,
    				gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score, gpu_storage->target_batch_end,
    				gpu_storage->target_batch_start, actual_n_alns);
    	} else {
    		gasal_semi_global_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,
    				gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score, gpu_storage->target_batch_end,
    				actual_n_alns);
    	}

    } else if (algo == GLOBAL) {
    	gasal_global_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,
    			gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score, actual_n_alns);
    }
    else {
    	fprintf(stderr, "[GASAL] Algo type invalid\n");
    	exit(EXIT_FAILURE);
    }
    //-----------------------------------------------------------------------------------------------------------------------
    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "[GASAL] CUDA error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    //------------------------copy alignment results from GPU to CPU--------------------------------------
    if (host_aln_score != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpy(host_aln_score, gpu_storage->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    else {
    	fprintf(stderr, "[GASAL] The *host_aln_score input can't be NULL\n");
    	exit(EXIT_FAILURE);
    }
    if (host_query_batch_start != NULL && gpu_storage->query_batch_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_query_batch_start, gpu_storage->query_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_target_batch_start != NULL && gpu_storage->target_batch_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_target_batch_start, gpu_storage->target_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_query_batch_end != NULL && gpu_storage->query_batch_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_query_batch_end, gpu_storage->query_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_target_batch_end != NULL && gpu_storage->target_batch_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_target_batch_end, gpu_storage->target_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    //------------------------------------------------------------------------------------------------------

}






void gasal_gpu_mem_alloc(gasal_gpu_storage_t *gpu_storage, int max_query_batch_bytes, int max_target_batch_bytes, int max_n_alns, int algo, int start) {

	cudaError_t err;
	if (gpu_storage->max_query_batch_bytes % 8) {
		fprintf(stderr, "max_query_batch_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (gpu_storage->max_target_batch_bytes % 8) {
		fprintf(stderr, "max_target_batch_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_query_batch), max_query_batch_bytes * sizeof(uint8_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_target_batch), max_target_batch_bytes * sizeof(uint8_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_query_batch), (max_query_batch_bytes/8) * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_target_batch), (max_target_batch_bytes/8) * sizeof(uint32_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_lens), max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_lens), max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_offsets), max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_offsets), max_n_alns * sizeof(uint32_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->aln_score), max_n_alns * sizeof(int32_t)));
	if (algo == GLOBAL) {
		gpu_storage->query_batch_start = NULL;
		gpu_storage->query_batch_end = NULL;
		gpu_storage->target_batch_start = NULL;
		gpu_storage->target_batch_end = NULL;
	} else {
		CHECKCUDAERROR(
				cudaMalloc(&(gpu_storage->target_batch_end),
						max_n_alns * sizeof(uint32_t)));
		if (start == WITH_START) {
			CHECKCUDAERROR(
					cudaMalloc(&(gpu_storage->target_batch_start),
							max_n_alns * sizeof(uint32_t)));
		} else
			gpu_storage->target_batch_start = NULL;
		if (algo == LOCAL) {
			CHECKCUDAERROR(
					cudaMalloc(&(gpu_storage->query_batch_end),
							max_n_alns * sizeof(uint32_t)));
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->query_batch_start),
								max_n_alns * sizeof(uint32_t)));
			} else
				gpu_storage->query_batch_start = NULL;
		} else {
			gpu_storage->query_batch_start = NULL;
			gpu_storage->query_batch_end = NULL;
		}
	}

	gpu_storage->max_query_batch_bytes = max_query_batch_bytes;
	gpu_storage->max_target_batch_bytes = max_target_batch_bytes;
	gpu_storage->max_n_alns = max_n_alns;

}




void gasal_gpu_mem_free(gasal_gpu_storage_t *gpu_storage) {

	cudaError_t err;

	if (gpu_storage->unpacked_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_query_batch));
	if (gpu_storage->unpacked_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_target_batch));
	if (gpu_storage->packed_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_query_batch));
	if (gpu_storage->packed_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_target_batch));
	if (gpu_storage->query_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_offsets));
	if (gpu_storage->target_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_offsets));
	if (gpu_storage->query_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_lens));
	if (gpu_storage->target_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_lens));
	if (gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->aln_score));
	if (gpu_storage->query_batch_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_start));
	if (gpu_storage->target_batch_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_start));
	if (gpu_storage->query_batch_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_end));
	if (gpu_storage->target_batch_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_end));

}


void gasal_copy_subst_scores(gasal_subst_scores *subst){

	cudaError_t err;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapO, &(subst->gap_open), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtend, &(subst->gap_extend), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	int32_t gapoe = subst->gap_open + subst->gap_extend;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOE, &(gapoe), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMatchScore, &(subst->match), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst->mismatch), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	return;
}




