#include "gasal.h"


enum system_type{
	HOST,
	GPU
};

#define CUDA_ERROR_CHECK



#define CUDAMALLOCCHECK(err, system) \
 if (cudaSuccess != err ) { \
	 if (system == GPU) return -1; \
	 else return -2; \
 }

#define CUDAMEMCPYCHECK(err, copy_to) \
 if (cudaSuccess != err ) { \
	 if(copy_to == GPU) return -3; \
	 else return -4;\
 }

#define CUDAMEMCPYTOSYMBOLCHECK(err) \
 if (cudaSuccess != err ) { \
	 return -5; \
 }

#define CUDAMEMFREECHECK(err) \
 if (cudaSuccess != err ) { \
	 return -6; \
 }

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


// The gasal local alignment function without start position computation
template <int algo, int start>
gasal_error_t gasal_aln(const uint8_t *batch1, const uint32_t *batch1_lens, const uint32_t *batch1_offsets, const uint8_t *batch2, const uint32_t *batch2_lens, const uint32_t *batch2_offsets,  const uint32_t n_alns, gasal_gpu_storage *gpu_storage, const uint64_t batch1_bytes, const uint64_t batch2_bytes) {


	if (n_alns <= 0) {
		std::cerr << "Number of alignments should be greater than 0" << std::endl;
		exit(EXIT_FAILURE);
	}
	if (batch1_bytes <= 0) {
		std::cerr << "Number of batch1_bytes should be greater than 0" << std::endl;
		exit(EXIT_FAILURE);
	}
	if (batch2_bytes <= 0) {
		std::cerr << "Number of batch2_bytes should be greater than 0" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (batch1_bytes % 8) {
		std::cerr << "Number of batch1_bytes should be multiple of 8" << std::endl;
		exit(EXIT_FAILURE);
	}
	if (batch2_bytes % 8) {
		std::cerr << "Number of batch2_bytes should be multiple of 8" << std::endl;
		exit(EXIT_FAILURE);
	}



	cudaStream_t str;

	cudaError_t error = cudaStreamCreate(&str);
	if(error != cudaSuccess) return -12;


	CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->unpacked1), batch1_bytes * sizeof(uint8_t)), GPU)
	CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->unpacked2), batch2_bytes * sizeof(uint8_t)), GPU)

	CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->packed1_4bit), (batch1_bytes/8) * sizeof(uint32_t)), GPU)
	CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->packed2_4bit), (batch2_bytes/8) * sizeof(uint32_t)), GPU)

	CUDAMEMCPYCHECK(cudaMemcpyAsync(gpu_storage->unpacked1, batch1, batch1_bytes, cudaMemcpyHostToDevice, str), GPU)
	CUDAMEMCPYCHECK(cudaMemcpyAsync(gpu_storage->unpacked1, batch2, batch2_bytes, cudaMemcpyHostToDevice, str), GPU)


    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM, 0, str>>>((uint32_t*)gpu_storage->unpacked1,
    						(uint32_t*)gpu_storage->unpacked2, gpu_storage->packed1_4bit, gpu_storage->packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, batch1_bytes/4, batch2_bytes/4);
    if (CudaCheckKernelLaunch() == -1) {
    	return -7;
    }



    CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->lens1), n_alns * sizeof(uint32_t)), GPU)
    CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->lens2), n_alns * sizeof(uint32_t)), GPU)
    CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->offsets1), n_alns * sizeof(uint32_t)), GPU)
    CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->offsets2), n_alns * sizeof(uint32_t)), GPU)

    CUDAMEMCPYCHECK(cudaMemcpyAsync(gpu_storage->lens1, batch1_lens, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, str), GPU)
    CUDAMEMCPYCHECK(cudaMemcpyAsync(gpu_storage->lens2, batch2_lens, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, str), GPU)
    CUDAMEMCPYCHECK(cudaMemcpyAsync(gpu_storage->offsets1, batch1_offsets, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, str), GPU)
    CUDAMEMCPYCHECK(cudaMemcpyAsync(gpu_storage->offsets2, batch2_offsets, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, str), GPU)





	CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->aln_score), n_alns * sizeof(int32_t)), GPU)
	if (algo == GLOBAL) {
		gpu_storage->batch1_start = NULL;
		gpu_storage->batch1_end = NULL;
		gpu_storage->batch2_start = NULL;
		gpu_storage->batch2_end = NULL;
	} else {
		CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->batch2_end), n_alns * sizeof(uint32_t)), GPU)
		if (start == WITH_START)
			CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->batch2_start), n_alns * sizeof(uint32_t)), GPU)
			else gpu_storage->batch2_start = NULL;
		if (algo == LOCAL) {
			CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->batch1_end), n_alns * sizeof(uint32_t)), GPU)
			if (start == WITH_START)
				CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->batch1_start), n_alns * sizeof(uint32_t)), GPU)
				else gpu_storage->batch1_start = NULL;
		} else {
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch1_end = NULL;
		}
	}



    if (WITH_START) {
		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM, 0, str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, gpu_storage->batch1_start,
				gpu_storage->batch2_start, n_alns);
	} else {
		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM, 0, str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, n_alns);
	}

    if (CudaCheckKernelLaunch() == -1) {
       	return -8;
    }

    gpu_storage->str = str;
    return 0;
}




gasal_error_t gasal_get_aln_results(gasal_gpu_storage *gpu_storage, uint32_t n_alns, int32_t *host_aln_score = NULL, int32_t *host_batch1_start = NULL, int32_t *host_batch2_start = NULL, int32_t *host_batch1_end = NULL, int32_t *host_batch2_end = NULL) {

	cudaError_t q = cudaStreamQuery(gpu_storage->str);

	if (q != cudaSuccess) {
		if ( q == cudaErrorNotReady) return -9;
		else return -15;

	}


	if (host_aln_score != NULL && gpu_storage->aln_score != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_aln_score, gpu_storage->aln_score, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice), HOST)
	if (host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_batch1_start, gpu_storage->batch1_start, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice), HOST)
	if (host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_batch2_start, gpu_storage->batch2_start, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice), HOST)
	if (host_batch1_end != NULL && gpu_storage->batch2_end != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_batch1_end, gpu_storage->batch1_end, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice), HOST)
	if (host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_batch2_end, gpu_storage->batch2_end, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice), HOST)

	if (gpu_storage->unpacked1 != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->unpacked1))
	if (gpu_storage->unpacked1 != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->unpacked2))
	if (gpu_storage->packed1_4bit != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->packed1_4bit))
	if (gpu_storage->packed2_4bit != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->packed2_4bit))
	if (gpu_storage->offsets1 != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->offsets1))
	if (gpu_storage->offsets2 != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->offsets2))
	if (gpu_storage->lens1 != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->lens1))
	if (gpu_storage->lens2 != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->lens2))
	if (gpu_storage->aln_score != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->aln_score))
	if (gpu_storage->batch1_start != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->batch1_start))
	if (gpu_storage->batch2_start != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->batch2_start))
	if (gpu_storage->batch1_end != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->batch1_end))
	if (gpu_storage->batch2_end != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->batch2_end))

	return 0;
}


gasal_error_t gasal_init(gasal_subst_scores *subst, int dev_id = 0){

	cudaError_t error = cudaSetDevice(dev_id);
	if(error != cudaSuccess) return -11;
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaGapO, &(subst->gap_open), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaGapExtend, &(subst->gap_extend), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	int32_t gapoe = subst->gap_open + subst->gap_extend;
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaGapOE, &(gapoe), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaMatchScore, &(subst->match), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst->mismatch), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	return 0;
}













