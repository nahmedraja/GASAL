#include "gasal.h"


enum system_type{
	HOST,
	GPU
};

#define CUDA_ERROR_CHECK



#define CUDAMALLOCCHECK(error, system) \
		err = error;\
	 if (cudaSuccess != err ) { \
	 	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
	 	 exit(EXIT_FAILURE);\
	  }

#define CUDAMEMCPYCHECK(error, copy_to) \
		err = error;\
	 if (cudaSuccess != err ) { \
	 	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
	 	 exit(EXIT_FAILURE);\
	  }

#define CUDAMEMCPYTOSYMBOLCHECK(error) \
		err = error;\
		if (cudaSuccess != err ) { \
			fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
			exit(EXIT_FAILURE);\
		}

#define CUDAMEMFREECHECK(error) \
		err = error;\
		if (cudaSuccess != err ) { \
			fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
			exit(EXIT_FAILURE);\
		}
#define CUDASTREAMCREATEANDDESTROYCHECK(error) \
		err = error;\
		if (cudaSuccess != err ) { \
			fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
			exit(EXIT_FAILURE);\
		}
#define CUDASTREAMQUERYCHECK(error) \
		err = error;\
		if (cudaSuccess != err ) { \
			if (err == cudaErrorNotReady) return -1; \
			else{\
			fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
			exit(EXIT_FAILURE);\
			}\
		}

#define CUDASETDEVICECHECK(error) \
		err = error;\
		if (cudaSuccess != err ) { \
			fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
			exit(EXIT_FAILURE);\
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

gasal_error_t gasal_aln(const uint8_t *batch1, const uint32_t *batch1_lens, const uint32_t *batch1_offsets, const uint8_t *batch2, const uint32_t *batch2_lens, const uint32_t *batch2_offsets,  const uint32_t n_alns, gasal_gpu_storage *gpu_storage, const uint32_t batch1_bytes, const uint32_t batch2_bytes, int algo, int start) {

	cudaError_t err;
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

	CUDASTREAMCREATEANDDESTROYCHECK(cudaStreamCreate(&str))


	CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->unpacked1), batch1_bytes * sizeof(uint8_t)), GPU)
	CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->unpacked2), batch2_bytes * sizeof(uint8_t)), GPU)

	CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->packed1_4bit), (batch1_bytes/8) * sizeof(uint32_t)), GPU)
	CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->packed2_4bit), (batch2_bytes/8) * sizeof(uint32_t)), GPU)

	CUDAMEMCPYCHECK(cudaMemcpyAsync(gpu_storage->unpacked1, batch1, batch1_bytes, cudaMemcpyHostToDevice, str), GPU)
	CUDAMEMCPYCHECK(cudaMemcpyAsync(gpu_storage->unpacked2, batch2, batch2_bytes, cudaMemcpyHostToDevice, str), GPU)


    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM, 0, str>>>((uint32_t*)(gpu_storage->unpacked1),
    						(uint32_t*)(gpu_storage->unpacked2), gpu_storage->packed1_4bit, gpu_storage->packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, batch1_bytes/4, batch2_bytes/4);
    cudaError_t pack_kernel_err = cudaGetLastError();
    if ( cudaSuccess != pack_kernel_err )
    {
    	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
         exit(EXIT_FAILURE);
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
		if (start == WITH_START){
			CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->batch2_start), n_alns * sizeof(uint32_t)), GPU)
		}
		else gpu_storage->batch2_start = NULL;
		if (algo == LOCAL) {
			CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->batch1_end), n_alns * sizeof(uint32_t)), GPU)
			if (start == WITH_START){
				CUDAMALLOCCHECK(cudaMalloc(&(gpu_storage->batch1_start), n_alns * sizeof(uint32_t)), GPU)
			}
			else gpu_storage->batch1_start = NULL;
		} else {
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch1_end = NULL;
		}
	}



    if (start == WITH_START) {
		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM, 0, str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, gpu_storage->batch1_start,
				gpu_storage->batch2_start, n_alns);
	} else {
		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM, 0, str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, n_alns);
	}

    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    gpu_storage->str = str;
    return 0;
}




gasal_error_t gasal_get_aln_results(gasal_gpu_storage *gpu_storage, uint32_t n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end) {

	cudaError_t err;
	CUDASTREAMQUERYCHECK(cudaStreamQuery(gpu_storage->str))

	if (host_aln_score != NULL && gpu_storage->aln_score != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_aln_score, gpu_storage->aln_score, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost), HOST)
	if (host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_batch1_start, gpu_storage->batch1_start, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost), HOST)
	if (host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_batch2_start, gpu_storage->batch2_start, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost), HOST)
	if (host_batch1_end != NULL && gpu_storage->batch1_end != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_batch1_end, gpu_storage->batch1_end, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost), HOST)
	if (host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CUDAMEMCPYCHECK(cudaMemcpy(host_batch2_end, gpu_storage->batch2_end, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost), HOST)

	if (gpu_storage->unpacked1 != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->unpacked1))
	if (gpu_storage->unpacked2 != NULL) CUDAMEMFREECHECK(cudaFree(gpu_storage->unpacked2))
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

	CUDASTREAMCREATEANDDESTROYCHECK(cudaStreamDestroy(gpu_storage->str))

	return 0;
}


gasal_error_t gasal_init(gasal_subst_scores *subst, int dev_id = 0){

	cudaError_t err;
	CUDASETDEVICECHECK(cudaSetDevice(dev_id))
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaGapO, &(subst->gap_open), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaGapExtend, &(subst->gap_extend), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	int32_t gapoe = subst->gap_open + subst->gap_extend;
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaGapOE, &(gapoe), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaMatchScore, &(subst->match), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CUDAMEMCPYTOSYMBOLCHECK(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst->mismatch), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	return 0;
}













