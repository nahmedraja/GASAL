sm_35:
	 nvcc -c -std=c++11 -O3 -Xcompiler -Wall -Xptxas -Werror  --gpu-architecture=compute_35 --gpu-code=sm_35 -lineinfo --ptxas-options=-v --default-stream per-thread  -o gasal.o gasal.cu

clean:
	rm -f *.exe *.o *.txt *~
