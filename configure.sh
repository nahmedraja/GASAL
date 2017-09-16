#!/bin/bash


cuda_path=$1

if [ "$cuda_path" = "" ]; then
  echo "Must provide path to CUDA installation directory"
  echo "Configuration incomplete"
  echo "Exiting"	
  exit 1	
fi	

cuda_nvcc_path=$cuda_path/bin/nvcc

if [ -f $cuda_nvcc_path ]; then
 echo "NVCC found ($cuda_nvcc_path)"
else
  echo "NVCC not found"
  echo "Configuration incomplete"
  echo "Exiting"	
  exit 1	
fi	


cuda_lib_path="${cuda_path}/lib64"


if [ -d $cuda_lib_path ]; then
 echo "CUDA runtime library found (${cuda_lib_path})"
else
  echo "CUDA runtime library not found" 
  echo "Configuration incomplete"
  echo "Exiting"
  exit 1	
fi

cuda_runtime_file="${cuda_path}/include/cuda_runtime.h"

if [ -f $cuda_runtime_file ]; then
 echo  "CUDA runtime header file found (${cuda_runtime_file})"
else
  echo  "CUDA runtime header file not found"
  echo  "Configuration incomplete"
  echo  "Exiting"
  exit 1	
fi


echo "Configuring Makefile_unconfig..."

sed  "s,nvcc,$cuda_nvcc_path," Makefile_unconfig  > Makefile

echo "Configuring gasal_unconfig.h..."

sed  "s,/usr/local/cuda/include/cuda_runtime.h,$cuda_runtime_file," ./src/gasal_unconfig.h  > ./src/gasal.h

mkdir -p ./include

cp ./src/gasal.h ./include

echo "Done"

