#include "cuda.cu.h"

#include <cstdio>
#include <ctime>
#include <stdio.h>
#include <stdint.h>

#include "handle_error.cu.h"
#include "netmodel/cmodel_sorter.cu.h"
#include "netmodel/cmodel_gan.cu.h"
#include "netmodel/cmodel_srgan.cu.h"
#include "netmodel/cmodel_vae.cu.h"
#include "../system/system.h"

//----------------------------------------------------------------------------------------------------
//главная функция программы на CUDA
//----------------------------------------------------------------------------------------------------

void CUDA_Start(void)
{
/*
struct cudaDeviceProp
{
    char   name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    size_t memPitch;
    int    maxThreadsPerBlock;
    int    maxThreadsDim [3];
    int    maxGridSize   [3];
    size_t totalConstMem;
    int    major;
    int    minor;
    int    clockRate;
    size_t textureAlignment;
    int    deviceOverlap;
    int    multiProcessorCount;
}
*/


 int deviceCount;
 cudaDeviceProp devProp;

 HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
 printf("Found %d devices\n",deviceCount);
 for(int device=0;device<deviceCount;device++)
 {
  char str[1024];
  HANDLE_ERROR(cudaGetDeviceProperties(&devProp,device));
  sprintf(str,"Device %d\n", device );
  SYSTEM::PutMessageToConsole(str);
  sprintf(str,"Compute capability     : %lu.%lu",devProp.major,devProp.minor);
  SYSTEM::PutMessageToConsole(str);
  sprintf(str,"Name                   : %s",devProp.name);
  SYSTEM::PutMessageToConsole(str);
  sprintf(str,"Total Global Memory    : %lu",devProp.totalGlobalMem);
  SYSTEM::PutMessageToConsole(str);
  sprintf(str,"Shared memory per block: %lu",devProp.sharedMemPerBlock);
  SYSTEM::PutMessageToConsole(str);
  sprintf(str,"Registers per block    : %lu",devProp.regsPerBlock);
  SYSTEM::PutMessageToConsole(str);
  sprintf(str,"Warp size              : %lu",devProp.warpSize);
  SYSTEM::PutMessageToConsole(str);
  sprintf(str,"Max threads per block  : %lu",devProp.maxThreadsPerBlock);
  SYSTEM::PutMessageToConsole(str);

  sprintf(str,"Max Grid Size: %lux%lux%lu",devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
  SYSTEM::PutMessageToConsole(str);

  sprintf(str,"Max Threads dim: %lux%lux%lu",devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
  SYSTEM::PutMessageToConsole(str);


  SYSTEM::PutMessageToConsole(str);
  sprintf(str,"Total constant memory  : %lu",devProp.totalConstMem);
  SYSTEM::PutMessageToConsole(str);
 }
 HANDLE_ERROR(cudaSetDevice(0));
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceReset());
 HANDLE_ERROR(cudaGetLastError());

 //CModelSorter<float> cModelSorter;
 //cModelSorter.Execute();

 //CModelGAN<float> cModelGAN;
 //cModelGAN.Execute();

 //CModelVAE<float> cModelVAE;
 //cModelVAE.Execute();

 CModelSR_GAN<float> cModelSR_GAN;
 cModelSR_GAN.Execute();
}
