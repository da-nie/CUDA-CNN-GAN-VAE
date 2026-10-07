#ifndef GEMM_TENSORCORE_GEN_ONE_TYPE_C_CU_H
#define GEMM_TENSORCORE_GEN_ONE_TYPE_C_CU_H

#ifdef USE_TENSOR_CORE_GENERATION_ONE
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <stdint.h>

#include "tensorkernel.cu.h"

namespace NSTensorCoreGenOneTypeC
{
 static constexpr int BLOCK_M=64;
 static constexpr int BLOCK_N=64;
 static constexpr int BLOCK_K=32;
 static constexpr int THREADS=256;

 static constexpr int WMMA_M=16;
 static constexpr int WMMA_N=16;
 static constexpr int WMMA_K=16;

 static const uint32_t WMMA_TILE_M=4;
 static const uint32_t WMMA_TILE_N=2;
 static const uint32_t WMMA_TILE_K=2;

 static const uint32_t WMMA_BLOCK_ROWS=(WMMA_TILE_M*WMMA_M);
 static const uint32_t WMMA_BLOCK_COLS=(WMMA_TILE_N*WMMA_N);
 static const uint32_t WMMA_BLOCK_DEPTH=(WMMA_TILE_K*WMMA_K);

 static constexpr int LDA_SMEM=BLOCK_K+8; //40 half
 static constexpr int LDB_SMEM=BLOCK_N+8; //72 half

 static constexpr int C_ELEMS=(BLOCK_M*BLOCK_N)/THREADS; //16

 //----------------------------------------------------------------------------------------------------
 //функция CUDA для умножения тензоров с использованием тензорных ядер первого поколения (чуть медленнее варианта A, но корректнее, так как не привязан к магическим цифрам размеров)
 //----------------------------------------------------------------------------------------------------
 template<class type_t, class kernel_output_t, class kernel_left_t, class kernel_right_t>
 __global__ void __launch_bounds__(32u*WMMA_TILE_M*WMMA_TILE_N)
  CUDATensorMulTensorFunction(kernel_output_t tensor_output, kernel_left_t tensor_left, kernel_right_t tensor_right)
 {
  //потоков ровно по числу фрагментов 16x16
  static const int32_t THREAD_COUNT=WMMA_TILE_M*WMMA_TILE_N*32;

  static const int32_t A_ELEMS=WMMA_BLOCK_ROWS*WMMA_BLOCK_DEPTH;
  static const int32_t B_ELEMS=WMMA_BLOCK_DEPTH*WMMA_BLOCK_COLS;
  static const int32_t A_PER_THREAD=A_ELEMS/THREAD_COUNT;
  static const int32_t B_PER_THREAD=B_ELEMS/THREAD_COUNT;

  static_assert(THREAD_COUNT<=1024,"WMMA_TILE_M*WMMA_TILE_N>32: слишком много варпов в блоке");
  static_assert(A_ELEMS%THREAD_COUNT==0 && B_ELEMS%THREAD_COUNT==0,"Блок A/B не делится нацело по потокам");

  uint32_t out_z=Mod(blockIdx.z,tensor_output.Size_Z);
  uint32_t w_out=blockIdx.z/tensor_output.Size_Z;
  uint32_t w_left=Mod(w_out,tensor_left.Size_W);
  uint32_t w_right=Mod(w_out,tensor_right.Size_W);
  w_out=Mod(w_out,tensor_output.Size_W);

  tensor_left.SelectW(w_left);
  tensor_right.SelectW(w_right);
  tensor_output.SelectW(w_out);

  tensor_left.SelectZ(out_z);
  tensor_right.SelectZ(out_z);
  tensor_output.SelectZ(out_z);

  __shared__ half sh_A[WMMA_BLOCK_ROWS][WMMA_BLOCK_DEPTH];
  __shared__ half sh_B_T[WMMA_BLOCK_COLS][WMMA_BLOCK_DEPTH];
  __shared__ float sh_out[WMMA_TILE_M][WMMA_TILE_N][WMMA_M*WMMA_N];

  int32_t block_row=blockIdx.y*WMMA_BLOCK_ROWS;
  int32_t block_col=blockIdx.x*WMMA_BLOCK_COLS;

  int32_t tid=threadIdx.y*blockDim.x+threadIdx.x;

  int32_t warp_id=tid/32;
  int32_t mi=warp_id/WMMA_TILE_N;
  int32_t ni=warp_id-mi*WMMA_TILE_N;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,nvcuda::wmma::row_major> a_frag[WMMA_TILE_K];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,nvcuda::wmma::col_major> b_frag[WMMA_TILE_K];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,float> acc_frag;

  nvcuda::wmma::fill_fragment(acc_frag,0.0f);

  int32_t padded_K=tensor_left.Size_X;
  int32_t padded_N=tensor_output.Size_X;
  int32_t padded_M=tensor_output.Size_Y;

  for(int32_t k_step=0;k_step<padded_K;k_step+=WMMA_BLOCK_DEPTH)
  {
   //матрица A: sh_A[r][c]=A(block_row+r,k_step+c)
   #pragma unroll
   for(int32_t i=0;i<A_PER_THREAD;i++)
   {
    //ВАЖНО: idx=tid+i*THREAD_COUNT, а не tid*PER_THREAD+i —
    //иначе на каждой инструкции варп читает разреженные сектора
    int32_t idx=tid+i*THREAD_COUNT;
    int32_t r=idx/WMMA_BLOCK_DEPTH;
    int32_t c=idx-r*WMMA_BLOCK_DEPTH;
    sh_A[r][c]=__float2half(tensor_left.GetElement(block_row+r,k_step+c));
   }
   //матрица B с транспонированием: sh_B_T[n][k]=B(k_step+k,block_col+n)
   #pragma unroll
   for(int32_t i=0;i<B_PER_THREAD;i++)
   {
    int32_t idx=tid+i*THREAD_COUNT;
    int32_t k=idx/WMMA_BLOCK_COLS;
    int32_t n=idx-k*WMMA_BLOCK_COLS;
    sh_B_T[n][k]=__float2half(tensor_right.GetElement(k_step+k,block_col+n));
   }
   __syncthreads();
   //умножение строки на столбец с суммированием
   #pragma unroll
   for(int32_t ki=0;ki<WMMA_TILE_K;ki++)
   {
    nvcuda::wmma::load_matrix_sync(a_frag[ki],&sh_A[mi*WMMA_M][ki*WMMA_K],WMMA_BLOCK_DEPTH);
    //ldm равен WMMA_BLOCK_DEPTH: шаг между столбцами n в sh_B_T;
    //при TILE_N*16==TILE_K*16 совпадает с WMMA_BLOCK_COLS, как в вашей версии
    nvcuda::wmma::load_matrix_sync(b_frag[ki],&sh_B_T[ni*WMMA_N][ki*WMMA_K],WMMA_BLOCK_DEPTH);
    nvcuda::wmma::mma_sync(acc_frag,a_frag[ki],b_frag[ki],acc_frag);
   }
   __syncthreads();
  }
  //считывание результата
  nvcuda::wmma::store_matrix_sync(sh_out[mi][ni],acc_frag,WMMA_N,nvcuda::wmma::mem_row_major);
  __syncthreads();
  //переписываем результат в выходную матрицу
  for(int32_t idx=tid;idx<WMMA_BLOCK_ROWS*WMMA_BLOCK_COLS;idx+=THREAD_COUNT)
  {
   int32_t r=idx/WMMA_BLOCK_COLS;
   int32_t c=idx-r*WMMA_BLOCK_COLS;
   int32_t cy=block_row+r;
   int32_t cx=block_col+c;
   if (cy<padded_M && cx<padded_N)
   {
    int32_t local_mi=r/WMMA_M;
    int32_t local_ni=c/WMMA_N;
    int32_t local_r=r-local_mi*WMMA_M;
    int32_t local_c=c-local_ni*WMMA_N;
    tensor_output.SetElement(cy,cx,sh_out[local_mi][local_ni][local_r*WMMA_N+local_c]);
   }
  }
 }
 //----------------------------------------------------------------------------------------------------
 //умножить тензоры
 //----------------------------------------------------------------------------------------------------
 template<class type_t,class kernel_output_t,class kernel_left_t,class kernel_right_t>
 __host__ void MulAbstract(CTensor<type_t> &cTensor_Output,kernel_output_t &sTensorKernel_Output,const CTensor<type_t> &cTensor_Left,kernel_left_t &sTensorKernel_Left,const CTensor<type_t> &cTensor_Right,kernel_right_t &sTensorKernel_Right)
 {
  if (sTensorKernel_Left.Size_X!=sTensorKernel_Right.Size_Y  || sTensorKernel_Left.Size_Z!=sTensorKernel_Right.Size_Z ||
      sTensorKernel_Output.Size_Y!=sTensorKernel_Left.Size_Y || sTensorKernel_Output.Size_X!=sTensorKernel_Right.Size_X ||
      sTensorKernel_Output.Size_Z!=sTensorKernel_Right.Size_Z)
  {
   throw "CTensor::MulAbstract: Размерности тензоров не совпадают!";
  }

  //копируем данные с устройство
  cTensor_Left.CopyToDevice();
  cTensor_Right.CopyToDevice();

  uint32_t block_z=sTensorKernel_Output.Size_Z*sTensorKernel_Output.Size_W;
  if (block_z==0) block_z=1;

  //ровно WMMA_TILE_M*WMMA_TILE_N варпов; для (4,2,2) это те же 256 потоков
  dim3 thread(32*WMMA_TILE_N,WMMA_TILE_M,1);
  dim3 blocks((sTensorKernel_Output.Size_X+WMMA_BLOCK_COLS-1)/WMMA_BLOCK_COLS,(sTensorKernel_Output.Size_Y+WMMA_BLOCK_ROWS-1)/WMMA_BLOCK_ROWS,block_z);

  CUDATensorMulTensorFunction<type_t,kernel_output_t,kernel_left_t,kernel_right_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  cTensor_Output.SetDeviceOnChange();
 }
}

#endif



#endif
