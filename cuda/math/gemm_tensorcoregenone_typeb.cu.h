#ifndef GEMM_TENSORCORE_GEN_ONE_TYPE_B_CU_H
#define GEMM_TENSORCORE_GEN_ONE_TYPE_B_CU_H

#ifdef USE_TENSOR_CORE_GENERATION_ONE
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <stdint.h>

#include "tensorkernel.cu.h"

namespace NSTensorCoreGenOneTypeB
{
 static constexpr int BLOCK_M=64;
 static constexpr int BLOCK_N=64;
 static constexpr int BLOCK_K=32;
 static constexpr int THREADS=256;

 static constexpr int WMMA_M=16;
 static constexpr int WMMA_N=16;
 static constexpr int WMMA_K=16;

 static constexpr int LDA_SMEM=BLOCK_K+8; //40 half
 static constexpr int LDB_SMEM=BLOCK_N+8; //72 half

 static constexpr int C_ELEMS=(BLOCK_M*BLOCK_N)/THREADS; //16

 //----------------------------------------------------------------------------------------------------
 //функция CUDA для умножения тензоров с использованием тензорных ядер первого поколения
 //----------------------------------------------------------------------------------------------------
 template<class type_t,class kernel_output_t,class kernel_left_t,class kernel_right_t>
 __global__ void CUDATensorMulTensorFunction(kernel_output_t tensor_output,kernel_left_t tensor_left,kernel_right_t tensor_right)
 {
  const int M=tensor_left.Size_Y;
  const int N=tensor_right.Size_X;

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

  __shared__ union
  {
   struct
   {
    half sh_A[BLOCK_M][LDA_SMEM];
    half sh_B[BLOCK_K][LDB_SMEM];
   } buf;
   float sh_C[BLOCK_M][BLOCK_N];
  } smem;

  const int tx=threadIdx.x;
  const int ty=threadIdx.y;
  const int tid=ty*16+tx;
  const int warp_id=tid>>5;
  const int warp_m=warp_id>>1; //0..3
  const int warp_n=warp_id&1;  //0..1

  const int block_row=blockIdx.y*BLOCK_M;
  const int block_col=blockIdx.x*BLOCK_N;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,float> acc[2];
  #pragma unroll
  for(int j=0;j<2;++j) nvcuda::wmma::fill_fragment(acc[j],0.0f);

  //раскладка загрузки: A — 4 строки х 2 элемента, B — 2 строки х 2 пары столбцов
  const int a_row0=ty*4;
  const int a_col=tx<<1;
  const int b_row0=ty*2;
  const int b_col=tx<<1;
  const int b_gc0=block_col+b_col;
  const int b_gc1=block_col+32+b_col;

  const int k_steps=(tensor_left.Size_X+BLOCK_K-1)/BLOCK_K;

  for(int ks=0;ks<k_steps;++ks)
  {
   const int k0=ks*BLOCK_K;

   #pragma unroll
   for(int i=0;i<4;++i)
   {
    const int gr=block_row+a_row0+i;
    *reinterpret_cast<half2*>(&smem.buf.sh_A[a_row0+i][a_col])=__floats2half2_rn(tensor_left.GetElement(gr,k0+a_col),tensor_left.GetElement(gr,k0+a_col+1));
   }
   #pragma unroll
   for(int i=0;i<2;++i)
   {
    const int gr=b_row0+i;
    *reinterpret_cast<half2*>(&smem.buf.sh_B[b_row0+i][b_col])=__floats2half2_rn(tensor_right.GetElement(k0+gr,b_gc0),tensor_right.GetElement(k0+gr,b_gc0+1));
    *reinterpret_cast<half2*>(&smem.buf.sh_B[b_row0+i][b_col+32])=__floats2half2_rn(tensor_right.GetElement(k0+gr,b_gc1),tensor_right.GetElement(k0+gr,b_gc1+1));
   }
   __syncthreads();

   #pragma unroll
   for(int kk=0;kk<BLOCK_K;kk+=WMMA_K)
   {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::load_matrix_sync(a_frag,&smem.buf.sh_A[warp_m*WMMA_M][kk],LDA_SMEM);
    #pragma unroll
    for(int j=0;j<2;++j)
    {
     nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,nvcuda::wmma::row_major> b_frag;
     nvcuda::wmma::load_matrix_sync(b_frag,&smem.buf.sh_B[kk][warp_n*32+j*WMMA_N],LDB_SMEM);
     nvcuda::wmma::mma_sync(acc[j],a_frag,b_frag,acc[j]);
    }
   }
   __syncthreads();
  }
  //эпилог: A/B переиспользуется под C
  #pragma unroll
  for(int j=0;j<2;++j)
   nvcuda::wmma::store_matrix_sync(&smem.sh_C[warp_m*WMMA_M][warp_n*32+j*WMMA_N],acc[j],BLOCK_N,nvcuda::wmma::mem_row_major);
  __syncthreads();

  #pragma unroll
  for(int i=0;i<C_ELEMS;++i)
  {
   const int idx=tid+i*THREADS;
   const int r=idx>>6;
   const int c=idx&63;
   const int gr=block_row+r;
   const int gc=block_col+c;
   if (gr<M && gc<N) tensor_output.SetElement(gr,gc,smem.sh_C[r][c]);
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

  dim3 thread(16,16,1);
  dim3 blocks((sTensorKernel_Output.Size_X+BLOCK_N-1)/BLOCK_N,(sTensorKernel_Output.Size_Y+BLOCK_M-1)/BLOCK_M,block_z);
  CUDATensorMulTensorFunction<type_t,kernel_output_t,kernel_left_t,kernel_right_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  cTensor_Output.SetDeviceOnChange();
 }
}

#endif



#endif
