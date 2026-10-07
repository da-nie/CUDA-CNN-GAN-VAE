#ifndef GEMM_CUDACORE_TYPE_B_CU_H
#define GEMM_CUDACORE_TYPE_B_CU_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <stdint.h>

#include "tensorkernel.cu.h"

namespace NSCUDACoreTypeB
{
 static const uint32_t TENSOR_OPERATION_TILE_SIZE_X=128;///<размер блока операций с тензорами по X
 static const uint32_t TENSOR_OPERATION_TILE_SIZE_Y=128;///<размер блока операций с тензорами по Y
 static const uint32_t TENSOR_OPERATION_TILE_SIZE_K=16;///<размер блока операций с тензорами по общей стороне K
 static const uint32_t WORK_PER_TILE_SIZE_X=8;///<сколько элементов обрабатывает поток по X
 static const uint32_t WORK_PER_TILE_SIZE_Y=8;///<сколько элементов обрабатывает поток по Y
 static const uint32_t PART_WORK_PER_TILE_X=TENSOR_OPERATION_TILE_SIZE_X/WORK_PER_TILE_SIZE_X;///<количество обрабатываемых блоков по WORK_PER_TILE_SIZE_X элементов
 static const uint32_t PART_WORK_PER_TILE_Y=TENSOR_OPERATION_TILE_SIZE_Y/WORK_PER_TILE_SIZE_Y;///<количество обрабатываемых блоков по WORK_PER_TILE_SIZE_Y элементов
 static const uint32_t LPTA=(TENSOR_OPERATION_TILE_SIZE_K*WORK_PER_TILE_SIZE_Y*WORK_PER_TILE_SIZE_X)/(TENSOR_OPERATION_TILE_SIZE_X);//The amount of loads-per-thread for A
 static const uint32_t LPTB=(TENSOR_OPERATION_TILE_SIZE_K*WORK_PER_TILE_SIZE_Y*WORK_PER_TILE_SIZE_X)/(TENSOR_OPERATION_TILE_SIZE_Y);//The amount of loads-per-thread for B

 //----------------------------------------------------------------------------------------------------
 //функция CUDA для умножения тензоров
 //----------------------------------------------------------------------------------------------------
 template<class type_t,class kernel_output_t,class kernel_left_t,class kernel_right_t>
 __global__ void CUDATensorMulTensorFunction(kernel_output_t tensor_output,kernel_left_t tensor_left,kernel_right_t tensor_right)
 {
  const uint32_t tid_x=threadIdx.x;//Local x ID (max: TSN/WPTN == RTSN)
  const uint32_t tid_y=threadIdx.y;//Local y ID (max: TSM/WPTM == RTSM)
  const uint32_t offset_x=TENSOR_OPERATION_TILE_SIZE_X*blockIdx.x;//Work-group offset
  const uint32_t offset_y=TENSOR_OPERATION_TILE_SIZE_Y*blockIdx.y;//Work-group offset
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

  //Local memory to fit a tile of A and B
  volatile __shared__ type_t a_sub[TENSOR_OPERATION_TILE_SIZE_K][TENSOR_OPERATION_TILE_SIZE_Y];
  volatile __shared__ type_t b_sub[TENSOR_OPERATION_TILE_SIZE_X][TENSOR_OPERATION_TILE_SIZE_K+2];

  //Allocate register space
  type_t a_reg;
  type_t b_reg[WORK_PER_TILE_SIZE_X];
  type_t acc[WORK_PER_TILE_SIZE_Y][WORK_PER_TILE_SIZE_X];

  //Initialise the accumulation registers
  #pragma unroll
  for(uint32_t wy=0;wy<WORK_PER_TILE_SIZE_Y;wy++)
  {
   #pragma unroll
   for(uint32_t wx=0;wx<WORK_PER_TILE_SIZE_X;wx++) acc[wy][wx]=0;
  }

  //Loop over all tiles
  uint32_t numTiles=tensor_left.Size_X/TENSOR_OPERATION_TILE_SIZE_K;
  if (tensor_left.Size_X%(TENSOR_OPERATION_TILE_SIZE_K)) numTiles++;

  for(uint32_t t=0;t<numTiles;t++)
  {
   //Load one tile of A and B into local memory
   #pragma unroll
   for(uint32_t la=0;la<LPTA;la++)
   {
    uint32_t tid=tid_x*PART_WORK_PER_TILE_Y+tid_y;
    uint32_t id=la*PART_WORK_PER_TILE_X*PART_WORK_PER_TILE_Y+tid;
    uint32_t x=id/TENSOR_OPERATION_TILE_SIZE_Y;
    uint32_t y=id-x*TENSOR_OPERATION_TILE_SIZE_Y;
    uint32_t tiled_index=TENSOR_OPERATION_TILE_SIZE_K*t+x;
    a_sub[x][y]=tensor_left.GetElement(offset_y+y,tiled_index);//A[tiled_index*M+offset_y+y];
    b_sub[y][x]=tensor_right.GetElement(tiled_index,offset_x+y);//B[tiled_index*N+offset_x+y];
   }

   //Synchronise to make sure the tile is loaded
   __syncthreads();

   //Loop over the values of a single tile
   for(uint32_t k=0;k<TENSOR_OPERATION_TILE_SIZE_K;k++)
   {
    //Cache the values of b_sub in registers
    #pragma unroll
    for(uint32_t wx=0;wx<WORK_PER_TILE_SIZE_X;wx++)
    {
     uint32_t x=tid_x+wx*PART_WORK_PER_TILE_X;
     b_reg[wx]=b_sub[x][k];
    }
    //Perform the computation
    #pragma unroll
    for(uint32_t wy=0;wy<WORK_PER_TILE_SIZE_Y;wy++)
    {
     uint32_t y=tid_y+wy*PART_WORK_PER_TILE_Y;
     a_reg=a_sub[k][y];
     #pragma unroll
     for(uint32_t wx=0;wx<WORK_PER_TILE_SIZE_X;wx++)
     {
     //printf("%f x %f\r\n",a_reg,b_reg[wx]);
      acc[wy][wx]+=a_reg*b_reg[wx];
     }
    }
   }
   //Synchronise before loading the next tile
   __syncthreads();
  //Next tile
  }

  //Store the final results in C
  #pragma unroll
  for(uint32_t wy=0;wy<WORK_PER_TILE_SIZE_Y;wy++)
  {
   uint32_t global_y=offset_y+tid_y+wy*PART_WORK_PER_TILE_Y;
   #pragma unroll
   for(uint32_t wx=0;wx<WORK_PER_TILE_SIZE_X;wx++)
   {
    uint32_t global_x=offset_x+tid_x+wx*PART_WORK_PER_TILE_X;
    tensor_output.SetElement(global_y,global_x,acc[wy][wx]);
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
  //копируем данные на устройство
  cTensor_Left.CopyToDevice();
  cTensor_Right.CopyToDevice();

  dim3 thread(PART_WORK_PER_TILE_X,PART_WORK_PER_TILE_Y);

  uint32_t scale_x=WORK_PER_TILE_SIZE_X*PART_WORK_PER_TILE_X;
  uint32_t block_x=sTensorKernel_Right.Size_X/scale_x;
  if (sTensorKernel_Right.Size_X%scale_x) block_x++;
  uint32_t scale_y=WORK_PER_TILE_SIZE_Y*PART_WORK_PER_TILE_Y;
  uint32_t block_y=sTensorKernel_Left.Size_Y/scale_y;
  if (sTensorKernel_Left.Size_Y%scale_y) block_y++;
  uint32_t block_z=sTensorKernel_Output.Size_Z*sTensorKernel_Output.Size_W;

  dim3 blocks(block_x,block_y,block_z);
  if (blocks.x==0) blocks.x=1;
  if (blocks.y==0) blocks.y=1;
  if (blocks.z==0) blocks.z=1;

  CUDATensorMulTensorFunction<type_t,kernel_output_t,kernel_left_t,kernel_right_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  cTensor_Output.SetDeviceOnChange();
 }
}

#endif
