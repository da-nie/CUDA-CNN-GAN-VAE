#ifndef GEMM_CUDACORE_TYPE_A_CU_H
#define GEMM_CUDACORE_TYPE_A_CU_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <stdint.h>

#include "tensorkernel.cu.h"

namespace NSCUDACoreTypeA
{
 static const uint32_t TENSOR_MUL_TILE_SIZE_SCALE=2;///<размер множителя блока операции умножения с тензорами
 static const uint32_t TENSOR_MUL_TILE_BLOCK_SIZE=16;///<размер блока операции умножения с тензорами
 static const uint32_t TILE_BLOCK_SIZE=16;///<размер блока операций с тензорами

 //----------------------------------------------------------------------------------------------------
 //функция CUDA для умножения тензоров
 //----------------------------------------------------------------------------------------------------
 template<class type_t,class kernel_output_t,class kernel_left_t,class kernel_right_t>
 __global__ void CUDATensorMulTensorFunction(kernel_output_t tensor_output,kernel_left_t tensor_left,kernel_right_t tensor_right)
 {
  //блок TENSOR_MUL_TILE_BLOCK_SIZE x TENSOR_MUL_TILE_BLOCK_SIZE в выходном тензоре
  uint32_t block_x=blockIdx.x;
  uint32_t block_y=blockIdx.y;
  //координаты элементов блока в выходном тензоре
  uint32_t out_in_block_x=threadIdx.x*TENSOR_MUL_TILE_SIZE_SCALE;
  uint32_t out_in_block_y=threadIdx.y*TENSOR_MUL_TILE_SIZE_SCALE;
  //координаты блока в выходном тензоре
  uint32_t out_block_x=TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE*block_x;
  uint32_t out_block_y=TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE*block_y;
  //глобальные координаты в выходном тензоре
  uint32_t out_x=out_in_block_x+out_block_x;
  uint32_t out_y=out_in_block_y+out_block_y;

  uint32_t out_z=Mod(blockIdx.z,tensor_output.Size_Z);

  uint32_t w_out=blockIdx.z/tensor_output.Size_Z;
  uint32_t w_left=Mod(w_out,tensor_left.Size_W);
  uint32_t w_right=Mod(w_out,tensor_right.Size_W);
  w_out=Mod(w_out,tensor_output.Size_W);

  //получаем подматрицу выходной матрицы
  type_t Cvalue[TENSOR_MUL_TILE_SIZE_SCALE][TENSOR_MUL_TILE_SIZE_SCALE];
  for(uint32_t ky=0;ky<TENSOR_MUL_TILE_SIZE_SCALE;ky++)
  {
   for(uint32_t kx=0;kx<TENSOR_MUL_TILE_SIZE_SCALE;kx++)
   {
    Cvalue[ky][kx]=0;
   }
  }

  tensor_left.SelectW(w_left);
  tensor_right.SelectW(w_right);
  tensor_output.SelectW(w_out);

  tensor_left.SelectZ(out_z);
  tensor_right.SelectZ(out_z);
  tensor_output.SelectZ(out_z);

  //считаем, сколькно нужно проходов блоком по X
  uint32_t m_max=tensor_left.Size_X/(TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE);
  if (tensor_left.Size_X%(TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE)) m_max++;

  volatile __shared__ type_t As[TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE][TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE];
  volatile __shared__ type_t Bs[TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE][TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE];

  for(uint32_t m=0;m<m_max;m++)
  {
   uint32_t offset=m*TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE;
   uint32_t py_left=out_in_block_y+out_block_y;
   uint32_t py_right=out_in_block_y+offset;
   uint32_t py=out_in_block_y;
   for(uint32_t ky=0;ky<TENSOR_MUL_TILE_SIZE_SCALE;ky++,py_left++,py_right++,py++)
   {
    uint32_t px_left=out_in_block_x+offset;
    uint32_t px_right=out_in_block_x+out_block_x;
    uint32_t px=out_in_block_x;
    for(uint32_t kx=0;kx<TENSOR_MUL_TILE_SIZE_SCALE;kx++,px_left++,px_right++,px++)
    {
     As[py][px]=tensor_left.GetElement(py_left,px_left);
     Bs[py][px]=tensor_right.GetElement(py_right,px_right);
    }
   }
   __syncthreads();

   //выполняем умножение только для тех элементов, которые не выходят за размер выходного тензора
   if ((out_y<tensor_output.Size_Y) && (out_x<tensor_output.Size_X))
   {
    for(uint32_t ky=0;ky<TENSOR_MUL_TILE_SIZE_SCALE;ky++)
    {
     for(uint32_t kx=0;kx<TENSOR_MUL_TILE_SIZE_SCALE;kx++)
     {
      type_t &cv=Cvalue[ky][kx];
      uint32_t ay=out_in_block_y+ky;
      uint32_t bx=out_in_block_x+kx;
      for(uint32_t e=0;e<TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE;e++) cv+=As[ay][e]*Bs[e][bx];
     }
    }
   }
   __syncthreads();
  }
  uint32_t py=out_y;
  for(uint32_t ky=0;ky<TENSOR_MUL_TILE_SIZE_SCALE;ky++,py++)
  {
   uint32_t px=out_x;
   for(uint32_t kx=0;kx<TENSOR_MUL_TILE_SIZE_SCALE;kx++,px++)
   {
    tensor_output.SetElement(py,px,Cvalue[ky][kx]);
   }
  }
  __syncthreads();
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

  //разбиваем выходной тензор на блоки по TENSOR_MUL_TILE_BLOCK_SIZExTENSOR_MUL_TILE_BLOCK_SIZE элементов
  //для каждого из этих элементов запускаем по нити (всего TENSOR_MUL_TILE_BLOCK_SIZExTENSOR_MUL_TILE_BLOCK_SIZE нитей)

  dim3 thread(TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE,TENSOR_MUL_TILE_BLOCK_SIZE*TENSOR_MUL_TILE_SIZE_SCALE);

  uint32_t block_x=sTensorKernel_Output.Size_X/thread.x;
  if (sTensorKernel_Output.Size_X%thread.x) block_x++;
  uint32_t block_y=sTensorKernel_Output.Size_Y/thread.y;
  if (sTensorKernel_Output.Size_Y%thread.y) block_y++;
  uint32_t block_z=sTensorKernel_Output.Size_Z*sTensorKernel_Output.Size_W;

  dim3 blocks(block_x,block_y,block_z);
  if (blocks.x==0) blocks.x=1;
  if (blocks.y==0) blocks.y=1;
  if (blocks.z==0) blocks.z=1;

  dim3 thread_basic(TENSOR_MUL_TILE_BLOCK_SIZE,TENSOR_MUL_TILE_BLOCK_SIZE);

  CUDATensorMulTensorFunction<type_t,kernel_output_t,kernel_left_t,kernel_right_t><<<blocks,thread_basic>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  cTensor_Output.SetDeviceOnChange();
 }

}

#endif
