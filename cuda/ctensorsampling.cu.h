#ifndef C_TENSOR_SAMPLING_CU_H
#define C_TENSOR_SAMPLING_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensosampling.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции увеличения/уменьшения размерности над тензорами произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "ctensor.cu.h"
#include "ctensormath.cu.h"
#include <vector>
#include "ccudatimespent.cu.h"

//****************************************************************************************************
//макроопределения
//****************************************************************************************************

//****************************************************************************************************
//константы
//****************************************************************************************************

//****************************************************************************************************
//предварительные объявления
//****************************************************************************************************

//****************************************************************************************************
//прототипы функций
//****************************************************************************************************


//****************************************************************************************************
///!Операции увеличения/уменьшения размерности над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorSampling
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
 public:
  //-конструктор----------------------------------------------------------------------------------------
  //-конструктор копирования----------------------------------------------------------------------------
  //-деструктор-----------------------------------------------------------------------------------------
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  static void UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y);///<увеличение разрешения тензора
  static void DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y);///<уменьшение разрешения тензора
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};

//****************************************************************************************************
//реализация функций
//****************************************************************************************************

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//функция CUDA для увеличения разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAUpSamplingTensor(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,uint32_t upsampling_x,uint32_t upsampling_y)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=blockIdx.x/tensor_output.Size_Z;
 uint32_t w_in=Mod(w,tensor_input.Size_W);
 uint32_t w_out=Mod(w,tensor_output.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 uint32_t ixp=xp/upsampling_x;
 uint32_t iyp=yp/upsampling_y;

 uint32_t offset_output=xp+yp*tensor_output.Size_X;
 type_t *o_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset_output;

 if (ixp>=tensor_input.Size_X || iyp>=tensor_input.Size_Y)
 {
  *o_ptr=0;
  __syncthreads();
  return;
 }

 uint32_t offset_input=ixp+iyp*tensor_input.Size_X;

 type_t *i_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset_input;

 *o_ptr=*i_ptr;

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!увеличение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorSampling<type_t>::UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X/upsampling_x || cTensor_Input.Size_Y!=cTensor_Output.Size_Y/upsampling_y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::UpSampling: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDAUpSamplingTensor<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,upsampling_x,upsampling_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для уменьшение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADownSamplingTensor(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,uint32_t downsampling_x,uint32_t downsampling_y)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=blockIdx.x/tensor_output.Size_Z;
 uint32_t w_in=Mod(w,tensor_input.Size_W);
 uint32_t w_out=Mod(w,tensor_output.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 uint32_t ixp=xp*downsampling_x;
 uint32_t iyp=yp*downsampling_y;

 uint32_t offset_output=xp+yp*tensor_output.Size_X;
 uint32_t offset_input=ixp+iyp*tensor_input.Size_X;

 type_t *i_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset_input;
 type_t *o_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset_output;

 type_t summ=0;
 for(uint32_t y=0;y<downsampling_y;y++,i_ptr+=tensor_input.StrideX)
 {
  type_t *i_ptr_local=i_ptr;
  for(uint32_t x=0;x<downsampling_x;x++,i_ptr_local++)
  {
   summ+=*i_ptr_local;
  }
 }

 *o_ptr=summ/static_cast<type_t>(downsampling_x*downsampling_y);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!уменьшение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorSampling<type_t>::DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y)
{
 if (cTensor_Input.Size_X/downsampling_x!=cTensor_Output.Size_X || cTensor_Input.Size_Y/downsampling_y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::DownSampling: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDADownSamplingTensor<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,downsampling_x,downsampling_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

#endif

#endif
