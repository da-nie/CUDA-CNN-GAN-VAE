#ifndef C_TENSOR_CLIP_CU_H
#define C_TENSOR_CLIP_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensoclip.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции ограничения над тензорами произвольной размерности
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
///!Операции ограничения над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorClip
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
  static void Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value);///<выполнить отсечку значений тензора
  static void ClipByNormXY(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold);///<ограничить тензор по норме XY
  static void ClipByNormX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold);///<ограничить тензор по норме X
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
//функция CUDA для выполнения отсечки значений тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAClipTensor(STensorKernel<type_t> tensor,type_t min_value,type_t max_value)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor.Size_Z),tensor.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor.Size_X || yp>=tensor.Size_Y) return;

 type_t value=tensor.GetElement(w,z,yp,xp);
 if (value>max_value) value=max_value;
 if (value<min_value) value=min_value;
 tensor.SetElement(w,z,yp,xp,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!выполнить отсечку значений тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorClip<type_t>::Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value)
{
 cTensor.CopyToDevice();

 STensorKernel<type_t> sTensorKernel(cTensor);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor.Size_X/thread.x;
 if (cTensor.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor.Size_Y/thread.y;
 if (cTensor.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor.Size_Z*cTensor.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDAClipTensor<type_t><<<blocks,thread>>>(sTensorKernel,min_value,max_value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для ограничения нормы элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorClipByNormXYTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t threshold)
{
 uint32_t w_in=Mod(blockIdx.x,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.x,tensor_output.Size_W);
 uint32_t z=Mod(blockIdx.y,tensor_output.Size_Z);

 //суммируем по X и Y
 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z);
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z);
 type_t *d_xin_local=d_xin;
 type_t summ=0;
 for(uint32_t y=0;y<tensor_input.Size_Y;y++)
 {
  for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin_local++)
  {
   type_t v=*d_xin_local;
   summ+=v*v;
  }
 }
 summ=sqrt(summ);
 //нормируем, если нужно
 if (summ>threshold)
 {
  type_t k=threshold/summ;
  for(uint32_t y=0;y<tensor_input.Size_Y;y++)
  {
   for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin++,d_xout++)
   {
    type_t v=*d_xin;
    v*=k;
    *d_xout=v;
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//!ограничить тензор по норме XY
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorClip<type_t>::ClipByNormXY(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold)
{
 if (cTensor_Input.Size_W!=cTensor_Output.Size_W || cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::ClipByNormXY: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input.Size_W,cTensor_Input.Size_Z);

 CUDATensorClipByNormXYTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,threshold);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для ограничения нормы элементов по X для каждого Y и Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorClipByNormXTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t threshold)
{
 uint32_t w_in=Mod(blockIdx.x,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.x,tensor_output.Size_W);
 uint32_t z=Mod(blockIdx.y,tensor_output.Size_Z);
 uint32_t y=blockIdx.z;

 //суммируем по X
 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z)+y*tensor_input.Size_X;
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z)+y*tensor_input.Size_X;
 type_t *d_xin_local=d_xin;
 type_t summ=0;
 for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin_local++)
 {
  type_t v=*d_xin_local;
  summ+=v*v;
 }
 summ=sqrt(summ);
 //нормируем, если нужно
 if (summ>threshold)
 {
  type_t k=threshold/summ;
  for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin++,d_xout++)
  {
   type_t v=*d_xin;
   v*=k;
   *d_xout=v;
  }
 }
}

//----------------------------------------------------------------------------------------------------
//!ограничить тензор по норме X
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorClip<type_t>::ClipByNormX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold)
{
 if (cTensor_Input.Size_W!=cTensor_Output.Size_W || cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::ClipByNormX: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input.Size_W,cTensor_Input.Size_Z,cTensor_Input.Size_Y);

 CUDATensorClipByNormXTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,threshold);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


#endif

#endif
