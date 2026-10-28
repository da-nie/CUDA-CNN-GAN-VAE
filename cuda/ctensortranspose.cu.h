#ifndef C_TENSOR_TRANSPOSE_CU_H
#define C_TENSOR_TRANSPOSE_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensotranspose.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции перестановки осей над тензорами произвольной размерности
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
///!Операции перестановки осей над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorTranspose
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
  static void Transpose(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<транспонировать тензор
  static CTensor<type_t> Transpose(const CTensor<type_t> &cTensor_Input);///<получить транспонированный тензор
  static void TransposeZX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<переставить оси Z и X
  static void TransposeYX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<переставить оси Y и X
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
//функция CUDA для транспонирования тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATranspose(STensorKernel<type_t> tensor_output,const STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_input.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_input.Size_Z),tensor_input.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_input.Size_X || yp>=tensor_input.Size_Y) return;

 type_t value=tensor_input.GetElement(w,z,yp,xp);
 tensor_output.SetElement(w,z,xp,yp,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!транспонировать тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorTranspose<type_t>::Transpose(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Output.Size_Y!=cTensor_Input.Size_X || cTensor_Output.Size_X!=cTensor_Input.Size_Y || cTensor_Output.Size_Z!=cTensor_Input.Size_Z || cTensor_Output.Size_W!=cTensor_Input.Size_W)
 {
  throw "void CTensor::Transpose: Размерности матриц не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Input.Size_Z*cTensor_Input.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATranspose<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//получить транспонированный тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> CTensorTranspose<type_t>::Transpose(const CTensor<type_t> &cTensor_Input)
{
 CTensor<type_t> cTensor(cTensor_Input.Size_W,cTensor_Input.Size_Z,cTensor_Input.Size_X,cTensor_Input.Size_Y);
 CTensorTranspose<type_t>::Transpose(cTensor,cTensor_Input);
 return(cTensor);
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для перестановки осей Z и X
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATransposeZX(STensorKernel<type_t> tensor_output,const STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_input.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_input.Size_Z),tensor_input.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_input.Size_X || yp>=tensor_input.Size_Y) return;

 type_t value=tensor_input.GetElement(w,z,yp,xp);
 tensor_output.SetElement(w,xp,yp,z,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!переставить оси Z и X
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorTranspose<type_t>::TransposeZX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Output.Size_W!=cTensor_Input.Size_W || cTensor_Output.Size_Z!=cTensor_Input.Size_X || cTensor_Output.Size_Y!=cTensor_Input.Size_Y || cTensor_Output.Size_X!=cTensor_Input.Size_Z)
 {
  throw "CTensorMath::TransposeZX: Размерности входных и выходных тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Input.Size_Z*cTensor_Input.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATransposeZX<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для перестановки осей Y и X
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATransposeYX(STensorKernel<type_t> tensor_output,const STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_input.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_input.Size_Z),tensor_input.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_input.Size_X || yp>=tensor_input.Size_Y) return;

 type_t value=tensor_input.GetElement(w,z,yp,xp);
 tensor_output.SetElement(w,z,xp,yp,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!переставить оси Y и X
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorTranspose<type_t>::TransposeYX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Output.Size_W!=cTensor_Input.Size_W || cTensor_Output.Size_Z!=cTensor_Input.Size_Z || cTensor_Output.Size_Y!=cTensor_Input.Size_X || cTensor_Output.Size_X!=cTensor_Input.Size_Y)
 {
  throw "CTensorMath::TransposeYX: Размерности входных и выходных тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Input.Size_Z*cTensor_Input.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATransposeYX<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


#endif

#endif
