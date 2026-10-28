#ifndef C_TENSOR_MATH_CU_H
#define C_TENSOR_MATH_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensormath.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции над тензорами произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "ctensor.cu.h"
#include "math/tensorkernel.cu.h"
#include "math/gemm_cudacore_typea.cu.h"
#include "math/gemm_cudacore_typeb.cu.h"//самый быстрый вариант на CUDA-ядрах
#include "math/gemm_tensorcoregenone_typea.cu.h"
#include "math/gemm_tensorcoregenone_typeb.cu.h"//самый быстрый вариант на тензорных ядрах первого поколения
#include "math/gemm_tensorcoregenone_typec.cu.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

//****************************************************************************************************
//макроопределения
//****************************************************************************************************

//****************************************************************************************************
//константы
//****************************************************************************************************
static const double TENSOR_EPS=0.0000000001;

//****************************************************************************************************
//предварительные объявления
//****************************************************************************************************

//****************************************************************************************************
//прототипы функций
//****************************************************************************************************


//****************************************************************************************************
///!Операции над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorMath
{
 template<class new_type_t>
 friend struct STensorKernel;

 template<class new_type_t>
 friend struct STensorKernel_Transpose;

 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
  static const uint32_t TENSOR_MUL_TILE_SIZE_SCALE=2;///<размер множителя блока операции умножения с тензорами
  static const uint32_t TENSOR_MUL_TILE_BLOCK_SIZE=16;///<размер блока операции умножения с тензорами
  static const uint32_t TILE_BLOCK_SIZE=16;///<размер блока операций с тензорами

 private:
  //-переменные-----------------------------------------------------------------------------------------
 public:
  //-конструктор----------------------------------------------------------------------------------------
  //-конструктор копирования----------------------------------------------------------------------------
  //-деструктор-----------------------------------------------------------------------------------------
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  static void ConcatecationZ(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_InputA,const CTensor<type_t> &cTensor_InputB);///<объединить два тензора по Z
  static void SplitZ(CTensor<type_t> &cTensor_OutputA,CTensor<type_t> &cTensor_OutputB,const CTensor<type_t> &cTensor_Input);///<разделить тензор на два по Z
  static void Fill(CTensor<type_t> &cTensor_Output,type_t value=0);///<записать в тензор число
  static void Inv(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<вычислить обратный тензор
  static void Div(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<поделить тензоры
  static void Add(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<сложить тензоры
  static void AddValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value);///<прибавить к каждому элементу тензора число
  static void Sub(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<вычесть тензоры
  static void SubValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value);///<отнять от каждого элемента тензора число
  static void Set(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale=1,type_t add_value=0);///<скопировать тензор с масштабированием
  static void Pow2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale=1);///<возведение элементов тензора в квадрат
  static void SQRT(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_sqrt_value);///<вычисление квадратного корня из элементов тензора
  static void AddBias(CTensor<type_t> &cTensor_Working,const CTensor<type_t> &cTensor_Bias);///<добавить смещения к элементам тензора (смещения одинаковы для x и y, но по z смещения разные)
  static void AddToXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_UnitWZ,type_t scale_input=1,type_t scale_unit_wz=1);///<прибавить одинаковые значения элементов по X и Y для каждого Z и W
  static void LayerAddX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_ValueX);///<добавить значения по слою X
  static void SubTensor(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,int32_t w,int32_t z,int32_t y,int32_t x);///<скопировать один тензор в другой с позиции

  template<class kernel_output_t,class kernel_left_t,class kernel_right_t>
  static void MulAbstract(CTensor<type_t> &cTensor_Output,kernel_output_t &sTensorKernel_Output,const CTensor<type_t> &cTensor_Left,kernel_left_t &sTensorKernel_Left,const CTensor<type_t> &cTensor_Right,kernel_right_t &sTensorKernel_Right);///<умножить тензоры

  static void Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить тензоры
  static void LeftTransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить транспонированный левый тензор на правый
  static void RightTransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить левый тензор на транспонированный правый
  static void Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const type_t &value_right);///<умножить тензор на число
  static void Mul(CTensor<type_t> &cTensor_Output,const type_t &value_left,const CTensor<type_t> &cTensor_Right);///<умножить тензор на число
  static void TensorItemProduction(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Left,CTensor<type_t> &cTensor_Right);///<поэлементное произведение тензора на тензор

  static void Signum(CTensor<type_t> &cTensor);///<выставить знак элементам тензора
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};

//****************************************************************************************************
//реализация функций
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//оператор "+"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator+(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Right.Size_W,cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Left.Size_X);
 CTensorMath<type_t>::Add(cTensor,cTensor_Left,cTensor_Right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "-"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator-(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Right.Size_W,cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Left.Size_X);
 CTensorMath<type_t>::Sub(cTensor,cTensor_Left,cTensor_Right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator*(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Right.Size_W,cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Right.Size_X);
 CTensorMath<type_t>::Mul(cTensor,cTensor_Left,cTensor_Right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator*(const CTensor<type_t> &cTensor_Left,const type_t &value_right)
{
 CTensor<type_t> cTensor(cTensor_Left.Size_W,cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Left.Size_X);
 CTensorMath<type_t>::Mul(cTensor,cTensor_Left,value_right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator*(const type_t &value_left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Right.Size_W,cTensor_Right.Size_Z,cTensor_Right.Size_Y,cTensor_Right.Size_X);
 CTensorMath<type_t>::Mul(cTensor,value_left,cTensor_Right);
 return(cTensor);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************


//----------------------------------------------------------------------------------------------------
//функция CUDA для объединения двух тензоров по Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorConcatecationZFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input_a,STensorKernel<type_t> tensor_input_b)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=blockIdx.x/tensor_output.Size_Z;
 uint32_t w_in_a=Mod(w,tensor_input_a.Size_W);
 uint32_t w_in_b=Mod(w,tensor_input_b.Size_W);
 uint32_t w_out=Mod(w,tensor_output.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 type_t v=0;
 if (z<tensor_input_a.Size_Z) v=tensor_input_a.GetElement(w_in_a,z,yp,xp);
                             else v=tensor_input_b.GetElement(w_in_b,z-tensor_input_a.Size_Z,yp,xp);
 tensor_output.SetElement(w_out,z,yp,xp,v);
}

//----------------------------------------------------------------------------------------------------
//объединить два тензора по Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::ConcatecationZ(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_InputA,const CTensor<type_t> &cTensor_InputB)
{
 if (cTensor_Output.Size_X!=cTensor_InputA.Size_X || cTensor_Output.Size_X!=cTensor_InputB.Size_X ||
     cTensor_Output.Size_Y!=cTensor_InputA.Size_Y || cTensor_Output.Size_Y!=cTensor_InputB.Size_Y ||
     cTensor_Output.Size_Z!=(cTensor_InputA.Size_Z+cTensor_InputB.Size_Z))
 {
  throw "CTensor::ConcatecationZ: Размерности тензоров не совпадают!";
 }

 cTensor_InputA.CopyToDevice();
 cTensor_InputB.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_InputA(cTensor_InputA);
 STensorKernel<type_t> sTensorKernel_InputB(cTensor_InputB);

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
 CUDATensorConcatecationZFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_InputA,sTensorKernel_InputB);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для разделения двух тензоров по Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSplitZFunction(STensorKernel<type_t> tensor_output_a,STensorKernel<type_t> tensor_output_b,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_input.Size_Z);
 uint32_t w=blockIdx.x/tensor_input.Size_Z;
 uint32_t w_in=Mod(w,tensor_input.Size_W);
 uint32_t w_out_a=Mod(w,tensor_output_a.Size_W);
 uint32_t w_out_b=Mod(w,tensor_output_b.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_input.Size_X || yp>=tensor_input.Size_Y) return;

 type_t v=tensor_input.GetElement(w_in,z,yp,xp);
 if (z<tensor_output_a.Size_Z) tensor_output_a.SetElement(w_out_a,z,yp,xp,v);
                              else tensor_output_b.SetElement(w_out_b,z-tensor_output_a.Size_Z,yp,xp,v);
}

//----------------------------------------------------------------------------------------------------
//разделить тензор на два по Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SplitZ(CTensor<type_t> &cTensor_OutputA,CTensor<type_t> &cTensor_OutputB,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_OutputA.Size_X!=cTensor_Input.Size_X || cTensor_OutputB.Size_X!=cTensor_Input.Size_X ||
     cTensor_OutputB.Size_Y!=cTensor_Input.Size_Y || cTensor_OutputB.Size_Y!=cTensor_Input.Size_Y ||
     (cTensor_OutputA.Size_Z+cTensor_OutputB.Size_Z)!=cTensor_Input.Size_Z)
 {
  throw "CTensor::SplitZ: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_OutputA(cTensor_OutputA);
 STensorKernel<type_t> sTensorKernel_OutputB(cTensor_OutputB);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_OutputA.Size_X/thread.x;
 if (cTensor_OutputA.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_OutputA.Size_Y/thread.y;
 if (cTensor_OutputA.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Input.Size_Z*cTensor_Input.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorSplitZFunction<type_t><<<blocks,thread>>>(sTensorKernel_OutputA,sTensorKernel_OutputB,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_OutputA.SetDeviceOnChange();
 cTensor_OutputB.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для записи числа в тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorFillFunction(STensorKernel<type_t> tensor_output,type_t value)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_output.Size_Z),tensor_output.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w,z)+offset;
 *out_ptr=value;
}

//----------------------------------------------------------------------------------------------------
//записать в тензор число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Fill(CTensor<type_t> &cTensor_Output,type_t value)
{
 if (cTensor_Output.Size_X*cTensor_Output.Size_Y*cTensor_Output.Size_Z<CTensorMath<type_t>::TILE_BLOCK_SIZE*CTensorMath<type_t>::TILE_BLOCK_SIZE*CTensorMath<type_t>::TILE_BLOCK_SIZE)
 {
  cTensor_Output.Fill(value);
  return;
 }

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

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
 CUDATensorFillFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления обратного тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorInvTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
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

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;
 type_t e=(*in_ptr);

 *out_ptr=1.0/e;
}

//----------------------------------------------------------------------------------------------------
//вычисление обратного тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Inv(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::Inv: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorInvTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для деления тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorDivTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right,type_t left_scale,type_t right_scale)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=blockIdx.x/tensor_output.Size_Z;
 uint32_t w_left=Mod(w,tensor_left.Size_W);
 uint32_t w_right=Mod(w,tensor_right.Size_W);
 uint32_t w_out=Mod(w,tensor_output.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *a_ptr=tensor_left.GetTensorDataPtr(w_left,z)+offset;
 type_t *b_ptr=tensor_right.GetTensorDataPtr(w_right,z)+offset;
 type_t *c_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *c_ptr=(*a_ptr)*left_scale/((*b_ptr)*right_scale);
}

//----------------------------------------------------------------------------------------------------
//поделить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Div(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale,type_t right_scale)
{
 if (cTensor_Left.Size_X!=cTensor_Right.Size_X || cTensor_Left.Size_Y!=cTensor_Right.Size_Y || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Div: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Left.Size_X/thread.x;
 if (cTensor_Left.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorDivTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для сложения тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right,type_t left_scale,type_t right_scale)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=blockIdx.x/tensor_output.Size_Z;
 uint32_t w_left=Mod(w,tensor_left.Size_W);
 uint32_t w_right=Mod(w,tensor_right.Size_W);
 uint32_t w_output=Mod(w,tensor_output.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *a_ptr=tensor_left.GetTensorDataPtr(w_left,z)+offset;
 type_t *b_ptr=tensor_right.GetTensorDataPtr(w_right,z)+offset;
 type_t *c_ptr=tensor_output.GetTensorDataPtr(w_output,z)+offset;

 *c_ptr=(*a_ptr)*left_scale+((*b_ptr)*right_scale);
}

//----------------------------------------------------------------------------------------------------
//сложить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Add(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale,type_t right_scale)
{
 if (cTensor_Left.Size_X!=cTensor_Right.Size_X || cTensor_Left.Size_Y!=cTensor_Right.Size_Y || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Add: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=sTensorKernel_Output.Size_X/thread.x;
 if (sTensorKernel_Output.Size_X%thread.x) block_z++;
 uint32_t block_y=sTensorKernel_Output.Size_Y/thread.y;
 if (sTensorKernel_Output.Size_Y%thread.y) block_y++;
 uint32_t block_x=sTensorKernel_Output.Size_Z*sTensorKernel_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;


 CUDATensorAddTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для прибавления числа к элементам тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddValueTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t tensor_scale,type_t value)
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

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *out_ptr=(*in_ptr)*tensor_scale+value;
}

//----------------------------------------------------------------------------------------------------
//прибавить к каждому элементу тензора число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::AddValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::AddValue: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorAddValueTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,tensor_scale,value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//вычесть тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Sub(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale,type_t right_scale)
{
 if (cTensor_Left.Size_X!=cTensor_Right.Size_X || cTensor_Left.Size_Y!=cTensor_Right.Size_Y || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Sub: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 //запускаем процесс

 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Left.Size_X/thread.x;
 if (cTensor_Left.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorAddTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,-right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//отнять от каждого элемента тензора число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SubValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::SubValue: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorAddValueTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,tensor_scale,-value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для копирования с масштабированием тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSetTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t scale,type_t add_value)
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

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *out_ptr=(*in_ptr)*scale+add_value;

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//скопировать тензор с масштабированием
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Set(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_value)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::Set: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс

 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorSetTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,scale,add_value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для возведение элементов тензора в квадрат
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorPow2TensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t scale)
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

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;
 type_t e=(*in_ptr);

 *out_ptr=e*e*scale;
}

//----------------------------------------------------------------------------------------------------
//возведение элементов тензора в квадрат
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Pow2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::Pow2: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorPow2TensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисление квадратного корня из элементов тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSQRTTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t scale,type_t add_sqrt_value)
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

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;
 type_t e=(*in_ptr);

 *out_ptr=(sqrt(e+add_sqrt_value))*scale;
}

//----------------------------------------------------------------------------------------------------
//вычисление квадратного корня из элементов тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SQRT(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_sqrt_value)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::SQRT: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorSQRTTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,scale,add_sqrt_value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для добавления смещения к элементам тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddBiasFunction(STensorKernel<type_t> tensor_working,STensorKernel<type_t> tensor_bias)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_working.Size_Z);
 uint32_t w=blockIdx.x/tensor_working.Size_Z;
 uint32_t w_bias=Mod(w,tensor_bias.Size_W);
 uint32_t w_working=Mod(w,tensor_working.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_working.Size_X || yp>=tensor_working.Size_Y) return;

 uint32_t offset=xp+yp*tensor_working.Size_X;
 type_t *a_ptr=tensor_working.GetTensorDataPtr(w_working,z)+offset;
 type_t *b_ptr=tensor_bias.GetTensorDataPtr(w_bias,z);

 *a_ptr=(*a_ptr)+(*b_ptr);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//добавить смещения к элементам тензора (смещения одинаковы для x и y, но по z смещения разные)
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::AddBias(CTensor<type_t> &cTensor_Working,const CTensor<type_t> &cTensor_Bias)
{
 if (cTensor_Working.Size_Z!=cTensor_Bias.Size_Z)
 {
  throw "CTensor::AddBias: Размерности тензоров не совпадают!";
 }
 cTensor_Bias.CopyToDevice();
 cTensor_Working.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Bias(cTensor_Bias);
 STensorKernel<type_t> sTensorKernel_Working(cTensor_Working);
 //запускаем процесс

 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Working.Size_X/thread.x;
 if (cTensor_Working.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Working.Size_Y/thread.y;
 if (cTensor_Working.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Working.Size_Z*cTensor_Working.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorAddBiasFunction<type_t><<<blocks,thread>>>(sTensorKernel_Working,sTensorKernel_Bias);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Working.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для прибавления одинаковых значений элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddToXYTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<type_t> tensor_unit_wz,type_t scale_input,type_t scale_unit_wz)
{
 uint32_t w_unit_wz=Mod(blockIdx.x,tensor_unit_wz.Size_W);
 uint32_t w_in=Mod(blockIdx.x,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.x,tensor_output.Size_W);
 uint32_t z=Mod(blockIdx.y,tensor_output.Size_Z);

 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z);
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z);

 type_t d=tensor_unit_wz.GetElement(w_unit_wz,z,0,0);
 d*=scale_unit_wz;

 for(uint32_t y=0;y<tensor_input.Size_Y;y++)
 {
  for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin++,d_xout++)
  {
   type_t v=(*d_xin);
   v*=scale_input,
   (*d_xout)=v+d;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//прибавить одинаковые значения элементов по X и Y для каждого Z и W
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::AddToXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_UnitWZ,type_t scale_input,type_t scale_unit_wz)
{
 if (cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_W!=cTensor_Output.Size_W)
 {
  throw "CTensor::AddXY: Размерности тензоров не совпадают!";
 }
 if (cTensor_UnitWZ.Size_X!=1 || cTensor_UnitWZ.Size_Y!=1)
 {
  throw "CTensor::AddXY: Добавляемый тензор должен иметь по x и y размерность 1!";
 }
 if (cTensor_UnitWZ.Size_W!=cTensor_Input.Size_W || cTensor_UnitWZ.Size_Z!=cTensor_Input.Size_Z)
 {
  throw "CTensor::AddXY: Добавляемый тензор должен иметь по w и z размерность равную входному тензору!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_UnitWZ(cTensor_UnitWZ);

 //запускаем процесс
 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input.Size_W,cTensor_Input.Size_Z);

 CUDATensorAddToXYTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,sTensorKernel_UnitWZ,scale_input,scale_unit_wz);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для добавления значения по слою X
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorLayerAddXTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<type_t> tensor_valuex)
{
 uint32_t w_in=Mod(blockIdx.x,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.x,tensor_output.Size_W);
 uint32_t w_in_x=Mod(blockIdx.x,tensor_valuex.Size_W);
 uint32_t z=Mod(blockIdx.y,tensor_output.Size_Z);
 uint32_t y=Mod(blockIdx.z,tensor_output.Size_Y);


 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z)+y*tensor_input.Size_X;
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z)+y*tensor_input.Size_X;

 type_t *d_valuex=tensor_valuex.GetTensorDataPtr(w_in_x,z);
 for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin++,d_xout++,d_valuex++)
 {
  type_t v=(*d_xin);
  v+=(*d_valuex);
  *d_xout=v;
 }
}

//----------------------------------------------------------------------------------------------------
//добавить значения по слою X
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::LayerAddX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_ValueX)
{
 if (cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_W!=cTensor_Output.Size_W ||
     cTensor_Input.Size_Z!=cTensor_ValueX.Size_Z || cTensor_Input.Size_X!=cTensor_ValueX.Size_X || cTensor_ValueX.Size_Y!=1)
 {
  throw "CTensor::LayerAddX: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_ValueX.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_ValueX(cTensor_ValueX);

 //запускаем процесс
 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input.Size_W,cTensor_Input.Size_Z,cTensor_Input.Size_Y);

 CUDATensorLayerAddXTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,sTensorKernel_ValueX);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для копирования одного тензора в другой с позиции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSubTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,int32_t w_offset,int32_t z_offset,int32_t y_offset,int32_t x_offset)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=blockIdx.x/tensor_output.Size_Z;
 uint32_t w_in=w;
 uint32_t w_out=w;
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 type_t value=tensor_input.GetElement(w_in+w_offset,z+z_offset,y+y_offset,x+x_offset);
 tensor_output.SetElement(w_out,z,yp,xp,value);
}

//----------------------------------------------------------------------------------------------------
//скопировать один тензор в другой с позиции
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SubTensor(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,int32_t w,int32_t z,int32_t y,int32_t x)
{
/* if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::Inv: Размерности тензоров не совпадают!";
 }
 */

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
 CUDATensorSubTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,w,z,y,x);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//умножить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t> template<class kernel_output_t,class kernel_left_t,class kernel_right_t>
__host__ void CTensorMath<type_t>::MulAbstract(CTensor<type_t> &cTensor_Output,kernel_output_t &sTensorKernel_Output,const CTensor<type_t> &cTensor_Left,kernel_left_t &sTensorKernel_Left,const CTensor<type_t> &cTensor_Right,kernel_right_t &sTensorKernel_Right)
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

 #ifdef USE_TENSOR_CORE_GENERATION_ONE
 NSTensorCoreGenOneTypeB::MulAbstract<type_t,kernel_output_t,kernel_left_t,kernel_right_t>(cTensor_Output,sTensorKernel_Output,cTensor_Left,sTensorKernel_Left,cTensor_Right,sTensorKernel_Right);
 #endif

 #ifndef USE_TENSOR_CORE_GENERATION_ONE
 NSCUDACoreTypeB::MulAbstract<type_t,kernel_output_t,kernel_left_t,kernel_right_t>(cTensor_Output,sTensorKernel_Output,cTensor_Left,sTensorKernel_Left,cTensor_Right,sTensorKernel_Right);
 #endif

}

//----------------------------------------------------------------------------------------------------
//умножить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);
 MulAbstract<STensorKernel<type_t>,STensorKernel<type_t>,STensorKernel<type_t>>(cTensor_Output,sTensorKernel_Output,cTensor_Left,sTensorKernel_Left,cTensor_Right,sTensorKernel_Right);
}

//----------------------------------------------------------------------------------------------------
//умножить транспонированный левый тензор на правый
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::LeftTransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel_Transpose<type_t> sTensorKernel_Transpose_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);
 MulAbstract<STensorKernel<type_t>,STensorKernel_Transpose<type_t>,STensorKernel<type_t>>(cTensor_Output,sTensorKernel_Output,cTensor_Left,sTensorKernel_Transpose_Left,cTensor_Right,sTensorKernel_Right);
}
//----------------------------------------------------------------------------------------------------
//умножить левый тензор на транспонированный правый
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::RightTransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel_Transpose<type_t> sTensorKernel_Transpose_Right(cTensor_Right);
 MulAbstract<STensorKernel<type_t>,STensorKernel<type_t>,STensorKernel_Transpose<type_t>>(cTensor_Output,sTensorKernel_Output,cTensor_Left,sTensorKernel_Left,cTensor_Right,sTensorKernel_Transpose_Right);
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для умножения тензора на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorMulValueFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t value)
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

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=(*a_ptr)*value;

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//умножить тензор на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const type_t &value_right)
{
 if (cTensor_Output.Size_X!=cTensor_Left.Size_X || cTensor_Output.Size_Y!=cTensor_Left.Size_Y || cTensor_Output.Size_Z!=cTensor_Left.Size_Z)
 {
  throw "CTensor::Mul: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Left);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Left.Size_X/thread.x;
 if (cTensor_Left.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorMulValueFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,value_right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
//----------------------------------------------------------------------------------------------------
//умножить тензор на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const type_t &value_left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Mul: Размерности тензоров не совпадают!";
 }

 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Right);

 //запускаем процесс

 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Right.Size_X/thread.x;
 if (cTensor_Right.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Right.Size_Y/thread.y;
 if (cTensor_Right.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorMulValueFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,value_left);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления скалярного произведения строк тензора между собой
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorItemProductionFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=blockIdx.x/tensor_output.Size_Z;
 uint32_t w_left=Mod(w,tensor_left.Size_W);
 uint32_t w_right=Mod(w,tensor_right.Size_W);
 uint32_t w_out=Mod(w,tensor_output.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 uint32_t offset=xp+yp*tensor_output.Size_X;
 type_t *a_ptr=tensor_left.GetTensorDataPtr(w_left,z)+offset;
 type_t *b_ptr=tensor_right.GetTensorDataPtr(w_right,z)+offset;
 type_t *c_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *c_ptr=(*a_ptr)*(*b_ptr);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//поэлементное произведение тензора на тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::TensorItemProduction(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Left,CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Right.Size_X!=cTensor_Left.Size_X || cTensor_Right.Size_Y!=cTensor_Left.Size_Y || cTensor_Right.Size_Z!=cTensor_Left.Size_Z) throw("Ошибка поэлементного умножения тензора на тензор");
 if (cTensor_Right.Size_X!=cTensor_Output.Size_X || cTensor_Right.Size_Y!=cTensor_Output.Size_Y || cTensor_Right.Size_Z!=cTensor_Output.Size_Z) throw("Ошибка поэлементного умножения тензора на тензор");

 //копируем данные на устройство
 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 //разбиваем выходную матрицы тензора на блоки по TILE_BLOCK_SIZExTILE_BLOCK_SIZE элементов
 //для каждого из этих элементов запускаем по нити (всего TILE_BLOCK_SIZExTILE_BLOCK_SIZE нитей)

 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Right.Size_X/thread.x;
 if (cTensor_Right.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorItemProductionFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для установки знака элементов тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASignumTensor(STensorKernel<type_t> tensor)
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
 if (value>0) value=1;
 if (value<0) value=-1;
 tensor.SetElement(w,z,yp,xp,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!выставить знак элементов тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Signum(CTensor<type_t> &cTensor)
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
 CUDASignumTensor<type_t><<<blocks,thread>>>(sTensorKernel);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor.SetDeviceOnChange();
}

#endif

#endif
