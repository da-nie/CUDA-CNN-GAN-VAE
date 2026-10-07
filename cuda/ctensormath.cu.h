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
 friend struct STensorTransposeKernel;
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  struct SPos
  {
   uint32_t X;
   uint32_t Y;
  };
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
  static void AddSumW(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<сложить тензоры по координате W
  static void AddValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value);///<прибавить к каждому элементу тензора число
  static void Sub(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<вычесть тензоры
  static void SubValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value);///<отнять от каждого элемента тензора число
  static void Set(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale=1,type_t add_value=0);///<скопировать тензор с масштабированием
  static void Pow2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale=1);///<возведение элементов тензора в квадрат
  static void SQRT(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_sqrt_value);///<вычисление квадратного корня из элементов тензора
  static void AddBias(CTensor<type_t> &cTensor_Working,const CTensor<type_t> &cTensor_Bias);///<добавить смещения к элементам тензора (смещения одинаковы для x и y, но по z смещения разные)
  static void SumXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,type_t scale=1);///<вычислить сумму элементов по X и Y для каждого Z
  static void AddToXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_UnitWZ,type_t scale_input=1,type_t scale_unit_wz=1);///<прибавить одинаковые значения элементов по X и Y для каждого Z и W
  static void LayerNormalizeX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta);///<выполнить нормализацию по слою X
  static void LayerAddX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_ValueX);///<добавить значения по слою X
  static void SubTensor(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,int32_t w,int32_t z,int32_t y,int32_t x);///<скопировать один тензор в другой с позиции
  static void SplitKQVTensor(CTensor<type_t> &cTensor_Q,CTensor<type_t> &cTensor_K,CTensor<type_t> &cTensor_V,const CTensor<type_t> &cTensor_QKV,int32_t num_total_tokens,int32_t head_dim,int32_t arch_dim,int32_t q_split_index,int32_t k_split_index,int32_t v_split_index);//разделить тензор на Q,K,V

  template<class kernel_output_t,class kernel_left_t,class kernel_right_t>
  static void MulAbstract(CTensor<type_t> &cTensor_Output,kernel_output_t &sTensorKernel_Output,const CTensor<type_t> &cTensor_Left,kernel_left_t &sTensorKernel_Left,const CTensor<type_t> &cTensor_Right,kernel_right_t &sTensorKernel_Right);///<умножить тензоры

  static void Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить тензоры
  static void LeftTransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить транспонированный левый тензор на правый
  static void RightTransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить левый тензор на транспонированный правый
  static void Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const type_t &value_right);///<умножить тензор на число
  static void Mul(CTensor<type_t> &cTensor_Output,const type_t &value_left,const CTensor<type_t> &cTensor_Right);///<умножить тензор на число
  static void Transpose(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<транспонировать тензор
  static void TensorItemProduction(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Left,CTensor<type_t> &cTensor_Right);///<поэлементное произведение тензора на тензор
  static CTensor<type_t> Transpose(const CTensor<type_t> &cTensor_Input);///<получить транспонированный тензор

  static void UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y);///<увеличение разрешения тензора
  static void DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y);///<уменьшение разрешения тензора

  static void MaxPooling(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y);///<уменьшение разрешения тензора выборкой большего элемента
  static void MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y);///<обратный проход при увеличении разрешения тензора выборкой большего элемента
  static void Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value);///<выполнить отсечку значений тензора
  static void Signum(CTensor<type_t> &cTensor);///<выставить знак элементам тензора

  static void Adam(CTensor<type_t> &cTensor_Weight,CTensor<type_t> &cTensor_dWeight,CTensor<type_t> &cTensor_M,CTensor<type_t> &cTensor_V,uint32_t batch_size,double speed,double beta1,double beta2,double epsilon,double iteration);///<выполнить алгоритм Adam к весовому тензору

  static void SetTimeStep(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,const CTensor<uint32_t> &cTensor_TimeStep,type_t scale);///<добавить к тензору временной шаг

  static void CreateDropOutMatrix(CTensor<type_t> &cTensor_Output,type_t drop_out);///<создать матрицу исключения

  static void SetNormalNoise(CTensor<type_t> &cTensor_Output);///<задать тензор случайными значениями с нормальным распределением

  static void GetNoiseImageAndNoise(CTensor<type_t> &cTensor_NoisyImage,CTensor<type_t> &cTensor_Noise,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_SqrtAlphaBar,const CTensor<type_t> &cTensor_SqrtOneMinusAlphaBar);///<заполнить тензоры зашумлённым изображением и шумом

  static void ClipByNormXY(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold);///<ограничить тензор по норме XY
  static void ClipByNormX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold);///<ограничить тензор по норме X

  static void GroupNormForward(CTensor<type_t> &cTensor_Output, CTensor<type_t> &cTensor_Input, CTensor<type_t> &cTensor_Gamma, CTensor<type_t> &cTensor_Beta, CTensor<type_t> &cTensor_XHAT, CTensor<type_t> &cTensor_InvStd, uint32_t num_groups, uint32_t channels_per_group,type_t epsilon);///<прямой проход GroupNorm
  static void GroupNormBackward(CTensor<type_t> &cTensor_Delta_Array, CTensor<type_t> &cTensor_XHAT_Array, CTensor<type_t> &cTensor_Gamma, CTensor<type_t> &cTensor_InvStd_Array, CTensor<type_t> &cTensor_PrevLayerError_Array, CTensor<type_t> &cTensor_dGamma, CTensor<type_t> &cTensor_dBeta, uint32_t num_groups, uint32_t channels_per_group, bool calc_weights);///<обратный проход GroupNorm
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
//функция CUDA для сложения тензоров по координате W
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddSumWTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right,type_t left_scale,type_t right_scale)
{
 //блок TILE_BLOCK_SIZE x TILE_BLOCK_SIZE в выходном тензоре
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x+blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE;
 uint32_t y=threadIdx.y+blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE;

 type_t summ_left=0;
 type_t summ_right=0;
 for(uint32_t w=0;w<tensor_left.Size_W;w++)
 {
  type_t v=tensor_left.GetElement(w,z,y,x);
  summ_left+=v;
 }
 summ_left*=left_scale;
 for(uint32_t w=0;w<tensor_right.Size_W;w++)
 {
  type_t v=tensor_right.GetElement(w,z,y,x);
  summ_right+=v;
 }
 summ_right*=right_scale;
 type_t summ_output=summ_left+summ_right;
 tensor_output.SetElement(0,z,y,x,summ_output);//помещаем сумму в нулевой слой w
 for(uint32_t w=1;w<tensor_output.Size_W;w++) tensor_output.SetElement(w,z,y,x,0);//все остальные слои w обнулены

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//сложить тензоры по координате W
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::AddSumW(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale,type_t right_scale)
{
 if (cTensor_Left.Size_X!=cTensor_Right.Size_X || cTensor_Left.Size_Y!=cTensor_Right.Size_Y || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::AddSumW: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_x=sTensorKernel_Output.Size_X/thread.x;
 if (sTensorKernel_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=sTensorKernel_Output.Size_Y/thread.y;
 if (sTensorKernel_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=sTensorKernel_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 CUDATensorAddSumWTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,right_scale);
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
//функция CUDA для вычисления суммы элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSumXYTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t scale)
{
 uint32_t w_in=Mod(blockIdx.x,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.x,tensor_output.Size_W);
 uint32_t z=Mod(blockIdx.y,tensor_output.Size_Z);

 //суммируем по X и Y
 type_t summ=0;

 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z);
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z);

 for(uint32_t y=0;y<tensor_input.Size_Y;y++)
 {
  for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin++) summ+=*d_xin;
 }
 *d_xout=summ*scale;
}

//----------------------------------------------------------------------------------------------------
//вычислить сумму элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SumXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,type_t scale)
{
 if (cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Input.Size_W!=cTensor_Output.Size_W)
 {
  throw "CTensor::SumXY: Размерности тензоров не совпадают!";
 }
 if (cTensor_Output.Size_X!=1 || cTensor_Output.Size_Y!=1)
 {
  throw "CTensor::SumXY: Размерность выходного тензора по x и y должна быть 1!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input.Size_W,cTensor_Input.Size_Z);

/*
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
 CUDATensorSumXYTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
*/
 CUDATensorSumXYTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
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
//функция CUDA для вычисления нормализации по слою X
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorLayerNormalizeXTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<type_t> tensor_dgamma,STensorKernel<type_t> tensor_dbeta)
{
 const type_t LAYER_NORM_EPS=1e-5f;

 uint32_t w_in=Mod(blockIdx.x,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.x,tensor_output.Size_W);
 uint32_t z=Mod(blockIdx.y,tensor_output.Size_Z);
 uint32_t y=Mod(blockIdx.z,tensor_output.Size_Y);

 //выполняем нормализацию слоя

 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z)+y*tensor_input.Size_X;
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z)+y*tensor_input.Size_X;

 type_t *d_xin_local;
 type_t *d_xout_local;
 //считаем среднее по X
 type_t mean=0;
 d_xin_local=d_xin;
 for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin_local++) mean+=*d_xin_local;
 mean/=static_cast<type_t>(tensor_input.Size_X);
 //считаем дисперсию
 type_t var=0;
 d_xin_local=d_xin;
 d_xout_local=d_xout;
 for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin_local++,d_xout_local++)
 {
  type_t v=*d_xin_local;
  v-=mean;
  *d_xout_local=v;
  var+=v*v;
 }
 var/=static_cast<type_t>(tensor_input.Size_X);
 //обратная дисперсия
 type_t inv_std=1.0/sqrtf(var+LAYER_NORM_EPS);
 //нормируем
 d_xin_local=d_xin;
 d_xout_local=d_xout;
 type_t *d_gamma=tensor_dgamma.GetTensorDataPtr(w_in,z);
 type_t *d_beta=tensor_dbeta.GetTensorDataPtr(w_out,z);
 for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin_local++,d_xout_local++,d_gamma++,d_beta++)
 {
  type_t v=(*d_xout_local);
  v*=inv_std;
  type_t dgamma=*d_gamma;
  type_t dbeta=*d_beta;
  v=v*dgamma+dbeta;
  *d_xout_local=v;
 }
}

//----------------------------------------------------------------------------------------------------
//выполнить нормализацию по слою X
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::LayerNormalizeX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta)
{
 if (cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_W!=cTensor_Output.Size_W ||
     cTensor_Input.Size_Z!=cTensor_dGamma.Size_Z || cTensor_Input.Size_X!=cTensor_dGamma.Size_X || cTensor_Input.Size_W!=cTensor_dGamma.Size_W || cTensor_dGamma.Size_Y!=1 ||
     cTensor_Input.Size_Z!=cTensor_dBeta.Size_Z || cTensor_Input.Size_X!=cTensor_dBeta.Size_X || cTensor_Input.Size_W!=cTensor_dBeta.Size_W || cTensor_dBeta.Size_Y!=1)
 {
  throw "CTensor::LayerNormalizeX: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_dGamma.CopyToDevice();
 cTensor_dBeta.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_dGamma(cTensor_dGamma);
 STensorKernel<type_t> sTensorKernel_dBeta(cTensor_dBeta);

 //запускаем процесс
 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input.Size_W,cTensor_Input.Size_Z,cTensor_Input.Size_Y);

 CUDATensorLayerNormalizeXTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,sTensorKernel_dGamma,sTensorKernel_dBeta);
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
 uint32_t z=Mod(blockIdx.y,tensor_output.Size_Z);
 uint32_t y=Mod(blockIdx.z,tensor_output.Size_Y);

 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z)+y*tensor_input.Size_X;
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z)+y*tensor_input.Size_X;

 type_t *d_valuex=tensor_valuex.GetTensorDataPtr(w_in,z);
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
     cTensor_Input.Size_Z!=cTensor_ValueX.Size_Z || cTensor_Input.Size_X!=cTensor_ValueX.Size_X || cTensor_Input.Size_W!=cTensor_ValueX.Size_W || cTensor_ValueX.Size_Y!=1)
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
//функция CUDA для разделения тензора на Q,K,V
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSplitQKVTensorFunction(STensorKernel<type_t> tensor_q,STensorKernel<type_t> tensor_k,STensorKernel<type_t> tensor_v,STensorKernel<type_t> tensor_qkv,int32_t num_total_tokens,int32_t head_dim,int32_t arch_dim,int32_t q_split_index,int32_t k_split_index,int32_t v_split_index)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t zp=Mod(blockIdx.x,tensor_k.Size_Z);
 uint32_t wp=blockIdx.x/tensor_k.Size_Z;
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 type_t q=0;
 type_t k=0;
 type_t v=0;
 if (yp<num_total_tokens)
 {
  q=tensor_qkv.GetElement(wp,0,yp,q_split_index*arch_dim+zp*head_dim+xp);
  k=tensor_qkv.GetElement(wp,0,yp,k_split_index*arch_dim+zp*head_dim+xp);
  v=tensor_qkv.GetElement(wp,0,yp,v_split_index*arch_dim+zp*head_dim+xp);
 }

 tensor_q.SetElement(wp,zp,yp,xp,q);
 tensor_k.SetElement(wp,zp,yp,xp,k);
 tensor_v.SetElement(wp,zp,yp,xp,v);
}

//----------------------------------------------------------------------------------------------------
//разделить тензор на Q,K,V
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SplitKQVTensor(CTensor<type_t> &cTensor_Q,CTensor<type_t> &cTensor_K,CTensor<type_t> &cTensor_V,const CTensor<type_t> &cTensor_QKV,int32_t num_total_tokens,int32_t head_dim,int32_t arch_dim,int32_t q_split_index,int32_t k_split_index,int32_t v_split_index)
{
 cTensor_QKV.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_QKV(cTensor_QKV);
 STensorKernel<type_t> sTensorKernel_Q(cTensor_Q);
 STensorKernel<type_t> sTensorKernel_K(cTensor_K);
 STensorKernel<type_t> sTensorKernel_V(cTensor_V);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_K.Size_X/thread.x;
 if (cTensor_K.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_K.Size_Y/thread.y;
 if (cTensor_K.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_K.Size_Z*cTensor_K.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorSplitQKVTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Q,sTensorKernel_K,sTensorKernel_V,sTensorKernel_QKV,num_total_tokens,head_dim,arch_dim,q_split_index,k_split_index,v_split_index);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_K.SetDeviceOnChange();
 cTensor_Q.SetDeviceOnChange();
 cTensor_V.SetDeviceOnChange();
}


/*
//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления суммы элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t,uint32_t blockSize>
__global__ void CUDASumXYTensorFunction(uint32_t size,STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t w_in=Mod(blockIdx.y,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.y,tensor_output.Size_W);
 uint32_t z=Mod(blockIdx.z,tensor_output.Size_Z);

 volatile __shared__ type_t sdata[blockSize];

 uint32_t tid=threadIdx.x;
 uint32_t gridSize=blockSize*gridDim.x;
 uint32_t i=blockIdx.x*blockSize+tid;

 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z);
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z);

 sdata[tid]=0;

 while(i<size)
 {
  sdata[tid]+=d_xin[i];
  i+=gridSize;
 }
 __syncthreads();
 if (blockSize>=512)
 {
  if (tid<256) sdata[tid]+=sdata[tid+256];
 }
 __syncthreads();
 if (blockSize>=256)
 {
  if (tid<128) sdata[tid]+=sdata[tid+128];
 }
 __syncthreads();
 if (blockSize>=128)
 {
  if (tid<64) sdata[tid]+=sdata[tid+64];
 }
 __syncthreads();
 if (tid<32)
 {
  if (blockSize>=64) sdata[tid]+=sdata[tid+32];
  if (blockSize>=32) sdata[tid]+=sdata[tid+16];
  if (blockSize>=16) sdata[tid]+=sdata[tid+8];
  if (blockSize>=8)  sdata[tid]+=sdata[tid+4];
  if (blockSize>=4)  sdata[tid]+=sdata[tid+2];
  if (blockSize>=2)  sdata[tid]+=sdata[tid+1];
 }
 if (tid==0)
 {
  if (blockIdx.x<size) d_xout[blockIdx.x]=sdata[0];
 }
 __syncthreads();
}


//----------------------------------------------------------------------------------------------------
//вычислить сумму элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SumXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::SumXY: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 uint32_t input_x=cTensor_Input.Size_X;
 uint32_t input_y=cTensor_Input.Size_Y;
 uint32_t input_z=cTensor_Input.Size_Z;
 uint32_t input_w=cTensor_Input.Size_W;

 cTensor_Input.ReinterpretSize(input_w,input_z,1,input_y*input_x);

 CTensor<type_t> cTensor_InputCopy(cTensor_Input.Size_W,cTensor_Input.Size_Z,cTensor_Input.Size_Y,cTensor_Input.Size_X);

 int32_t amount=input_y*input_x;

 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_InputCopy(cTensor_InputCopy);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

 uint32_t size=amount;
 const uint32_t grid_Size=256;
 const uint32_t block_Size=128;

 {
  dim3 block(grid_Size,cTensor_Input.Size_W,cTensor_Input.Size_Z);
  dim3 thread(block_Size);
  CUDASumXYTensorFunction<type_t,block_Size><<<block,thread>>>(size,sTensorKernel_InputCopy,sTensorKernel_Input);
  cudaDeviceSynchronize();
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
 }
 {
  dim3 block(1,cTensor_Input.Size_W,cTensor_Input.Size_Z);
  dim3 thread(block_Size);
  if (size>grid_Size) size=grid_Size;
  CUDASumXYTensorFunction<type_t,block_Size><<<block,thread>>>(size,sTensorKernel_Input,sTensorKernel_InputCopy);
  cudaDeviceSynchronize();
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
 }

 cTensor_Input.SetDeviceOnChange();
 for(size_t w=0;w<cTensor_Input.Size_W;w++)
 {
  for(size_t z=0;z<cTensor_Input.Size_Z;z++)
  {
   type_t value=cTensor_Input.GetElement(w,z,0,0);
   cTensor_Output.SetElement(w,z,0,0,value);
  }
 }
 cTensor_Input.ReinterpretSize(input_w,input_z,input_y,input_x);
}
*/

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
 STensorTransposeKernel<type_t> sTensorTransposeKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);
 MulAbstract<STensorKernel<type_t>,STensorTransposeKernel<type_t>,STensorKernel<type_t>>(cTensor_Output,sTensorKernel_Output,cTensor_Left,sTensorTransposeKernel_Left,cTensor_Right,sTensorKernel_Right);
}
//----------------------------------------------------------------------------------------------------
//умножить левый тензор на транспонированный правый
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::RightTransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorTransposeKernel<type_t> sTensorTransposeKernel_Right(cTensor_Right);
 MulAbstract<STensorKernel<type_t>,STensorKernel<type_t>,STensorTransposeKernel<type_t>>(cTensor_Output,sTensorKernel_Output,cTensor_Left,sTensorKernel_Left,cTensor_Right,sTensorTransposeKernel_Right);
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
//транспонировать тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Transpose(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Output.Size_Y!=cTensor_Input.Size_X || cTensor_Output.Size_X!=cTensor_Input.Size_Y || cTensor_Output.Size_Z!=cTensor_Input.Size_Z || cTensor_Output.Size_W!=cTensor_Input.Size_W)
 {
  throw "void CTensor::Transpose: Размерности матриц не совпадают!";
 }
 cTensor_Input.CopyFromDevice();

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  for(uint32_t z=0;z<cTensor_Input.Size_Z;z++)
  {
   const type_t *i_ptr=cTensor_Input.GetColumnPtr(w,z,0);
   type_t *o_ptr=cTensor_Output.GetColumnPtr(w,z,0);
   for(uint32_t y=0;y<cTensor_Input.Size_Y;y++,o_ptr++)
   {
    type_t *o_ptr_local=o_ptr;
    for(uint32_t x=0;x<cTensor_Input.Size_X;x++,o_ptr_local+=cTensor_Input.Size_Y,i_ptr++)
    {
     *o_ptr_local=*i_ptr;
    }
   }
  }
 }
 cTensor_Output.SetHostOnChange();
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
//получить транспонированный тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> CTensorMath<type_t>::Transpose(const CTensor<type_t> &cTensor_Input)
{
 CTensor<type_t> cTensor(cTensor_Input.Size_W,cTensor_Input.Size_Z,cTensor_Input.Size_X,cTensor_Input.Size_Y);
 Transpose(cTensor,cTensor_Input);
 return(cTensor);
}

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
void CTensorMath<type_t>::UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y)
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
void CTensorMath<type_t>::DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y)
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

//----------------------------------------------------------------------------------------------------
//функция CUDA для уменьшение разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMaxPoolingTensor(STensorKernel<type_t> tensor_output,STensorKernel<typename CTensorMath<type_t>::SPos> tensor_position,STensorKernel<type_t> tensor_input,uint32_t pooling_x,uint32_t pooling_y)
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

 uint32_t ixp=xp*pooling_x;
 uint32_t iyp=yp*pooling_y;

 uint32_t offset_output=xp+yp*tensor_output.Size_X;
 uint32_t offset_input=ixp+iyp*tensor_input.Size_X;

 type_t *i_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset_input;
 type_t *o_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset_output;

 type_t max_e=*i_ptr;
 uint32_t max_x=ixp;
 uint32_t max_y=iyp;
 for(uint32_t y=0;y<pooling_y;y++,i_ptr+=tensor_input.StrideX)
 {
  type_t *i_ptr_local=i_ptr;
  for(uint32_t x=0;x<pooling_x;x++,i_ptr_local++)
  {
   type_t e=*i_ptr_local;
   if (e>max_e)
   {
    max_e=e;
    max_x=ixp+x;
    max_y=iyp+y;
   }
  }
 }

 *o_ptr=max_e;
 typename CTensorMath<type_t>::SPos sPos;
 sPos.X=max_x;
 sPos.Y=max_y;
 tensor_position.SetElement(w_out,z,yp,xp,sPos);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!увеличение разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPooling(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y)
{
 if ((cTensor_Input.Size_X/pooling_x)!=cTensor_Output.Size_X || (cTensor_Input.Size_Y/pooling_y)!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Position.Size_W!=cTensor_Output.Size_W)
 {
  throw "CTensor::MaxPooling: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_Position.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<SPos> sTensorKernel_Position(cTensor_Position);

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
 CUDAMaxPoolingTensor<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Position,sTensorKernel_Input,pooling_x,pooling_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
 cTensor_Position.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для обратного прохода при уменьшении разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMaxPoolingTensorBackward(STensorKernel<type_t> tensor_output,STensorKernel<typename CTensorMath<type_t>::SPos> tensor_position,STensorKernel<type_t> tensor_input,uint32_t pooling_x,uint32_t pooling_y)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=blockIdx.x/tensor_output.Size_Z;
 uint32_t w_out=Mod(w,tensor_output.Size_W);
 uint32_t w_in=Mod(w,tensor_input.Size_W);
 uint32_t w_pos=Mod(w,tensor_position.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 uint32_t ixp=xp/pooling_x;
 uint32_t iyp=yp/pooling_y;

 type_t value=tensor_input.GetElement(w_in,z,iyp,ixp);
 typename CTensorMath<type_t>::SPos sPos=tensor_position.TensorData_Ptr[w_pos*tensor_position.StrideW+z*tensor_position.StrideZ+iyp*tensor_position.StrideX+ixp];
 if (sPos.X!=xp || sPos.Y!=yp) value=0;
 tensor_output.SetElement(w_out,z,yp,xp,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!обратный проход при увеличении разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y)
{
 if (cTensor_Input.Size_X!=(cTensor_Output.Size_X/pooling_x) || cTensor_Input.Size_Y!=(cTensor_Output.Size_Y/pooling_y) || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::MaxPooling: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_Position.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<SPos> sTensorKernel_Position(cTensor_Position);

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
 CUDAMaxPoolingTensorBackward<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Position,sTensorKernel_Input,pooling_x,pooling_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
 cTensor_Position.SetDeviceOnChange();
}


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
void CTensorMath<type_t>::Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value)
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


//----------------------------------------------------------------------------------------------------
//функция CUDA для выполнения алгоритма Adam
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAAdam(STensorKernel<type_t> tensor_weight,STensorKernel<type_t> tensor_dweight,STensorKernel<type_t> tensor_m,STensorKernel<type_t> tensor_v,uint32_t batch_size,double speed,double beta1,double beta2,double epsilon,double db1,double db2)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_weight.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_weight.Size_Z),tensor_weight.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_weight.Size_X || yp>=tensor_weight.Size_Y) return;

 tensor_dweight.SelectW(w);
 tensor_dweight.SelectZ(z);

 tensor_weight.SelectW(w);
 tensor_weight.SelectZ(z);

 tensor_m.SelectW(w);
 tensor_m.SelectZ(z);

 tensor_v.SelectW(w);
 tensor_v.SelectZ(z);

 type_t dweight=tensor_dweight.GetElement(yp,xp);
 type_t m=tensor_m.GetElement(yp,xp);
 type_t v=tensor_v.GetElement(yp,xp);

 dweight/=static_cast<type_t>(batch_size);

 m=beta1*m+(1.0-beta1)*dweight;
 v=beta2*v+(1.0-beta2)*dweight*dweight;

 type_t mc=m/db1;
 type_t vc=v/db2;

 dweight=speed*mc/(sqrt(vc)+epsilon);

 tensor_m.SetElement(yp,xp,m);
 tensor_v.SetElement(yp,xp,v);

 //корректируем веса
 type_t weight=tensor_weight.GetElement(yp,xp);
 tensor_weight.SetElement(yp,xp,weight-dweight);
}

//----------------------------------------------------------------------------------------------------
//!выполнить алгоритм Adam к весовому тензору
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Adam(CTensor<type_t> &cTensor_Weight,CTensor<type_t> &cTensor_dWeight,CTensor<type_t> &cTensor_M,CTensor<type_t> &cTensor_V,uint32_t batch_size,double speed,double beta1,double beta2,double epsilon,double iteration)
{
 if (cTensor_Weight.Size_X!=cTensor_dWeight.Size_X || cTensor_Weight.Size_Y!=cTensor_dWeight.Size_Y || cTensor_Weight.Size_Z!=cTensor_dWeight.Size_Z || cTensor_Weight.Size_W!=cTensor_dWeight.Size_W ||
     cTensor_Weight.Size_X!=cTensor_M.Size_X || cTensor_Weight.Size_Y!=cTensor_M.Size_Y || cTensor_Weight.Size_Z!=cTensor_M.Size_Z || cTensor_Weight.Size_W!=cTensor_M.Size_W ||
     cTensor_Weight.Size_X!=cTensor_V.Size_X || cTensor_Weight.Size_Y!=cTensor_V.Size_Y || cTensor_Weight.Size_Z!=cTensor_V.Size_Z || cTensor_Weight.Size_W!=cTensor_V.Size_W)
 {
  throw "CTensor::Adam: Размерности тензоров не совпадают!";
 }

 cTensor_Weight.CopyToDevice();
 cTensor_dWeight.CopyToDevice();
 cTensor_M.CopyToDevice();
 cTensor_V.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Weight(cTensor_Weight);
 STensorKernel<type_t> sTensorKernel_dWeight(cTensor_dWeight);
 STensorKernel<type_t> sTensorKernel_M(cTensor_M);
 STensorKernel<type_t> sTensorKernel_V(cTensor_V);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Weight.Size_X/thread.x;
 if (cTensor_Weight.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Weight.Size_Y/thread.y;
 if (cTensor_Weight.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Weight.Size_Z*cTensor_Weight.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 iteration+=1;//чтобы избежать деления на 0 при 0 итерации
 double db1=1.0-pow(beta1,iteration);
 double db2=1.0-pow(beta2,iteration);

 CUDAAdam<type_t><<<blocks,thread>>>(sTensorKernel_Weight,sTensorKernel_dWeight,sTensorKernel_M,sTensorKernel_V,batch_size,speed,beta1,beta2,epsilon,db1,db2);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Weight.SetDeviceOnChange();
 cTensor_M.SetDeviceOnChange();
 cTensor_V.SetDeviceOnChange();
}

/*
//----------------------------------------------------------------------------------------------------
//функция CUDA для добавления временного шага
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASetTimeStep(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<uint32_t> tensor_time_step,type_t scale)
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

 uint32_t size=tensor_input.Size_X*tensor_input.Size_Y*tensor_input.Size_Z;
 uint32_t pos=xp+yp*tensor_input.Size_X+z*tensor_input.Size_X*tensor_input.Size_Y;

 type_t time_step=tensor_time_step.GetElement(w,0,0,0);

 type_t angle=time_step/pow(10000.0f,static_cast<type_t>(pos)/static_cast<type_t>(size));
 type_t value=tensor_input.GetElement(w,z,yp,xp);
 if ((pos&0x01)==0) value+=sin(angle)*scale;//чётное
               else value+=cos(angle)*scale;//нечётное
 tensor_output.SetElement(w,z,yp,xp,value);
 __syncthreads();
}*/


//----------------------------------------------------------------------------------------------------
// Функция CUDA для добавления временного шага
//
// ВАЖНО: временное кодирование зависит ТОЛЬКО от канала z и шага t.
// Оно одинаково для всех пространственных позиций (xp, yp).
// Формула (стандарт "Attention is All You Need"):
//   emb[2i]  =sin( t/10000^(2i/C) )
//   emb[2i+1]=cos( t/10000^(2i/C) )
// где C — число каналов тензора, i=z/2.
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASetTimeStep(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<uint32_t> tensor_time_step,type_t scale)
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

 //шаг времени для данного элемента пакета (w — индекс картинки в батче)
 type_t time_step=static_cast<type_t>(tensor_time_step.GetElement(w,0,0,0));

 // --- Временное кодирование: зависит только от z и t ---
 const uint32_t C=tensor_input.Size_Z;//число каналов
 const uint32_t i=z/2;//индекс "полупары" sin/cos
 const type_t exponent=static_cast<type_t>(2*i)/static_cast<type_t>(C);
 const type_t freq=pow(10000.0f, exponent);
 const type_t angle=time_step/freq;

 type_t emb_z;
 if ((z & 0x01)==0) emb_z=sin(angle)*scale;//чётный канал
               else emb_z=cos(angle)*scale;//нечётный канал

 //добавление одинакового значения ко всем пикселям канала z
 type_t value=tensor_input.GetElement(w, z, yp, xp);
 value+=emb_z;
 tensor_output.SetElement(w, z, yp, xp, value);
}

//----------------------------------------------------------------------------------------------------
//!добавить к тензору временной шаг
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SetTimeStep(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,const CTensor<uint32_t> &cTensor_TimeStep,type_t scale)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z ||
     cTensor_Input.Size_W!=cTensor_Output.Size_W || cTensor_TimeStep.Size_W!=cTensor_Input.Size_W || cTensor_TimeStep.Size_X!=1 ||
     cTensor_TimeStep.Size_Y!=1 || cTensor_TimeStep.Size_Z!=1)
 {
  throw "CTensor::SetTimeStep: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_Output.CopyToDevice();
 cTensor_TimeStep.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<uint32_t> sTensorKernel_TimeStep(cTensor_TimeStep);

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
 CUDASetTimeStep<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,sTensorKernel_TimeStep,scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для заполнения матрицы исключения
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADropOut(STensorKernel<type_t> tensor_output,unsigned long long seed,float drop_out)
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

 uint32_t pos=(xp+yp*tensor_output.Size_X+z*tensor_output.Size_X*tensor_output.Size_Y)+w*tensor_output.Size_X*tensor_output.Size_Y*tensor_output.Size_Z;

 curandStatePhilox4_32_10_t state;
 curand_init(seed,pos,0,&state);
 type_t random=curand_uniform(&state);
 /*
 curandState state;
 curand_init(seed,pos,0,&state);
 type_t random=curand_uniform(&state);
 */

 type_t mult=static_cast<type_t>(1.0/(1.0-drop_out));
 if (random>=drop_out) tensor_output.SetElement(w,z,yp,xp,mult);
                  else tensor_output.SetElement(w,z,yp,xp,0);
}
//----------------------------------------------------------------------------------------------------
//!создать матрицу исключения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::CreateDropOutMatrix(CTensor<type_t> &cTensor_Output,type_t drop_out)
{
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

 CUDADropOut<type_t><<<blocks,thread>>>(sTensorKernel_Output,rand(),drop_out);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для заполнения тензора случайными числами с нормальным распределением
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASetNormalNoise(STensorKernel<type_t> tensor_output,unsigned long long seed)
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

 uint32_t pos=xp+yp*tensor_output.Size_X+z*tensor_output.Size_X*tensor_output.Size_Y+w*tensor_output.Size_X*tensor_output.Size_Y*tensor_output.Size_Z;
 curandState state;
 curand_init(seed,pos,0,&state);
 type_t value=curand_normal(&state);
 tensor_output.SetElement(w,z,yp,xp,value);
}

//----------------------------------------------------------------------------------------------------
//!задать тензор случайными значениями с нормальным распределением
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SetNormalNoise(CTensor<type_t> &cTensor_Output)
{
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

 CUDASetNormalNoise<type_t><<<blocks,thread>>>(sTensorKernel_Output,rand());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для заполнения тензора случайными числами с нормальным распределением и создания изображения с шумом
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAGetNoiseImageAndNoise(STensorKernel<type_t> tensor_noisy_image,STensorKernel<type_t> tensor_noise,STensorKernel<type_t> tensor_image,STensorKernel<type_t> tensor_sqrt_alpha_bar,STensorKernel<type_t> tensor_sqrt_one_minus_alpha_bar,unsigned long long seed)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_noisy_image.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_noisy_image.Size_Z),tensor_noisy_image.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_noisy_image.Size_X || yp>=tensor_noisy_image.Size_Y) return;

 uint32_t pos=(xp+yp*tensor_noisy_image.Size_X+z*tensor_noisy_image.Size_X*tensor_noisy_image.Size_Y)+w*tensor_noisy_image.Size_X*tensor_noisy_image.Size_Y*tensor_noisy_image.Size_Z;


 curandStatePhilox4_32_10_t state;
 curand_init(seed,pos,0,&state);
 type_t noise=curand_normal(&state);

 /*curandState state;
 curand_init(seed,pos,0,&state);
 type_t noise=curand_normal(&state);*/

 tensor_noise.SetElement(w,z,yp,xp,noise);

 type_t image=tensor_image.GetElement(w,z,yp,xp);
 type_t sqrt_alpha_bar=tensor_sqrt_alpha_bar.GetElement(w,0,0,0);
 type_t sqrt_one_minus_alpha_bar=tensor_sqrt_one_minus_alpha_bar.GetElement(w,0,0,0);

 type_t value=sqrt_alpha_bar*image+sqrt_one_minus_alpha_bar*noise;

 tensor_noisy_image.SetElement(w,z,yp,xp,value);
}

//----------------------------------------------------------------------------------------------------
//!заполнить тензоры зашумлённым изображением и шумом
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::GetNoiseImageAndNoise(CTensor<type_t> &cTensor_NoisyImage,CTensor<type_t> &cTensor_Noise,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_SqrtAlphaBar,const CTensor<type_t> &cTensor_SqrtOneMinusAlphaBar)
{
 cTensor_Image.CopyToDevice();
 cTensor_SqrtAlphaBar.CopyToDevice();
 cTensor_SqrtOneMinusAlphaBar.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_NoisyImage(cTensor_NoisyImage);
 STensorKernel<type_t> sTensorKernel_Noise(cTensor_Noise);
 STensorKernel<type_t> sTensorKernel_Image(cTensor_Image);
 STensorKernel<type_t> sTensorKernel_SqrtAlphaBar(cTensor_SqrtAlphaBar);
 STensorKernel<type_t> sTensorKernel_SqrtOneMinusAlphaBar(cTensor_SqrtOneMinusAlphaBar);

 if (cTensor_NoisyImage.Size_W!=cTensor_Noise.Size_W || cTensor_NoisyImage.Size_X!=cTensor_Noise.Size_X || cTensor_NoisyImage.Size_Y!=cTensor_Noise.Size_Y || cTensor_NoisyImage.Size_Z!=cTensor_Noise.Size_Z)
 {
  throw "CTensor::GetNoiseImageAndNoise: Размерности тензоров NoisyImage и Noise не совпадают!";
 }
 if (cTensor_NoisyImage.Size_W!=cTensor_Image.Size_W || cTensor_NoisyImage.Size_X!=cTensor_Image.Size_X || cTensor_NoisyImage.Size_Y!=cTensor_Image.Size_Y || cTensor_NoisyImage.Size_Z!=cTensor_Image.Size_Z)
 {
  throw "CTensor::GetNoiseImageAndNoise: Размерности тензоров NoisyImage и Image не совпадают!";
 }
 if (cTensor_SqrtAlphaBar.Size_W!=cTensor_Image.Size_W || cTensor_SqrtOneMinusAlphaBar.Size_W!=cTensor_Image.Size_W)
 {
  throw "CTensor::GetNoiseImageAndNoise: Размерности тензоров SqrtAlphaBar/SqrtOneMinusAlphaBar и Image не совпадают!";
 }

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Noise.Size_X/thread.x;
 if (cTensor_Noise.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Noise.Size_Y/thread.y;
 if (cTensor_Noise.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Noise.Size_Z*cTensor_Noise.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 CUDAGetNoiseImageAndNoise<type_t><<<blocks,thread>>>(sTensorKernel_NoisyImage,sTensorKernel_Noise,sTensorKernel_Image,sTensorKernel_SqrtAlphaBar,sTensorKernel_SqrtOneMinusAlphaBar,rand());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Noise.SetDeviceOnChange();
 cTensor_NoisyImage.SetDeviceOnChange();
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
void CTensorMath<type_t>::ClipByNormXY(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold)
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
void CTensorMath<type_t>::ClipByNormX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold)
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


//----------------------------------------------------------------------------------------------------
//функция CUDA для прямого прохода GroupNorm
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAGroupNormForwardFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<type_t> tensor_gamma,STensorKernel<type_t> tensor_beta,STensorKernel<type_t> tensor_xhat,STensorKernel<type_t> tensor_inv_std,uint32_t channels_per_group,type_t epsilon)
{
 uint32_t w=blockIdx.x;
 uint32_t g=blockIdx.y;

 uint32_t Z=tensor_input.Size_Z;
 uint32_t Y=tensor_input.Size_Y;
 uint32_t X=tensor_input.Size_X;
 type_t M=static_cast<type_t>(channels_per_group)*static_cast<type_t>(Y)*static_cast<type_t>(X);
 //считаем среднее
 type_t sum=0;
 for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
 {
  type_t *ptr=tensor_input.GetTensorDataPtr(w,c);
  for(uint32_t i=0;i<Y*X;i++) sum+=ptr[i];
 }
 type_t mean=sum/M;
 //считаем дисперсию
 type_t sq_sum=0;
 for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
 {
  type_t *ptr=tensor_input.GetTensorDataPtr(w,c);
  for(uint32_t i=0;i<Y*X;i++)
  {
   type_t diff=ptr[i]-mean;
   sq_sum+=diff*diff;
  }
 }
 type_t var=sq_sum/M;
 if (var<0) var=0;
 type_t inv_std=1.0/sqrt(var+epsilon);
 tensor_inv_std.SetElement(w,g,0,0,inv_std);
 //нормализуем и масштабируем
 for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
 {
  type_t gamma=tensor_gamma.GetElement(0,c,0,0);
  type_t beta=tensor_beta.GetElement(0,c,0,0);
  type_t *in_ptr=tensor_input.GetTensorDataPtr(w,c);
  type_t *out_ptr=tensor_output.GetTensorDataPtr(w,c);
  type_t *xhat_ptr=tensor_xhat.GetTensorDataPtr(w,c);
  for(uint32_t i=0;i<Y*X;i++)
  {
   type_t xhat=(in_ptr[i]-mean)*inv_std;
   if (!isfinite(xhat)) xhat=0;
   if (!isfinite(in_ptr[i])) xhat=0;
   xhat_ptr[i]=xhat;
   type_t out_val=gamma*xhat+beta;
   if (!isfinite(out_val)) out_val=0;
   out_ptr[i]=out_val;
  }
 }
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для обратного прохода GroupNorm
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAGroupNormBackwardFunction(STensorKernel<type_t> tensor_dy,STensorKernel<type_t> tensor_xhat,STensorKernel<type_t> tensor_gamma,STensorKernel<type_t> tensor_inv_std,STensorKernel<type_t> tensor_dx,STensorKernel<type_t> tensor_dgamma,STensorKernel<type_t> tensor_dbeta,uint32_t channels_per_group,bool calc_weights)
{
 uint32_t w=blockIdx.x;
 uint32_t g=blockIdx.y;

 uint32_t W=tensor_dy.Size_W;
 uint32_t Y=tensor_dy.Size_Y;
 uint32_t X=tensor_dy.Size_X;
 type_t M=static_cast<type_t>(channels_per_group)*static_cast<type_t>(Y)*static_cast<type_t>(X);
 //масштабирующий коэффициент для усреднения (1/BatchSize*Y*X)
 type_t scale=1.0;// 1.0/(static_cast<type_t>(W)*static_cast<type_t>(Y)*static_cast<type_t>(X));
 type_t inv_std=tensor_inv_std.GetElement(w,g,0,0);
 //первый проход: суммируем градиенты по группе
 type_t sum_dxhat=0;
 type_t sum_dxhat_xhat=0;

 for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
 {
  type_t gamma=tensor_gamma.GetElement(0,c,0,0);
  type_t *dy_ptr=tensor_dy.GetTensorDataPtr(w,c);
  type_t *xhat_ptr=tensor_xhat.GetTensorDataPtr(w,c);
  for(uint32_t i=0;i<Y*X;i++)
  {
   type_t dy=dy_ptr[i];
   type_t xhat=xhat_ptr[i];
   type_t dxhat=dy*gamma;

   sum_dxhat+=dxhat;
   sum_dxhat_xhat+=dxhat*xhat;

   if (calc_weights)
   {
    type_t dg=dy*xhat*scale;
    type_t db=dy*scale;
    if (isfinite(dg)) atomicAdd(&tensor_dgamma.GetTensorDataPtr(0,c)[0],dg);
    if (isfinite(db)) atomicAdd(&tensor_dbeta.GetTensorDataPtr(0,c)[0],db);
   }
  }
 }
 //второй проход: вычисляем градиент для предыдущего слоя
 for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
 {
  type_t gamma=tensor_gamma.GetElement(0,c,0,0);
  type_t *dy_ptr=tensor_dy.GetTensorDataPtr(w,c);
  type_t *xhat_ptr=tensor_xhat.GetTensorDataPtr(w,c);
  type_t *dx_ptr=tensor_dx.GetTensorDataPtr(w,c);
  for(uint32_t i=0;i<Y*X;i++)
  {
   type_t dy=dy_ptr[i];
   type_t xhat=xhat_ptr[i];
   type_t dxhat=dy*gamma;
   type_t dx=(1.0/M)*inv_std*(M*dxhat-sum_dxhat-xhat*sum_dxhat_xhat);
   if (!isfinite(dx)) dx=0;
   dx_ptr[i]=dx;
  }
 }
}

//----------------------------------------------------------------------------------------------------
//!выполнить прямой проход GroupNorm
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::GroupNormForward(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_Gamma,CTensor<type_t> &cTensor_Beta,CTensor<type_t> &cTensor_XHAT,CTensor<type_t> &cTensor_InvStd,uint32_t num_groups,uint32_t channels_per_group,type_t epsilon)
{
 cTensor_Input.CopyToDevice();
 cTensor_Gamma.CopyToDevice();
 cTensor_Beta.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Gamma(cTensor_Gamma);
 STensorKernel<type_t> sTensorKernel_Beta(cTensor_Beta);
 STensorKernel<type_t> sTensorKernel_XHAT(cTensor_XHAT);
 STensorKernel<type_t> sTensorKernel_InvStd(cTensor_InvStd);

 dim3 blocks(cTensor_Input.Size_W,num_groups);
 dim3 threads(1,1);

 CUDAGroupNormForwardFunction<type_t><<<blocks,threads>>>(sTensorKernel_Output,sTensorKernel_Input,sTensorKernel_Gamma,sTensorKernel_Beta,sTensorKernel_XHAT,sTensorKernel_InvStd,channels_per_group,epsilon);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
 cTensor_XHAT.SetDeviceOnChange();
 cTensor_InvStd.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//!выполнить обратный проход GroupNorm
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::GroupNormBackward(CTensor<type_t> &cTensor_Delta_Array,CTensor<type_t> &cTensor_XHAT_Array,CTensor<type_t> &cTensor_Gamma,CTensor<type_t> &cTensor_InvStd_Array,CTensor<type_t> &cTensor_PrevLayerError_Array,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta,uint32_t num_groups,uint32_t channels_per_group,bool calc_weights)
{
 cTensor_Delta_Array.CopyToDevice();
 cTensor_XHAT_Array.CopyToDevice();
 cTensor_Gamma.CopyToDevice();
 cTensor_InvStd_Array.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Dy(cTensor_Delta_Array);
 STensorKernel<type_t> sTensorKernel_Xhat(cTensor_XHAT_Array);
 STensorKernel<type_t> sTensorKernel_Gamma(cTensor_Gamma);
 STensorKernel<type_t> sTensorKernel_InvStd(cTensor_InvStd_Array);
 STensorKernel<type_t> sTensorKernel_Dx(cTensor_PrevLayerError_Array);
 STensorKernel<type_t> sTensorKernel_Dgamma(cTensor_dGamma);
 STensorKernel<type_t> sTensorKernel_Dbeta(cTensor_dBeta);

 dim3 blocks(cTensor_Delta_Array.Size_W,num_groups);
 dim3 threads(1,1);

 CUDAGroupNormBackwardFunction<type_t><<<blocks,threads>>>(sTensorKernel_Dy,sTensorKernel_Xhat,sTensorKernel_Gamma,sTensorKernel_InvStd,sTensorKernel_Dx,sTensorKernel_Dgamma,sTensorKernel_Dbeta,channels_per_group,calc_weights);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_PrevLayerError_Array.SetDeviceOnChange();
 if (calc_weights)
 {
  cTensor_dGamma.SetDeviceOnChange();
  cTensor_dBeta.SetDeviceOnChange();
 }
}

#endif

#endif
