#ifndef C_TENSOR_APPLY_FUNC_CU_H
#define C_TENSOR_APPLY_FUNC_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensorapplyfunc.h"

#endif


#ifndef USE_CPU

//****************************************************************************************************
//Применение функций к тензорам произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "ctensor.cu.h"
#include "ctensormath.cu.h"

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
///!Применение функций к тензорам произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorApplyFunc
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
  __host__ __device__ static type_t tanh(type_t v);///<гиперболический тангенс
  __host__ __device__ static type_t ch(type_t v);///<гиперболический косинус
  __host__ __device__ static type_t sech(type_t v);///<гиперболический секанс

  __host__ __device__ static type_t Sigmoid(type_t v);///<сигмоид
  __host__ __device__ static type_t ReLU(type_t v);///<ReLU
  __host__ __device__ static type_t GeLU(type_t v);///<GeLU
  __host__ __device__ static type_t LeakyReLU(type_t v);///<Leaky ReLU
  __host__ __device__ static type_t Linear(type_t v);///<линейная
  __host__ __device__ static type_t Tangence(type_t v);///<гиперболический тангенс

  __host__ __device__ static type_t dSigmoid(type_t v);///<производная сигмоида
  __host__ __device__ static type_t dReLU(type_t v);///<производная ReLU
  __host__ __device__ static type_t dGeLU(type_t v);///<производная GeLU
  __host__ __device__ static type_t dLeakyReLU(type_t v);///<производная Leaky ReLU
  __host__ __device__ static type_t dLinear(type_t v);///<производная линейной функции
  __host__ __device__ static type_t dTangence(type_t v);///<производная гиперболического тангенса

  static void ApplySigmoid(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию сигмоид
  static void ApplyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию ReLU
  static void ApplyGeLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию GeLU
  static void ApplyLeakyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию Leaky ReLU
  static void ApplyLinear(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить линейную функцию
  static void ApplyTangence(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию гиперболический тангенс
   //static void ApplySoftMax(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию softmax

  static void ApplyDifferentialSigmoid(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию  производной от сигмоида
  static void ApplyDifferentialReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию производной от ReLU
  static void ApplyDifferentialGeLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию производной от GeLU
  static void ApplyDifferentialLeakyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию производной от Leaky ReLU
  static void ApplyDifferentialLinear(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить производную линейной функций
  static void ApplyDifferentialTangence(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию производной от гиперболического тангенса
   //static void ApplyDifferentialSoftMax(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию производной от softmax

 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------------------------

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//****************************************************************************************************
//статические функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
///!гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::tanh(type_t v)
{
 return(::tanh(v));
}
//----------------------------------------------------------------------------------------------------
///!гиперболический косинус
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::ch(type_t v)
{
 return(cosh(v));
}
//----------------------------------------------------------------------------------------------------
///!гиперболический секанс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::sech(type_t v)
{
 return(1/ch(v));
}

//----------------------------------------------------------------------------------------------------
///!сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::Sigmoid(type_t v)
{
 if (v>30) v=29.99999;
 if (v<-30) v=-29.99999;

 return(1.0/(1.0+exp(-v)));
}
//----------------------------------------------------------------------------------------------------
///!ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::ReLU(type_t v)
{
 if (v>0) return(v);
 return(0);
}
//----------------------------------------------------------------------------------------------------
///!GeLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::GeLU(type_t v)
{
 const type_t PI=3.141592654;

 type_t ft=sqrt(2/PI)*(v+0.044715*v*v*v);
 //гиперболический тангенс
 ft=tanh(ft);
 //вычисляем GeLU
 type_t f=0.5*v*(1+ft);
 return(f);
}

//----------------------------------------------------------------------------------------------------
///!Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::LeakyReLU(type_t v)
{
 if (v>0) return(v);
 return(0.1*v);
}
//----------------------------------------------------------------------------------------------------
///!линейная
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::Linear(type_t v)
{
 return(v);
}
//----------------------------------------------------------------------------------------------------
///!гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::Tangence(type_t v)
{
 if (v>30) v=29.99999;
 if (v<-30) v=-29.99999;

 //if (v>20) return(1);
 //if (v<-20) return(-1);
 return(tanh(v));
}
//----------------------------------------------------------------------------------------------------
///!производная сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::dSigmoid(type_t v)
{
 type_t s=Sigmoid(v);
 return((1.0-s)*s);
}
//----------------------------------------------------------------------------------------------------
///!производная ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::dReLU(type_t v)
{
 if (v>=0) return(1);
 return(0);
}
//----------------------------------------------------------------------------------------------------
///!производная GeLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::dGeLU(type_t v)
{
 const type_t PI=3.141592654;

 type_t ft=sqrt(2/PI)*(v+0.044715*v*v*v);
 type_t fa=sqrt(2/PI)*(1+3*0.044715*v*v);
 //вычисляем производную GeLU
 type_t f=0.5*((1+tanh(ft))+0.5*v*sech(ft)*sech(ft))*fa;
 return(f);
}
//----------------------------------------------------------------------------------------------------
///!производная Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::dLeakyReLU(type_t v)
{
 if (v>=0) return(1);
 return(0.1);
}
//----------------------------------------------------------------------------------------------------
///!производная линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::dLinear(type_t v)
{
 return(1);
}
//----------------------------------------------------------------------------------------------------
///!производная гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CTensorApplyFunc<type_t>::dTangence(type_t v)
{
 type_t t=Tangence(v);
 return(1-t*t);
}

//----------------------------------------------------------------------------------------------------
///!инициализация блоков вычислений CUDA
//----------------------------------------------------------------------------------------------------
template<class type_t>
static void BlockInitFunction(dim3 &thread,dim3 &blocks,const CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Output.GetSizeX()!=cTensor_Input.GetSizeX() || cTensor_Output.GetSizeY()!=cTensor_Input.GetSizeY() || cTensor_Output.GetSizeZ()!=cTensor_Input.GetSizeZ())
 {
  throw "BlockInitFunction: Размерности тензоров не совпадают!";
 }

 thread=dim3(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_x=cTensor_Output.GetSizeX()/thread.x;
 if (cTensor_Output.GetSizeX()%thread.x) block_x++;
 uint32_t block_y=cTensor_Output.GetSizeY()/thread.y;
 if (cTensor_Output.GetSizeY()%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.GetSizeZ()*cTensor_Output.GetSizeW();

 blocks=dim3(block_x,block_y,block_z);
}


//----------------------------------------------------------------------------------------------------
///!функция CUDA для сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplySigmoidFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::Sigmoid(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplySigmoid(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);

 CUDATensorApplySigmoidFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
///!функция CUDA для ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyReLUFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::ReLU(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyReLUFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
///!функция CUDA для GeLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyGeLUFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::GeLU(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию GeLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyGeLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyGeLUFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
///!функция CUDA для Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyLeakyReLUFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::LeakyReLU(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyLeakyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyLeakyReLUFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
///!функция CUDA для Linear
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyLinearFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::Linear(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить линейную функцию
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyLinear(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyLinearFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
///!функция CUDA для гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyTangenceFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::Tangence(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyTangence(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyTangenceFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
/*
//----------------------------------------------------------------------------------------------------
///!функция CUDA для SoftMax
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplySoftMaxFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::SoftMax(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию softmax
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplySoftMax(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplySoftMaxFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
*/

//----------------------------------------------------------------------------------------------------
///!функция CUDA для производной сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyDifferentialSigmoidFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::dSigmoid(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyDifferentialSigmoid(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyDifferentialSigmoidFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
//----------------------------------------------------------------------------------------------------
///!функция CUDA для производной ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyDifferentialReLUFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::dReLU(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyDifferentialReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyDifferentialReLUFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
///!функция CUDA для производной GeLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyDifferentialGeLUFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::dGeLU(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от GeLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyDifferentialGeLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyDifferentialGeLUFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
///!функция CUDA для производной Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyDifferentialLeakyReLUFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::dLeakyReLU(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyDifferentialLeakyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyDifferentialLeakyReLUFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
//----------------------------------------------------------------------------------------------------
///!функция CUDA для производной линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyDifferentialLinearFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::dLinear(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить производную линейной функций
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyDifferentialLinear(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyDifferentialLinearFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
//----------------------------------------------------------------------------------------------------
///!функция CUDA для производной гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyDifferentialTangenceFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::dTangence(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyDifferentialTangence(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyDifferentialTangenceFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
/*
//----------------------------------------------------------------------------------------------------
///!функция CUDA для производной SoftMax
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorApplyDifferentialSoftMaxFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z%tensor_output.GetSizeZ();
 uint32_t w_in=(blockIdx.z/tensor_output.GetSizeZ())%tensor_input.GetSizeW();
 uint32_t w_out=(blockIdx.z/tensor_output.GetSizeZ())%tensor_output.GetSizeW();
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=CTensorApplyFunc<type_t>::dSoftMax(*a_ptr);

 __syncthreads();
}
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от softmax
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorApplyFunc<type_t>::ApplyDifferentialSoftMax(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread;
 dim3 blocks;
 BlockInitFunction(thread,blocks,cTensor_Output,cTensor_Input);
 CUDATensorApplyDifferentialSoftMaxFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
*/

#endif

#endif
