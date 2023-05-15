#ifndef C_CUDA_FUNCTION_H
#define C_CUDA_FUNCTION_H

//****************************************************************************************************
//Класс применения функции в CUDA
//****************************************************************************************************

#include <stdio.h>
#include <stdint.h>

#include "handle_error.cu.h"
#include "ctensor.cu.h"
#include "/../system/system.h"

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************

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
//класс применения функции сигмоид в CUDA
//****************************************************************************************************
template<class type_t>
class CCUDAFunction
{
 //-дружественные функции-------------------------------------------------------------------------------
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 public:
  //-переменные-----------------------------------------------------------------------------------------
  CCUDATensorKit<type_t> *cCUDATensorKit_Input_Ptr;//набор входных данных
  CCUDATensorKit<type_t> *cCUDATensorKit_Output_Ptr;//набор выходных данных
  size_t TensorAmount;//на сколько матриц создан набор
 private:
  public:
  //-конструктор----------------------------------------------------------------------------------------
  __host__ CCUDAFunction(size_t matrix_amount=0);
  //-деструктор-----------------------------------------------------------------------------------------
  __host__ ~CCUDAFunction();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ void Release(void);//очистить память

  __host__ void SetTensorAmount(size_t matrix_amount);//задать количество матриц в наборе

  __host__ void ApplySigmoid(void);//применить функцию сигмоид
  __host__ void ApplyReLU(void);//применить функцию ReLU
  __host__ void ApplyLeakyReLU(void);//применить функцию Leaky ReLU
  __host__ void ApplyLinear(void);//применить линейную функцию
  __host__ void ApplyTangence(void);//применить функцию гиперболический тангенс
  __host__ void ApplySoftMax(void);//применить функцию softmax

  __host__ void ApplyDifferentialSigmoid(void);//применить функцию производной от сигмоида
  __host__ void ApplyDifferentialReLU(void);//применить функцию производной от ReLU
  __host__ void ApplyDifferentialLeakyReLU(void);//применить функцию производной от Leaky ReLU
  __host__ void ApplyDifferentialLinear(void);//применить производную линейной функций
  __host__ void ApplyDifferentialTangence(void);//применить функцию производной от гиперболического тангенса
  __host__ void ApplyDifferentialSoftMax(void);//применить функцию производной от softmax

  __host__ __device__ static type_t Sigmoid(type_t v);//сигмоид
  __host__ __device__ static type_t ReLU(type_t v);//ReLU
  __host__ __device__ static type_t LeakyReLU(type_t v);//Leaky ReLU
  __host__ __device__ static type_t Linear(type_t v);//линейная
  __host__ __device__ static type_t Tangence(type_t v);//гиперболический тангенс

  __host__ __device__ static type_t dSigmoid(type_t v);//производная сигмоида
  __host__ __device__ static type_t dReLU(type_t v);//производная ReLU
  __host__ __device__ static type_t dLeakyReLU(type_t v);//производная Leaky ReLU
  __host__ __device__ static type_t dLinear(type_t v);//производная линейной функции
  __host__ __device__ static type_t dTangence(type_t v);//производная гиперболического тангенса
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
  __host__ void Init(void);//инициализация
};
//****************************************************************************************************
//конструктор и деструктор класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDAFunction<type_t>::CCUDAFunction(size_t matrix_amount)
{
 TensorAmount=matrix_amount;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDAFunction<type_t>::~CCUDAFunction()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::Sigmoid(type_t v)
{
 //if (v>20) v=19.9;
 //if (v<-20) v=-19.9;
 return(1.0/(1.0+exp(-v)));
}
//----------------------------------------------------------------------------------------------------
//ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::ReLU(type_t v)
{
 if (v>0) return(v);
 return(0);
}
//----------------------------------------------------------------------------------------------------
//Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::LeakyReLU(type_t v)
{
 if (v>0) return(v);
 return(0.1*v);
}
//----------------------------------------------------------------------------------------------------
//линейная
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::Linear(type_t v)
{
 return(v);
}
//----------------------------------------------------------------------------------------------------
//гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::Tangence(type_t v)
{
 //if (v>20) v=19.9;
 //if (v<-20) v=-19.9;
 type_t ep=exp(2*v);
 type_t en=exp(-2*v);
 return((ep-en)/(ep+en));
}
//----------------------------------------------------------------------------------------------------
//производная сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dSigmoid(type_t v)
{
 type_t s=Sigmoid(v);
 return((1.0-s)*s);
}
//----------------------------------------------------------------------------------------------------
//производная ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dReLU(type_t v)
{
 if (v>=0) return(1);
 return(0);
}
//----------------------------------------------------------------------------------------------------
//производная Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dLeakyReLU(type_t v)
{
 if (v>=0) return(1);
 return(0.1);
}
//----------------------------------------------------------------------------------------------------
//производная линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dLinear(type_t v)
{
 return(1);
}
//----------------------------------------------------------------------------------------------------
//производная гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dTangence(type_t v)
{
 type_t t=Tangence(v);
 return(1-t*t);
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для применения функции сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASigmoidFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=CCUDAFunction<type_t>::Sigmoid(value);
  }
}


//----------------------------------------------------------------------------------------------------
//применить функцию сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplySigmoid(void)
{
 Init();
 CUDASigmoidFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для применения функции ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAReLUFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=CCUDAFunction<type_t>::ReLU(value);
  }
}

//----------------------------------------------------------------------------------------------------
//применить функцию ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyReLU(void)
{
 Init();
 CUDAReLUFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для применения функции Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDALeakyReLUFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=CCUDAFunction<type_t>::LeakyReLU(value);
  }
}

//----------------------------------------------------------------------------------------------------
//применить функцию Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyLeakyReLU(void)
{
 Init();

 CUDALeakyReLUFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для применения линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDALinearFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=CCUDAFunction<type_t>::Linear(value);
  }
}

//----------------------------------------------------------------------------------------------------
//применить линейную функцию
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyLinear(void)
{
 Init();
 CUDALinearFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для применения функции гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATangenceFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=CCUDAFunction<type_t>::Tangence(value);
  }
}

//----------------------------------------------------------------------------------------------------
//применить функцию гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyTangence(void)
{
 Init();
 CUDATangenceFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для применения функции softmax
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASoftMaxFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

 type_t *m_input_y_ptr=m_input_ptr+y*input_x;
 double summ=0;
 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   summ+=exp(value);
  }
 }
 double value=exp(*m_input_y_ptr)/summ;
 for(size_t x=0;x<input_x;x++,m_output_ptr++) *m_output_ptr=static_cast<type_t>(value);
}

//----------------------------------------------------------------------------------------------------
//применить функцию softmax
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplySoftMax(void)
{
 Init();
 CUDASoftMaxFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}




//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной функции сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialSigmoidFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=CCUDAFunction<type_t>::dSigmoid(value);
  }
}

//----------------------------------------------------------------------------------------------------
//применить функцию производной от сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialSigmoid(void)
{
 Init();

 CUDADifferentialSigmoidFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной функции ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialReLUFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=CCUDAFunction<type_t>::dReLU(value);
  }
}

//----------------------------------------------------------------------------------------------------
//применить функцию производной от ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialReLU(void)
{
 Init();
 CUDADifferentialReLUFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной функции Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialLeakyReLUFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

 for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
 {
  type_t value=*m_input_ptr;
  *m_output_ptr=CCUDAFunction<type_t>::dLeakyReLU(value);
 }
}

//----------------------------------------------------------------------------------------------------
//применить функцию производной от Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialLeakyReLU(void)
{
 Init();

 CUDADifferentialLeakyReLUFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialLinearFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=CCUDAFunction<type_t>::dLinear(value);
  }
}

//----------------------------------------------------------------------------------------------------
//применить производную линейной функций
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialLinear(void)
{
 Init();
 CUDADifferentialLinearFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной функции гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialTangenceFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=CCUDAFunction<type_t>::dTangence(value);
  }
}

//----------------------------------------------------------------------------------------------------
//применить функцию производной от гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialTangence(void)
{
 Init();
 CUDADifferentialTangenceFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной функции softmax
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialSoftMaxFunction(STensorKernel<type_t> matrix_input,STensorKernel<type_t> matrix_output)
{
 size_t index=blockIdx.x;
 size_t y=threadIdx.x;

 size_t input_x=matrix_input.GetSizeX();
 size_t input_y=matrix_input.GetSizeY();

 type_t *m_output_ptr=matrix_output.GetTensorDataPtr(index)+y*input_x;
 type_t *m_input_ptr=matrix_input.GetTensorDataPtr(index)+y*input_x;

 type_t *m_input_y_ptr=m_input_ptr+y*input_x;
 double summ=0;
 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   summ+=exp(value);
  }
 }
 double value=exp(*m_input_y_ptr)/summ;
 value=value*(1-value);
 for(size_t x=0;x<input_x;x++,m_output_ptr++) *m_output_ptr=static_cast<type_t>(value);
}

//----------------------------------------------------------------------------------------------------
//применить функцию производной от softmax
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialSoftMax(void)
{
 Init();
 CUDADifferentialSoftMaxFunction<<<TensorAmount,cCUDATensorKit_Input_Ptr->GetSizeY()>>>(cCUDATensorKit_Input_Ptr->GetTensorKernel(),cCUDATensorKit_Output_Ptr->GetTensorKernel());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}

//----------------------------------------------------------------------------------------------------
//инициализация
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::Init(void)
{
 if (cCUDATensorKit_Input_Ptr->GetAmount()!=TensorAmount) throw "CUDAFunction<type_t>::Init: количество матриц в наборе должно быть равно количеству матриц, для которого создавался класс";
 /*
 //зачем нужны были эти строки уже неизвестно
 //задаём выходную матрицу
 cCUDATensorKit_Output_Ptr->Release();
 CCUDATensorKit<type_t> cCUDATensorKit(cCUDATensorKit_Input_Ptr->GetSizeY(),cCUDATensorKit_Input_Ptr->GetSizeX(),TensorAmount);
 cCUDATensorKit.Create();
 cCUDATensorKit_Output_Ptr->Move(cCUDATensorKit);
 */
}
 //****************************************************************************************************
//открытые функции класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//очистить память
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::Release(void)
{
 TensorAmount=0;
}

//----------------------------------------------------------------------------------------------------
//задать количество матриц в наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::SetTensorAmount(size_t matrix_amount)
{
 TensorAmount=matrix_amount;
}

#endif
