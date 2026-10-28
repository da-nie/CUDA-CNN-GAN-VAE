#ifndef C_TENSOR_SOFTMAX_CU_H
#define C_TENSOR_SOFTMAX_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensorsoftmax.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции softmax над тензорами произвольной размерности
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
///!Операции softmax над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorSoftMax
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
  static void SoftMaxX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<SoftMax по оси X
  static void SoftMaxX_Backward(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Delta,CTensor<type_t> &cTensor_Input);//выполнить обратный проход SoftMax по слою X
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
// device-функции для exp и блочной редукции
//----------------------------------------------------------------------------------------------------
__device__ __forceinline__ float exp_op(float x)
{
    return __expf(x); // быстрая экспонента; если нужна повышенная точность — expf(x)
}

__device__ __forceinline__ double exp_op(double x)
{
    return exp(x);
}

template<class type_t>
__device__ __forceinline__ type_t blockReduceMax(type_t val, type_t *shared)
{
    const int32_t tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    for (int32_t s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            type_t other = shared[tid + s];
            if (other > shared[tid])
                shared[tid] = other;
        }
        __syncthreads();
    }

    return shared[0];
}

template<class type_t>
__device__ __forceinline__ type_t blockReduceSum(type_t val, type_t *shared)
{
    const int32_t tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    for (int32_t s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    return shared[0];
}

//----------------------------------------------------------------------------------------------------
// функция CUDA для вычисления softmax по слою X
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSoftMaxXTensorFunction(
    STensorKernel<type_t> tensor_output,
    STensorKernel<type_t> tensor_input)
{
    const int32_t w = blockIdx.x;
    const int32_t z = blockIdx.y;
    const int32_t y = blockIdx.z;

    type_t *d_xin  = tensor_input.GetTensorDataPtr(w, z)  + y * tensor_input.Size_X;
    type_t *d_xout = tensor_output.GetTensorDataPtr(w, z) + y * tensor_output.Size_X;

    const int32_t n       = tensor_input.Size_X;
    const int32_t tid     = threadIdx.x;
    const int32_t nthreads = blockDim.x;

    extern __shared__ type_t sdata[];
    type_t *s_max = sdata;
    type_t *s_sum = sdata + nthreads;

    //------------------------------------------------------------------
    // 1. Поиск максимума по строке
    //------------------------------------------------------------------
    type_t local_max = d_xin[0];
    for (int32_t i = tid; i < n; i += nthreads)
    {
        type_t v = d_xin[i];
        if (v > local_max)
            local_max = v;
    }

    type_t max_val = blockReduceMax(local_max, s_max);

    //------------------------------------------------------------------
    // 2. Сумма экспонент и запись exp в выходной тензор
    //------------------------------------------------------------------
    type_t local_sum = type_t(0);
    for (int32_t i = tid; i < n; i += nthreads)
    {
        type_t v = exp_op(d_xin[i] - max_val);
        d_xout[i] = v;
        local_sum += v;
    }

    type_t sum = blockReduceSum(local_sum, s_sum);

    //------------------------------------------------------------------
    // 3. Нормировка
    //------------------------------------------------------------------
    type_t inv_sum = type_t(1) / sum;
    for (int32_t i = tid; i < n; i += nthreads)
    {
        d_xout[i] *= inv_sum;
    }
}

//----------------------------------------------------------------------------------------------------
// SoftMax по оси X
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorSoftMax<type_t>::SoftMaxX(
    CTensor<type_t> &cTensor_Output,
    const CTensor<type_t> &cTensor_Input)
{
    if (cTensor_Output.Size_W != cTensor_Input.Size_W ||
        cTensor_Output.Size_Z != cTensor_Input.Size_Z ||
        cTensor_Output.Size_Y != cTensor_Input.Size_Y ||
        cTensor_Output.Size_X != cTensor_Input.Size_X)
    {
        throw "CTensorMath::SoftMaxX: Размерности тензоров входа и выхода не совпадают!";
    }

    cTensor_Input.CopyToDevice();

    STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
    STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

    // Выбираем число потоков: 256 или ближайшую степень двойки от Size_X
    int threads = 256;
    if (cTensor_Input.Size_X < threads)
    {
        threads = 1;
        while (threads < cTensor_Input.Size_X)
            threads <<= 1;
        if (threads < 32)
            threads = 32;
    }

    dim3 thread(threads, 1, 1);
    dim3 blocks(cTensor_Input.Size_W, cTensor_Input.Size_Z, cTensor_Input.Size_Y);

    const size_t shared_mem = 2 * threads * sizeof(type_t);

    CUDATensorSoftMaxXTensorFunction<type_t><<<blocks, thread, shared_mem>>>(
        sTensorKernel_Output,
        sTensorKernel_Input);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    cTensor_Output.SetDeviceOnChange();
}


/*
//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления softmax по слою X
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSoftMaxXTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t w_in=Mod(blockIdx.x,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.x,tensor_output.Size_W);
 uint32_t z=Mod(blockIdx.y,tensor_output.Size_Z);
 uint32_t y=Mod(blockIdx.z,tensor_output.Size_Y);

 //выполняем softmax
 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z)+y*tensor_input.Size_X;
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z)+y*tensor_input.Size_X;

 //численная стабилизация — находим максимум
 type_t max_val=*d_xin;

 type_t *d_xin_local=d_xin;
 for(int32_t x=0;x<tensor_input.Size_X;x++,d_xin_local++)
 {
  type_t val=*d_xin_local;
  if (val>max_val) max_val=val;
 }
 //считаем сумму стабилизированных экспонент
 double summ=0;
 d_xin_local=d_xin;
 type_t *d_xout_local=d_xout;
 for(int32_t x=0;x<tensor_input.Size_X;x++,d_xin_local++,d_xout_local++)
 {
  type_t e=*d_xin_local;
  e-=max_val;
  type_t v=exp(e);
  summ+=v;
  *d_xout_local=v;
 }
 //делим каждый элемент на сумму
 for(int32_t x=0;x<tensor_input.Size_X;x++,d_xout++) *d_xout/=summ;
}

//----------------------------------------------------------------------------------------------------
//SoftMax по оси X
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorSoftMax<type_t>::SoftMaxX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Output.Size_W!=cTensor_Input.Size_W || cTensor_Output.Size_Z!=cTensor_Input.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Input.Size_Y || cTensor_Output.Size_X!=cTensor_Input.Size_X)
 {
  throw "CTensorMath::SoftMaxX: Размерности тензоров входа и выхода не совпадают!";
 }
 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input.Size_W,cTensor_Input.Size_Z,cTensor_Input.Size_Y);

 CUDATensorSoftMaxXTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
*/

//----------------------------------------------------------------------------------------------------
//функция CUDA для вобратного прохода SoftMax по слою X
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSoftMaxXBackwardFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_delta,STensorKernel<type_t> tensor_input)
{
 uint32_t w_in=Mod(blockIdx.x,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.x,tensor_output.Size_W);
 uint32_t w_delta=Mod(blockIdx.x,tensor_delta.Size_W);
 uint32_t z=Mod(blockIdx.y,tensor_input.Size_Z);
 uint32_t y=Mod(blockIdx.z,tensor_input.Size_Y);

 uint32_t size_x=tensor_input.Size_X;
 //вычисляем взвешенную сумму ошибок: sum_j (p_j * error_j)
 type_t weighted_error_sum=0;
 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z)+y*tensor_input.Size_X;
 type_t *d_delta=tensor_delta.GetTensorDataPtr(w_delta,z)+y*tensor_delta.Size_X;
 type_t *d_xin_local=d_xin;
 type_t *d_delta_local=d_delta;
 for(uint32_t x=0;x<size_x;x++,d_xin_local++,d_delta_local++)
 {
  type_t v=*d_xin_local;
  type_t d=*d_delta_local;
  weighted_error_sum+=v*d;
 }
 //дельта: delta_i = p_i * (error_i - weighted_error_sum)
 type_t *d_output=tensor_output.GetTensorDataPtr(w_out,z)+y*tensor_output.Size_X;
 for(uint32_t x=0;x<size_x;x++,d_xin++,d_delta++,d_output++)
 {
  type_t p=*d_xin;
  type_t d=*d_delta;
  type_t delta=p*(d-weighted_error_sum);
  *d_output=delta;
 }
}

//----------------------------------------------------------------------------------------------------
//выполнить обратный проход SoftMax по слою X
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorSoftMax<type_t>::SoftMaxX_Backward(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Delta,CTensor<type_t> &cTensor_Input)
{
 cTensor_Input.CopyToDevice();
 cTensor_Delta.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Delta(cTensor_Delta);

 //запускаем процесс
 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input.Size_W,cTensor_Input.Size_Z,cTensor_Input.Size_Y);

 CUDATensorSoftMaxXBackwardFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Delta,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

#endif

#endif

