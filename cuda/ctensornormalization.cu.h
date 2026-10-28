#ifndef C_TENSOR_NORMALIZATION_CU_H
#define C_TENSOR_NORMALIZATION_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensonormalization.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции нормализации над тензорами произвольной размерности
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
///!Операции нормализации над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorNormalization
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
  static const uint32_t LAYER_NORM_BLOCK_SIZE=256;///<размер CUDA-блока для функций LayerNorm
 private:
  //-переменные-----------------------------------------------------------------------------------------
 public:
  //-конструктор----------------------------------------------------------------------------------------
  //-конструктор копирования----------------------------------------------------------------------------
  //-деструктор-----------------------------------------------------------------------------------------
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  static void LayerNormalizeX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta);///<выполнить нормализацию по слою X
  static void GroupNormForward(CTensor<type_t> &cTensor_Output, CTensor<type_t> &cTensor_Input, CTensor<type_t> &cTensor_Gamma, CTensor<type_t> &cTensor_Beta, CTensor<type_t> &cTensor_XHAT, CTensor<type_t> &cTensor_InvStd, uint32_t num_groups, uint32_t channels_per_group,type_t epsilon);///<прямой проход GroupNorm
  static void GroupNormBackward(CTensor<type_t> &cTensor_Delta_Array, CTensor<type_t> &cTensor_XHAT_Array, CTensor<type_t> &cTensor_Gamma, CTensor<type_t> &cTensor_InvStd_Array, CTensor<type_t> &cTensor_PrevLayerError_Array, CTensor<type_t> &cTensor_dGamma, CTensor<type_t> &cTensor_dBeta, uint32_t num_groups, uint32_t channels_per_group, bool calc_weights);///<обратный проход GroupNorm

  static void LayerNormForward(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,const CTensor<type_t> &cTensor_Gamma,const CTensor<type_t> &cTensor_Beta,CTensor<type_t> &cTensor_XHAT,CTensor<type_t> &cTensor_InvStd,type_t epsilon);///<прямой проход LayerNorm (статистики по Z*Y*X для каждого элемента пакета)
  static void LayerNormBackward(const CTensor<type_t> &cTensor_Delta,const CTensor<type_t> &cTensor_XHAT,const CTensor<type_t> &cTensor_Gamma,const CTensor<type_t> &cTensor_InvStd,CTensor<type_t> &cTensor_PrevLayerError,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta);///<обратный проход LayerNorm (градиенты гаммы и беты НАКАПЛИВАЮТСЯ)

  static void LayerNormForwardGPT2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,const CTensor<type_t> &cTensor_Gamma,const CTensor<type_t> &cTensor_Beta,CTensor<type_t> &cTensor_XHAT,CTensor<type_t> &cTensor_InvStd,type_t epsilon);///<прямой проход LayerNorm: статистики по Z для каждой позиции (w,y,x)
  static void LayerNormBackwardGPT2(const CTensor<type_t> &cTensor_Delta,const CTensor<type_t> &cTensor_XHAT,const CTensor<type_t> &cTensor_Gamma,const CTensor<type_t> &cTensor_InvStd,CTensor<type_t> &cTensor_PrevLayerError,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta);///<обратный проход LayerNorm (dGamma и dBeta НАКАПЛИВАЮТСЯ)
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
void CTensorNormalization<type_t>::LayerNormalizeX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta)
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
void CTensorNormalization<type_t>::GroupNormForward(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_Gamma,CTensor<type_t> &cTensor_Beta,CTensor<type_t> &cTensor_XHAT,CTensor<type_t> &cTensor_InvStd,uint32_t num_groups,uint32_t channels_per_group,type_t epsilon)
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
void CTensorNormalization<type_t>::GroupNormBackward(CTensor<type_t> &cTensor_Delta_Array,CTensor<type_t> &cTensor_XHAT_Array,CTensor<type_t> &cTensor_Gamma,CTensor<type_t> &cTensor_InvStd_Array,CTensor<type_t> &cTensor_PrevLayerError_Array,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta,uint32_t num_groups,uint32_t channels_per_group,bool calc_weights)
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


//****************************************************************************************************
//функция CUDA для прямого прохода LayerNorm
//----------------------------------------------------------------------------------------------------
//Для каждого элемента пакета w статистики считаются по всем Z*Y*X элементам.
//Один CUDA-блок обрабатывает один элемент пакета (blockDim.x==CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE).
//****************************************************************************************************
template<class type_t>
__global__ void CUDALayerNormForwardFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<type_t> tensor_gamma,STensorKernel<type_t> tensor_beta,STensorKernel<type_t> tensor_xhat,STensorKernel<type_t> tensor_invstd,type_t epsilon)
{
 uint32_t w=blockIdx.x;///<номер элемента пакета
 uint32_t size_z=tensor_input.Size_Z;
 uint32_t size_y=tensor_input.Size_Y;
 uint32_t size_x=tensor_input.Size_X;
 uint64_t amount_xy=static_cast<uint64_t>(size_y)*size_x;
 uint64_t amount=static_cast<uint64_t>(size_z)*amount_xy;///<размер нормализации Z*Y*X

 __shared__ type_t s_sum[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];///<для редукции суммы
 __shared__ type_t s_sum2[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];///<для редукции суммы квадратов
 __shared__ type_t s_mean;
 __shared__ type_t s_invstd;

 uint32_t tid=threadIdx.x;
 uint32_t step=blockDim.x;

 //первый проход: сумма и сумма квадратов
 type_t sum=0;
 type_t sum2=0;
 for(uint64_t k=tid;k<amount;k+=step)
 {
  uint32_t x=static_cast<uint32_t>(k%size_x);
  uint32_t y=static_cast<uint32_t>((k/size_x)%size_y);
  uint32_t z=static_cast<uint32_t>(k/amount_xy);
  type_t value=tensor_input.GetElement(w,z,y,x);
  sum+=value;
  sum2+=value*value;
 }
 s_sum[tid]=sum;
 s_sum2[tid]=sum2;
 __syncthreads();

 for(uint32_t s=blockDim.x/2;s>0;s>>=1)
 {
  if (tid<s)
  {
   s_sum[tid]+=s_sum[tid+s];
   s_sum2[tid]+=s_sum2[tid+s];
  }
  __syncthreads();
 }

 if (tid==0)
 {
  type_t mean=s_sum[0]/static_cast<type_t>(amount);
  type_t variance=s_sum2[0]/static_cast<type_t>(amount)-mean*mean;
  if (variance<0) variance=0;//защита от отрицательной дисперсии из-за ошибок округления
  type_t invstd=static_cast<type_t>(1)/sqrt(variance+epsilon);
  s_mean=mean;
  s_invstd=invstd;
  tensor_invstd.SetElement(w,0,0,0,invstd);
 }
 __syncthreads();

 type_t mean=s_mean;
 type_t invstd=s_invstd;

 //второй проход: нормализация и применение Gamma и Beta
 for(uint64_t k=tid;k<amount;k+=step)
 {
  uint32_t x=static_cast<uint32_t>(k%size_x);
  uint32_t y=static_cast<uint32_t>((k/size_x)%size_y);
  uint32_t z=static_cast<uint32_t>(k/amount_xy);
  type_t value=tensor_input.GetElement(w,z,y,x);
  type_t xhat=(value-mean)*invstd;
  tensor_xhat.SetElement(w,z,y,x,xhat);
  type_t gamma=tensor_gamma.GetElement(0,z,0,0);
  type_t beta=tensor_beta.GetElement(0,z,0,0);
  tensor_output.SetElement(w,z,y,x,gamma*xhat+beta);
 }
}

//****************************************************************************************************
//функция CUDA для обратного прохода LayerNorm (ошибка для предыдущего слоя)
//----------------------------------------------------------------------------------------------------
//dInput = invstd*( dOut*Gamma - mean(dOut*Gamma) - XHAT*mean(dOut*Gamma*XHAT) )
//****************************************************************************************************
template<class type_t>
__global__ void CUDALayerNormBackwardDataFunction(STensorKernel<type_t> tensor_prev_error,STensorKernel<type_t> tensor_delta,STensorKernel<type_t> tensor_xhat,STensorKernel<type_t> tensor_gamma,STensorKernel<type_t> tensor_invstd)
{
 uint32_t w=blockIdx.x;
 uint32_t size_z=tensor_delta.Size_Z;
 uint32_t size_y=tensor_delta.Size_Y;
 uint32_t size_x=tensor_delta.Size_X;
 uint64_t amount_xy=static_cast<uint64_t>(size_y)*size_x;
 uint64_t amount=static_cast<uint64_t>(size_z)*amount_xy;

 __shared__ type_t s_sum[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];
 __shared__ type_t s_sumx[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];
 __shared__ type_t s_sum_dxhat;
 __shared__ type_t s_sum_dxhat_xhat;

 uint32_t tid=threadIdx.x;
 uint32_t step=blockDim.x;

 type_t sum=0;
 type_t sumx=0;
 for(uint64_t k=tid;k<amount;k+=step)
 {
  uint32_t x=static_cast<uint32_t>(k%size_x);
  uint32_t y=static_cast<uint32_t>((k/size_x)%size_y);
  uint32_t z=static_cast<uint32_t>(k/amount_xy);
  type_t dxhat=tensor_delta.GetElement(w,z,y,x)*tensor_gamma.GetElement(0,z,0,0);
  sum+=dxhat;
  sumx+=dxhat*tensor_xhat.GetElement(w,z,y,x);
 }
 s_sum[tid]=sum;
 s_sumx[tid]=sumx;
 __syncthreads();

 for(uint32_t s=blockDim.x/2;s>0;s>>=1)
 {
  if (tid<s)
  {
   s_sum[tid]+=s_sum[tid+s];
   s_sumx[tid]+=s_sumx[tid+s];
  }
  __syncthreads();
 }

 if (tid==0)
 {
  s_sum_dxhat=s_sum[0];
  s_sum_dxhat_xhat=s_sumx[0];
 }
 __syncthreads();

 type_t scale=tensor_invstd.GetElement(w,0,0,0)/static_cast<type_t>(amount);
 type_t sum_dxhat=s_sum_dxhat;
 type_t sum_dxhat_xhat=s_sum_dxhat_xhat;

 for(uint64_t k=tid;k<amount;k+=step)
 {
  uint32_t x=static_cast<uint32_t>(k%size_x);
  uint32_t y=static_cast<uint32_t>((k/size_x)%size_y);
  uint32_t z=static_cast<uint32_t>(k/amount_xy);
  type_t dxhat=tensor_delta.GetElement(w,z,y,x)*tensor_gamma.GetElement(0,z,0,0);
  type_t xhat=tensor_xhat.GetElement(w,z,y,x);
  tensor_prev_error.SetElement(w,z,y,x,scale*(static_cast<type_t>(amount)*dxhat-sum_dxhat-xhat*sum_dxhat_xhat));
 }
}

//****************************************************************************************************
//функция CUDA для обратного прохода LayerNorm (градиенты Gamma и Beta)
//----------------------------------------------------------------------------------------------------
//dGamma(z) += SUM(dOut*XHAT); dBeta(z) += SUM(dOut) — суммы по всем w,y,x.
//Один CUDA-блок обрабатывает один канал z. Градиенты НАКАПЛИВАЮТСЯ.
//****************************************************************************************************
template<class type_t>
__global__ void CUDALayerNormBackwardParamFunction(STensorKernel<type_t> tensor_dgamma,STensorKernel<type_t> tensor_dbeta,STensorKernel<type_t> tensor_delta,STensorKernel<type_t> tensor_xhat)
{
 uint32_t z=blockIdx.x;///<номер канала
 uint32_t size_w=tensor_delta.Size_W;
 uint32_t size_y=tensor_delta.Size_Y;
 uint32_t size_x=tensor_delta.Size_X;
 uint64_t amount_xy=static_cast<uint64_t>(size_y)*size_x;
 uint64_t amount=static_cast<uint64_t>(size_w)*amount_xy;

 __shared__ type_t s_sum[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];
 __shared__ type_t s_sumx[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];

 uint32_t tid=threadIdx.x;
 uint32_t step=blockDim.x;

 type_t sum=0;///<сумма dOut (для dBeta)
 type_t sumx=0;///<сумма dOut*XHAT (для dGamma)
 for(uint64_t k=tid;k<amount;k+=step)
 {
  uint32_t x=static_cast<uint32_t>(k%size_x);
  uint32_t y=static_cast<uint32_t>((k/size_x)%size_y);
  uint32_t w=static_cast<uint32_t>(k/amount_xy);
  type_t dout=tensor_delta.GetElement(w,z,y,x);
  sum+=dout;
  sumx+=dout*tensor_xhat.GetElement(w,z,y,x);
 }
 s_sum[tid]=sum;
 s_sumx[tid]=sumx;
 __syncthreads();

 for(uint32_t s=blockDim.x/2;s>0;s>>=1)
 {
  if (tid<s)
  {
   s_sum[tid]+=s_sum[tid+s];
   s_sumx[tid]+=s_sumx[tid+s];
  }
  __syncthreads();
 }

 if (tid==0)
 {
  tensor_dbeta.SetElement(0,z,0,0,tensor_dbeta.GetElement(0,z,0,0)+s_sum[0]);
  tensor_dgamma.SetElement(0,z,0,0,tensor_dgamma.GetElement(0,z,0,0)+s_sumx[0]);
 }
}

//----------------------------------------------------------------------------------------------------
//прямой проход LayerNorm
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorNormalization<type_t>::LayerNormForward(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,const CTensor<type_t> &cTensor_Gamma,const CTensor<type_t> &cTensor_Beta,CTensor<type_t> &cTensor_XHAT,CTensor<type_t> &cTensor_InvStd,type_t epsilon)
{
 if (cTensor_Output.Size_W!=cTensor_Input.Size_W || cTensor_Output.Size_Z!=cTensor_Input.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Input.Size_Y || cTensor_Output.Size_X!=cTensor_Input.Size_X ||
     cTensor_XHAT.Size_W!=cTensor_Input.Size_W || cTensor_XHAT.Size_Z!=cTensor_Input.Size_Z ||
     cTensor_XHAT.Size_Y!=cTensor_Input.Size_Y || cTensor_XHAT.Size_X!=cTensor_Input.Size_X)
 {
  throw "CTensorMath::LayerNormForward: Размерности входа, выхода и XHAT не совпадают!";
 }
 if (cTensor_Gamma.Size_W!=1 || cTensor_Gamma.Size_Z!=cTensor_Input.Size_Z || cTensor_Gamma.Size_Y!=1 || cTensor_Gamma.Size_X!=1 ||
     cTensor_Beta.Size_W!=1 || cTensor_Beta.Size_Z!=cTensor_Input.Size_Z || cTensor_Beta.Size_Y!=1 || cTensor_Beta.Size_X!=1)
 {
  throw "CTensorMath::LayerNormForward: Тензоры Gamma и Beta должны иметь размерность 1xZx1x1!";
 }
 if (cTensor_InvStd.Size_W!=cTensor_Input.Size_W || cTensor_InvStd.Size_Z!=1 || cTensor_InvStd.Size_Y!=1 || cTensor_InvStd.Size_X!=1)
 {
  throw "CTensorMath::LayerNormForward: Тензор InvStd должен иметь размерность BatchSize x 1 x 1 x 1!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_Gamma.CopyToDevice();
 cTensor_Beta.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Gamma(cTensor_Gamma);
 STensorKernel<type_t> sTensorKernel_Beta(cTensor_Beta);
 STensorKernel<type_t> sTensorKernel_XHAT(cTensor_XHAT);
 STensorKernel<type_t> sTensorKernel_InvStd(cTensor_InvStd);

 dim3 thread(CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE,1,1);
 dim3 blocks(cTensor_Input.Size_W,1,1);
 if (blocks.x==0) blocks.x=1;

 CUDALayerNormForwardFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,sTensorKernel_Gamma,sTensorKernel_Beta,sTensorKernel_XHAT,sTensorKernel_InvStd,epsilon);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
 cTensor_XHAT.SetDeviceOnChange();
 cTensor_InvStd.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//обратный проход LayerNorm
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorNormalization<type_t>::LayerNormBackward(const CTensor<type_t> &cTensor_Delta,const CTensor<type_t> &cTensor_XHAT,const CTensor<type_t> &cTensor_Gamma,const CTensor<type_t> &cTensor_InvStd,CTensor<type_t> &cTensor_PrevLayerError,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta)
{
 if (cTensor_PrevLayerError.Size_W!=cTensor_Delta.Size_W || cTensor_PrevLayerError.Size_Z!=cTensor_Delta.Size_Z ||
     cTensor_PrevLayerError.Size_Y!=cTensor_Delta.Size_Y || cTensor_PrevLayerError.Size_X!=cTensor_Delta.Size_X ||
     cTensor_XHAT.Size_W!=cTensor_Delta.Size_W || cTensor_XHAT.Size_Z!=cTensor_Delta.Size_Z ||
     cTensor_XHAT.Size_Y!=cTensor_Delta.Size_Y || cTensor_XHAT.Size_X!=cTensor_Delta.Size_X)
 {
  throw "CTensorMath::LayerNormBackward: Размерности дельты, XHAT и ошибки предыдущего слоя не совпадают!";
 }
 if (cTensor_Gamma.Size_W!=1 || cTensor_Gamma.Size_Z!=cTensor_Delta.Size_Z || cTensor_Gamma.Size_Y!=1 || cTensor_Gamma.Size_X!=1 ||
     cTensor_dGamma.Size_W!=1 || cTensor_dGamma.Size_Z!=cTensor_Delta.Size_Z || cTensor_dGamma.Size_Y!=1 || cTensor_dGamma.Size_X!=1 ||
     cTensor_dBeta.Size_W!=1 || cTensor_dBeta.Size_Z!=cTensor_Delta.Size_Z || cTensor_dBeta.Size_Y!=1 || cTensor_dBeta.Size_X!=1)
 {
  throw "CTensorMath::LayerNormBackward: Тензоры Gamma, dGamma и dBeta должны иметь размерность 1xZx1x1!";
 }
 if (cTensor_InvStd.Size_W!=cTensor_Delta.Size_W || cTensor_InvStd.Size_Z!=1 || cTensor_InvStd.Size_Y!=1 || cTensor_InvStd.Size_X!=1)
 {
  throw "CTensorMath::LayerNormBackward: Тензор InvStd должен иметь размерность BatchSize x 1 x 1 x 1!";
 }

 cTensor_Delta.CopyToDevice();
 cTensor_XHAT.CopyToDevice();
 cTensor_Gamma.CopyToDevice();
 cTensor_InvStd.CopyToDevice();
 cTensor_dGamma.CopyToDevice();//важно: градиенты накапливаются, текущие значения должны быть на устройстве
 cTensor_dBeta.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_PrevLayerError(cTensor_PrevLayerError);
 STensorKernel<type_t> sTensorKernel_Delta(cTensor_Delta);
 STensorKernel<type_t> sTensorKernel_XHAT(cTensor_XHAT);
 STensorKernel<type_t> sTensorKernel_Gamma(cTensor_Gamma);
 STensorKernel<type_t> sTensorKernel_InvStd(cTensor_InvStd);
 STensorKernel<type_t> sTensorKernel_dGamma(cTensor_dGamma);
 STensorKernel<type_t> sTensorKernel_dBeta(cTensor_dBeta);

 dim3 thread(CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE,1,1);

 //ошибка для предыдущего слоя: один CUDA-блок на элемент пакета
 dim3 blocks_data(cTensor_Delta.Size_W,1,1);
 if (blocks_data.x==0) blocks_data.x=1;
 CUDALayerNormBackwardDataFunction<type_t><<<blocks_data,thread>>>(sTensorKernel_PrevLayerError,sTensorKernel_Delta,sTensorKernel_XHAT,sTensorKernel_Gamma,sTensorKernel_InvStd);

 //градиенты параметров: один CUDA-блок на канал
 dim3 blocks_param(cTensor_Delta.Size_Z,1,1);
 if (blocks_param.x==0) blocks_param.x=1;
 CUDALayerNormBackwardParamFunction<type_t><<<blocks_param,thread>>>(sTensorKernel_dGamma,sTensorKernel_dBeta,sTensorKernel_Delta,sTensorKernel_XHAT);

 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_PrevLayerError.SetDeviceOnChange();
 cTensor_dGamma.SetDeviceOnChange();
 cTensor_dBeta.SetDeviceOnChange();
}


//****************************************************************************************************
//Функция CUDA прямого прохода LayerNorm (вариант GPT-2)
//----------------------------------------------------------------------------------------------------
//Статистики вычисляются по всем каналам Z независимо для каждой позиции (w,y,x).
//Для GPT-2: W-пакет, Z-эмбеддинг, Y-токены, X=1.
//Один CUDA-блок обрабатывает одну позицию (w,y,x).
//****************************************************************************************************
template<class type_t>
__global__ void CUDALayerNormForwardFunctionGTP2(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<type_t> tensor_gamma,STensorKernel<type_t> tensor_beta,STensorKernel<type_t> tensor_xhat,STensorKernel<type_t> tensor_invstd,type_t epsilon)
{
 uint32_t x=blockIdx.x;
 uint32_t y=blockIdx.y;
 uint32_t w=blockIdx.z;
 uint32_t size_z=tensor_input.Size_Z;

 __shared__ type_t s_sum[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];
 __shared__ type_t s_sum2[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];
 __shared__ type_t s_mean;
 __shared__ type_t s_invstd;

 uint32_t tid=threadIdx.x;
 uint32_t step=blockDim.x;

 //первый проход: сумма и сумма квадратов по Z
 type_t sum=0;
 type_t sum2=0;
 for(uint32_t z=tid;z<size_z;z+=step)
 {
  type_t v=static_cast<type_t>(tensor_input.GetElement(w,z,y,x));
  sum+=v;
  sum2+=v*v;
 }
 s_sum[tid]=sum;
 s_sum2[tid]=sum2;
 __syncthreads();

 for(uint32_t s=blockDim.x/2;s>0;s>>=1)
 {
  if (tid<s)
  {
   s_sum[tid]+=s_sum[tid+s];
   s_sum2[tid]+=s_sum2[tid+s];
  }
  __syncthreads();
 }

 if (tid==0)
 {
  type_t mean=s_sum[0]/static_cast<type_t>(size_z);
  type_t variance=s_sum2[0]/static_cast<type_t>(size_z)-mean*mean;
  if (variance<0) variance=0;//защита от отрицательной дисперсии из-за ошибок округления
  type_t invstd=1.0/sqrt(variance+static_cast<type_t>(epsilon));
  s_mean=mean;
  s_invstd=invstd;
  tensor_invstd.SetElement(w,0,y,x,static_cast<type_t>(invstd));
 }
 __syncthreads();

 type_t mean=s_mean;
 type_t invstd=s_invstd;

 //второй проход: нормализация, применение Gamma и Beta
 for(uint32_t z=tid;z<size_z;z+=step)
 {
  type_t v=static_cast<type_t>(tensor_input.GetElement(w,z,y,x));
  type_t xhat=static_cast<type_t>((v-mean)*invstd);
  tensor_xhat.SetElement(w,z,y,x,xhat);
  type_t gamma=tensor_gamma.GetElement(0,z,0,0);
  type_t beta=tensor_beta.GetElement(0,z,0,0);
  tensor_output.SetElement(w,z,y,x,gamma*xhat+beta);
 }
}

//----------------------------------------------------------------------------------------------------
//прямой проход LayerNorm (вариант GPT-2)
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorNormalization<type_t>::LayerNormForwardGPT2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,const CTensor<type_t> &cTensor_Gamma,const CTensor<type_t> &cTensor_Beta,CTensor<type_t> &cTensor_XHAT,CTensor<type_t> &cTensor_InvStd,type_t epsilon)
{
 if (cTensor_Output.Size_W!=cTensor_Input.Size_W || cTensor_Output.Size_Z!=cTensor_Input.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Input.Size_Y || cTensor_Output.Size_X!=cTensor_Input.Size_X ||
     cTensor_XHAT.Size_W!=cTensor_Input.Size_W || cTensor_XHAT.Size_Z!=cTensor_Input.Size_Z ||
     cTensor_XHAT.Size_Y!=cTensor_Input.Size_Y || cTensor_XHAT.Size_X!=cTensor_Input.Size_X)
 {
  throw "CTensorMath::LayerNormForward: Размерности входа, выхода и XHAT не совпадают!";
 }
 if (cTensor_Gamma.Size_W!=1 || cTensor_Gamma.Size_Z!=cTensor_Input.Size_Z || cTensor_Gamma.Size_Y!=1 || cTensor_Gamma.Size_X!=1 ||
     cTensor_Beta.Size_W!=1 || cTensor_Beta.Size_Z!=cTensor_Input.Size_Z || cTensor_Beta.Size_Y!=1 || cTensor_Beta.Size_X!=1)
 {
  throw "CTensorMath::LayerNormForward: Тензоры Gamma и Beta должны иметь размерность 1xZx1x1!";
 }
 if (cTensor_InvStd.Size_W!=cTensor_Input.Size_W || cTensor_InvStd.Size_Z!=1 ||
     cTensor_InvStd.Size_Y!=cTensor_Input.Size_Y || cTensor_InvStd.Size_X!=cTensor_Input.Size_X)
 {
  throw "CTensorMath::LayerNormForward: Тензор InvStd должен иметь размерность BatchSize x 1 x Y x X!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_Gamma.CopyToDevice();
 cTensor_Beta.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Gamma(cTensor_Gamma);
 STensorKernel<type_t> sTensorKernel_Beta(cTensor_Beta);
 STensorKernel<type_t> sTensorKernel_XHAT(cTensor_XHAT);
 STensorKernel<type_t> sTensorKernel_InvStd(cTensor_InvStd);

 dim3 thread(CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE,1,1);
 dim3 blocks(cTensor_Input.Size_X,cTensor_Input.Size_Y,cTensor_Input.Size_W);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 CUDALayerNormForwardFunctionGTP2<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,sTensorKernel_Gamma,sTensorKernel_Beta,sTensorKernel_XHAT,sTensorKernel_InvStd,epsilon);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
 cTensor_XHAT.SetDeviceOnChange();
 cTensor_InvStd.SetDeviceOnChange();
}


//****************************************************************************************************
//Функция CUDA обратного прохода LayerNorm: ошибка для предыдущего слоя
//----------------------------------------------------------------------------------------------------
//dInput = invstd*( dOut*Gamma - mean(dOut*Gamma) - XHAT*mean(dOut*Gamma*XHAT) )
//Средние берутся по Z для каждой позиции (w,y,x).
//****************************************************************************************************
template<class type_t>
__global__ void CUDALayerNormBackwardDataFunctionGTP2(STensorKernel<type_t> tensor_prev_error,STensorKernel<type_t> tensor_delta,STensorKernel<type_t> tensor_xhat,STensorKernel<type_t> tensor_gamma,STensorKernel<type_t> tensor_invstd)
{
 uint32_t x=blockIdx.x;
 uint32_t y=blockIdx.y;
 uint32_t w=blockIdx.z;
 uint32_t size_z=tensor_delta.Size_Z;

 __shared__ type_t s_sum[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];
 __shared__ type_t s_sumx[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];
 __shared__ type_t s_mean_dxhat;
 __shared__ type_t s_mean_dxhat_xhat;

 uint32_t tid=threadIdx.x;
 uint32_t step=blockDim.x;

 type_t sum=0;
 type_t sumx=0;
 for(uint32_t z=tid;z<size_z;z+=step)
 {
  type_t dxhat=static_cast<type_t>(tensor_delta.GetElement(w,z,y,x))*static_cast<type_t>(tensor_gamma.GetElement(0,z,0,0));
  sum+=dxhat;
  sumx+=dxhat*static_cast<type_t>(tensor_xhat.GetElement(w,z,y,x));
 }
 s_sum[tid]=sum;
 s_sumx[tid]=sumx;
 __syncthreads();

 for(uint32_t s=blockDim.x/2;s>0;s>>=1)
 {
  if (tid<s)
  {
   s_sum[tid]+=s_sum[tid+s];
   s_sumx[tid]+=s_sumx[tid+s];
  }
  __syncthreads();
 }

 if (tid==0)
 {
  s_mean_dxhat=s_sum[0]/static_cast<type_t>(size_z);
  s_mean_dxhat_xhat=s_sumx[0]/static_cast<type_t>(size_z);
 }
 __syncthreads();

 type_t invstd=static_cast<type_t>(tensor_invstd.GetElement(w,0,y,x));
 type_t mean_dxhat=s_mean_dxhat;
 type_t mean_dxhat_xhat=s_mean_dxhat_xhat;

 for(uint32_t z=tid;z<size_z;z+=step)
 {
  type_t dxhat=static_cast<type_t>(tensor_delta.GetElement(w,z,y,x))*static_cast<type_t>(tensor_gamma.GetElement(0,z,0,0));
  type_t xhat=static_cast<type_t>(tensor_xhat.GetElement(w,z,y,x));
  tensor_prev_error.SetElement(w,z,y,x,static_cast<type_t>(invstd*(dxhat-mean_dxhat-xhat*mean_dxhat_xhat)));
 }
}

//****************************************************************************************************
//Функция CUDA обратного прохода LayerNorm: градиенты Gamma и Beta
//----------------------------------------------------------------------------------------------------
//dGamma(z) += SUM(dOut*XHAT); dBeta(z) += SUM(dOut) — суммы по всем w,y,x.
//Один CUDA-блок обрабатывает один канал z. Градиенты НАКАПЛИВАЮТСЯ.
//****************************************************************************************************
template<class type_t>
__global__ void CUDALayerNormBackwardParamFunctionGTP2(STensorKernel<type_t> tensor_dgamma,STensorKernel<type_t> tensor_dbeta,STensorKernel<type_t> tensor_delta,STensorKernel<type_t> tensor_xhat)
{
 uint32_t z=blockIdx.x;
 uint32_t size_w=tensor_delta.Size_W;
 uint32_t size_y=tensor_delta.Size_Y;
 uint32_t size_x=tensor_delta.Size_X;
 uint64_t amount_xy=static_cast<uint64_t>(size_y)*size_x;
 uint64_t amount=static_cast<uint64_t>(size_w)*amount_xy;

 __shared__ type_t s_sum[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];
 __shared__ type_t s_sumx[CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE];

 uint32_t tid=threadIdx.x;
 uint32_t step=blockDim.x;

 type_t sum=0;///<сумма dOut (для dBeta)
 type_t sumx=0;///<сумма dOut*XHAT (для dGamma)
 for(uint64_t k=tid;k<amount;k+=step)
 {
  uint32_t x=static_cast<uint32_t>(k%size_x);
  uint32_t y=static_cast<uint32_t>((k/size_x)%size_y);
  uint32_t w=static_cast<uint32_t>(k/amount_xy);
  type_t dout=static_cast<type_t>(tensor_delta.GetElement(w,z,y,x));
  sum+=dout;
  sumx+=dout*static_cast<type_t>(tensor_xhat.GetElement(w,z,y,x));
 }
 s_sum[tid]=sum;
 s_sumx[tid]=sumx;
 __syncthreads();

 for(uint32_t s=blockDim.x/2;s>0;s>>=1)
 {
  if (tid<s)
  {
   s_sum[tid]+=s_sum[tid+s];
   s_sumx[tid]+=s_sumx[tid+s];
  }
  __syncthreads();
 }

 if (tid==0)
 {
  tensor_dbeta.SetElement(0,z,0,0,tensor_dbeta.GetElement(0,z,0,0)+static_cast<type_t>(s_sum[0]));
  tensor_dgamma.SetElement(0,z,0,0,tensor_dgamma.GetElement(0,z,0,0)+static_cast<type_t>(s_sumx[0]));
 }
}

//----------------------------------------------------------------------------------------------------
//обратный проход LayerNorm (вариант GPT-2)
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorNormalization<type_t>::LayerNormBackwardGPT2(const CTensor<type_t> &cTensor_Delta,const CTensor<type_t> &cTensor_XHAT,const CTensor<type_t> &cTensor_Gamma,const CTensor<type_t> &cTensor_InvStd,CTensor<type_t> &cTensor_PrevLayerError,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta)
{
 if (cTensor_PrevLayerError.Size_W!=cTensor_Delta.Size_W || cTensor_PrevLayerError.Size_Z!=cTensor_Delta.Size_Z ||
     cTensor_PrevLayerError.Size_Y!=cTensor_Delta.Size_Y || cTensor_PrevLayerError.Size_X!=cTensor_Delta.Size_X ||
     cTensor_XHAT.Size_W!=cTensor_Delta.Size_W || cTensor_XHAT.Size_Z!=cTensor_Delta.Size_Z ||
     cTensor_XHAT.Size_Y!=cTensor_Delta.Size_Y || cTensor_XHAT.Size_X!=cTensor_Delta.Size_X)
 {
  throw "CTensorMath::LayerNormBackward: Размерности дельты, XHAT и ошибки предыдущего слоя не совпадают!";
 }
 if (cTensor_Gamma.Size_W!=1 || cTensor_Gamma.Size_Z!=cTensor_Delta.Size_Z || cTensor_Gamma.Size_Y!=1 || cTensor_Gamma.Size_X!=1 ||
     cTensor_dGamma.Size_W!=1 || cTensor_dGamma.Size_Z!=cTensor_Delta.Size_Z || cTensor_dGamma.Size_Y!=1 || cTensor_dGamma.Size_X!=1 ||
     cTensor_dBeta.Size_W!=1 || cTensor_dBeta.Size_Z!=cTensor_Delta.Size_Z || cTensor_dBeta.Size_Y!=1 || cTensor_dBeta.Size_X!=1)
 {
  throw "CTensorMath::LayerNormBackward: Тензоры Gamma, dGamma и dBeta должны иметь размерность 1xZx1x1!";
 }
 if (cTensor_InvStd.Size_W!=cTensor_Delta.Size_W || cTensor_InvStd.Size_Z!=1 ||
     cTensor_InvStd.Size_Y!=cTensor_Delta.Size_Y || cTensor_InvStd.Size_X!=cTensor_Delta.Size_X)
 {
  throw "CTensorMath::LayerNormBackward: Тензор InvStd должен иметь размерность BatchSize x 1 x Y x X!";
 }

 cTensor_Delta.CopyToDevice();
 cTensor_XHAT.CopyToDevice();
 cTensor_Gamma.CopyToDevice();
 cTensor_InvStd.CopyToDevice();
 cTensor_dGamma.CopyToDevice();//градиенты накапливаются: текущие значения должны быть на устройстве
 cTensor_dBeta.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_PrevLayerError(cTensor_PrevLayerError);
 STensorKernel<type_t> sTensorKernel_Delta(cTensor_Delta);
 STensorKernel<type_t> sTensorKernel_XHAT(cTensor_XHAT);
 STensorKernel<type_t> sTensorKernel_Gamma(cTensor_Gamma);
 STensorKernel<type_t> sTensorKernel_InvStd(cTensor_InvStd);
 STensorKernel<type_t> sTensorKernel_dGamma(cTensor_dGamma);
 STensorKernel<type_t> sTensorKernel_dBeta(cTensor_dBeta);

 dim3 thread(CTensorNormalization<type_t>::LAYER_NORM_BLOCK_SIZE,1,1);

 //ошибка для предыдущего слоя: один блок на позицию (w,y,x)
 dim3 blocks_data(cTensor_Delta.Size_X,cTensor_Delta.Size_Y,cTensor_Delta.Size_W);
 if (blocks_data.x==0) blocks_data.x=1;
 if (blocks_data.y==0) blocks_data.y=1;
 if (blocks_data.z==0) blocks_data.z=1;
 CUDALayerNormBackwardDataFunctionGTP2<type_t><<<blocks_data,thread>>>(sTensorKernel_PrevLayerError,sTensorKernel_Delta,sTensorKernel_XHAT,sTensorKernel_Gamma,sTensorKernel_InvStd);
 HANDLE_ERROR(cudaGetLastError());

 //градиенты параметров: один блок на канал z
 dim3 blocks_param(cTensor_Delta.Size_Z,1,1);
 if (blocks_param.x==0) blocks_param.x=1;
 CUDALayerNormBackwardParamFunctionGTP2<type_t><<<blocks_param,thread>>>(sTensorKernel_dGamma,sTensorKernel_dBeta,sTensorKernel_Delta,sTensorKernel_XHAT);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_PrevLayerError.SetDeviceOnChange();
 cTensor_dGamma.SetDeviceOnChange();
 cTensor_dBeta.SetDeviceOnChange();
}


#endif

#endif
