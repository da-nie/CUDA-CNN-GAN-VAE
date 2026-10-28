#ifndef C_TENSOR_NORMALIZATION_H
#define C_TENSOR_NORMALIZATION_H

//****************************************************************************************************
//Операции нормализации над тензорами произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "ctensor.h"
#include "ctensormath.h"
#include <vector>

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

 const type_t LAYER_NORM_EPS=1e-5f;

 for(int32_t w=0;w<cTensor_Input.Size_W;w++)
 {
  uint32_t w_in=Mod(w,cTensor_Input.Size_W);
  uint32_t w_out=Mod(w,cTensor_Output.Size_W);
  for(int32_t z=0;z<cTensor_Input.Size_Z;z++)
  {
   for(int32_t y=0;y<cTensor_Input.Size_Y;y++)
   {
    //выполняем нормализацию слоя
    type_t *d_xin=cTensor_Input.GetTensorDataPtr(w_in,z)+y*cTensor_Input.Size_X;
    type_t *d_xout=cTensor_Output.GetTensorDataPtr(w_out,z)+y*cTensor_Input.Size_X;

    type_t *d_xin_local;
    type_t *d_xout_local;
    //считаем среднее по X
    type_t mean=0;
    d_xin_local=d_xin;
    for(uint32_t x=0;x<cTensor_Input.Size_X;x++,d_xin_local++) mean+=*d_xin_local;
    mean/=static_cast<type_t>(cTensor_Input.Size_X);
    //считаем дисперсию
    type_t var=0;
    d_xin_local=d_xin;
    d_xout_local=d_xout;
    for(uint32_t x=0;x<cTensor_Input.Size_X;x++,d_xin_local++,d_xout_local++)
    {
     type_t v=*d_xin_local;
     v-=mean;
     *d_xout_local=v;
     var+=v*v;
    }
    var/=static_cast<type_t>(cTensor_Input.Size_X);
    //обратная дисперсия
    type_t inv_std=1.0/sqrtf(var+LAYER_NORM_EPS);
    //нормируем
    d_xin_local=d_xin;
    d_xout_local=d_xout;
    type_t *d_gamma=cTensor_dGamma.GetTensorDataPtr(w_in,z);
    type_t *d_beta=cTensor_dBeta.GetTensorDataPtr(w_out,z);
    for(uint32_t x=0;x<cTensor_Input.Size_X;x++,d_xin_local++,d_xout_local++,d_gamma++,d_beta++)
    {
     type_t v=(*d_xout_local);
     v*=inv_std;
     type_t dgamma=*d_gamma;
     type_t dbeta=*d_beta;
     v=v*dgamma+dbeta;
     *d_xout_local=v;
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//!выполнить прямой проход GroupNorm
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorNormalization<type_t>::GroupNormForward(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_Gamma,CTensor<type_t> &cTensor_Beta,CTensor<type_t> &cTensor_XHAT,CTensor<type_t> &cTensor_InvStd,uint32_t num_groups,uint32_t channels_per_group,type_t epsilon)
{
 for(uint32_t w=0;w<cTensor_Input.Size_W;w++)
 {
  for(uint32_t g=0;g<num_groups;g++)
  {
   uint32_t Z=cTensor_Input.Size_Z;
   uint32_t Y=cTensor_Input.Size_Y;
   uint32_t X=cTensor_Input.Size_X;
   type_t M=static_cast<type_t>(channels_per_group)*static_cast<type_t>(Y)*static_cast<type_t>(X);
   //считаем среднее
   type_t sum=0;
   for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
   {
    type_t *ptr=cTensor_Input.GetColumnPtr(w,c,0);
    for(uint32_t i=0;i<Y*X;i++) sum+=ptr[i];
   }
   type_t mean=sum/M;
   //считаем дисперсию
   type_t sq_sum=0;
   for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
   {
    type_t *ptr=cTensor_Input.GetColumnPtr(w,c,0);
    for(uint32_t i=0;i<Y*X;i++)
    {
     type_t diff=ptr[i]-mean;
     sq_sum+=diff*diff;
    }
   }
   type_t var=sq_sum/M;
   if (var<0) var=0;
   type_t inv_std=1.0/sqrt(var+epsilon);
   cTensor_InvStd.SetElement(w,g,0,0,inv_std);
   //нормализуем и масштабируем
   for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
   {
    type_t gamma=cTensor_Gamma.GetElement(0,c,0,0);
    type_t beta=cTensor_Beta.GetElement(0,c,0,0);
    type_t *in_ptr=cTensor_Input.GetColumnPtr(w,c,0);
    type_t *out_ptr=cTensor_Output.GetColumnPtr(w,c,0);
    type_t *xhat_ptr=cTensor_XHAT.GetColumnPtr(w,c,0);
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
 }
}

//----------------------------------------------------------------------------------------------------
//!выполнить обратный проход GroupNorm
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorNormalization<type_t>::GroupNormBackward(CTensor<type_t> &cTensor_Delta_Array,CTensor<type_t> &cTensor_XHAT_Array,CTensor<type_t> &cTensor_Gamma,CTensor<type_t> &cTensor_InvStd_Array,CTensor<type_t> &cTensor_PrevLayerError_Array,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta,uint32_t num_groups,uint32_t channels_per_group,bool calc_weights)
{
  for(uint32_t w=0;w<cTensor_Delta_Array.Size_W;w++)
 {
  for(uint32_t g=0;g<num_groups;g++)
  {
   uint32_t W=cTensor_Delta_Array.Size_W;
   uint32_t Y=cTensor_Delta_Array.Size_Y;
   uint32_t X=cTensor_Delta_Array.Size_X;
   type_t M=static_cast<type_t>(channels_per_group)*static_cast<type_t>(Y)*static_cast<type_t>(X);
   //масштабирующий коэффициент для усреднения (1/BatchSize*Y*X)
   type_t scale=1.0;// 1.0/(static_cast<type_t>(W)*static_cast<type_t>(Y)*static_cast<type_t>(X));
   type_t inv_std=cTensor_InvStd_Array.GetElement(w,g,0,0);
   //первый проход: суммируем градиенты по группе
   type_t sum_dxhat=0;
   type_t sum_dxhat_xhat=0;
   for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
   {
    type_t gamma=cTensor_Gamma.GetElement(0,c,0,0);
    type_t *dy_ptr=cTensor_Delta_Array.GetColumnPtr(w,c,0);
    type_t *xhat_ptr=cTensor_XHAT_Array.GetColumnPtr(w,c,0);
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
      if (isfinite(dg)) cTensor_dGamma.SetElement(0,c,0,0,cTensor_dGamma.GetElement(0,c,0,0)+dg);
      if (isfinite(db)) cTensor_dBeta.SetElement(0,c,0,0,cTensor_dBeta.GetElement(0,c,0,0)+db);
     }
    }
   }
   //второй проход: вычисляем градиент для предыдущего слоя
   for(uint32_t c=g*channels_per_group;c<(g+1)*channels_per_group;c++)
   {
    type_t gamma=cTensor_Gamma.GetElement(0,c,0,0);
    type_t *dy_ptr=cTensor_Delta_Array.GetColumnPtr(w,c,0);
    type_t *xhat_ptr=cTensor_XHAT_Array.GetColumnPtr(w,c,0);
    type_t *dx_ptr=cTensor_PrevLayerError_Array.GetColumnPtr(w,c,0);
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
 }
}

#endif
