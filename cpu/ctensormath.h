#ifndef C_TENSOR_MATH_H
#define C_TENSOR_MATH_H

//****************************************************************************************************
//Операции над тензорами произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "ctensor.h"
#include "../common/crandom.h"

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

struct SPos
{
 uint32_t X;
 uint32_t Y;
};

template<class type_t>
class CTensorMath
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
  static void ConcatecationZ(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_InputA,const CTensor<type_t> &cTensor_InputB);///<объединить два тензора по Z
  static void SplitZ(CTensor<type_t> &cTensor_OutputA,CTensor<type_t> &cTensor_OutputB,const CTensor<type_t> &cTensor_Input);///<разделить тензор на два по Z
  static void Fill(CTensor<type_t> &cTensor_Output,type_t value=0);///<записать в тензор число
  static void Inv(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<вычислить обратный тензор
  static void Div(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<поделить тензоры
  static void Add(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<сложить тензоры
  static void AddSumW(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<сложить тензоры по координате W
  static void Sub(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<вычесть тензоры
  static void AddBias(CTensor<type_t> &cTensor_Working,const CTensor<type_t> &cTensor_Bias);///<добавить смещения к элементам тензора (смещения одинаковы для x и y, но по z смещения разные)
  static void Pow2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale=1);///<возведение элементов тензора в квадрат
  static void SQRT(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_sqrt_value);//вычисление квадратного корня из элементов тензора
  static void SumXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,type_t scale=1);///<вычислить сумму элементов по X и Y для каждого Z
  static void AddToXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_UnitWZ,type_t scale_input=1,type_t scale_unit_wz=1);///<прибавить одинаковые значения элементов по X и Y для каждого Z и W
  static void LayerNormalizeX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta);///<выполнить нормализацию по слою X
  static void LayerAddX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_ValueX);///<добавить значеня по слою X
  static void SubTensor(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,int32_t w,int32_t z,int32_t y,int32_t x);///<скопировать один тензор в другой с позиции
  static void SplitKQVTensor(CTensor<type_t> &cTensor_Q,CTensor<type_t> &cTensor_K,CTensor<type_t> &cTensor_V,const CTensor<type_t> &cTensor_QKV,int32_t num_total_tokens,int32_t head_dim,int32_t arch_dim,int32_t q_split_index,int32_t k_split_index,int32_t v_split_index);//разделить тензор на Q,K,V

  template<class kernel_output_t,class kernel_left_t,class kernel_right_t>
  static void MulAbstract(CTensor<type_t> &cTensor_Output,kernel_output_t &sTensorKernel_Output,const CTensor<type_t> &cTensor_Left,kernel_left_t &sTensorKernel_Left,const CTensor<type_t> &cTensor_Right,kernel_right_t &sTensorKernel_Right);///<умножить тензоры

  static void Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить тензоры
  static void TransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить транспонированный левый тензор на правый
  static void Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const type_t &value_right);///<умножить тензор на число
  static void Mul(CTensor<type_t> &cTensor_Output,const type_t &value_left,const CTensor<type_t> &cTensor_Right);///<умножить тензор на число
  static void Transpose(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<транспонировать тензор
  static void TensorItemProduction(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Left,CTensor<type_t> &cTensor_Right);///<поэлементное произведение тензора на тензор
  static CTensor<type_t> Transpose(const CTensor<type_t> &cTensor_Input);///<получить транспонированный тензор

  static void UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y);///<увеличение разрешения тензора
  static void DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y);///<уменьшение разрешения тензора

  static void MaxPooling(CTensor<type_t> &cTensor_Output,CTensor<SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y);///<уменьшение разрешения тензора выборкой большего элемента
  static void MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y);///<обратный проход при увеличении разрешения тензора выборкой большего элемента
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

static uint32_t Mod(uint32_t param,uint32_t divider)
{
 uint32_t div=param/divider;
 return(param-div*divider);
}

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
 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_in_a=Mod(w,cTensor_InputA.Size_W);
  uint32_t w_in_b=Mod(w,cTensor_InputB.Size_W);
  uint32_t w_out=Mod(w,cTensor_Output.Size_W);
  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t v=0;
     if (z<cTensor_InputA.Size_Z) v=cTensor_InputA.GetElement(w_in_a,z,y,x);
                             else v=cTensor_InputB.GetElement(w_in_b,z-cTensor_InputA.Size_Z,y,x);
     cTensor_Output.SetElement(w_out,z,y,x,v);
    }
   }
  }
 }
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
 for(uint32_t w=0;w<cTensor_OutputA.Size_W;w++)
 {
  uint32_t w_in=Mod(w,cTensor_Input.Size_W);
  uint32_t w_out_a=Mod(w,cTensor_OutputA.Size_W);
  uint32_t w_out_b=Mod(w,cTensor_OutputB.Size_W);
  for(uint32_t z=0;z<cTensor_OutputA.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_OutputA.Size_Y;y++)
   {
    if (y>=cTensor_Input.Size_Y) continue;
    for(uint32_t x=0;x<cTensor_OutputA.Size_X;x++)
    {
     if (x>=cTensor_Input.Size_X) continue;
     type_t v=cTensor_Input.GetElement(w_in,z,y,x);
     if (z<cTensor_OutputA.Size_Z) cTensor_OutputA.SetElement(w_out_a,z,y,x,v);
                              else cTensor_OutputB.SetElement(w_out_b,z-cTensor_OutputA.Size_Z,y,x,v);
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//записать в тензор число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Fill(CTensor<type_t> &cTensor_Output,type_t value)
{
 cTensor_Output.Fill(value);
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
 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t i=cTensor_Input.GetElement(w_input,z,y,x);
     type_t o=1.0/i;
     cTensor_Output.SetElement(w_output,z,y,x,o);
    }
   }
  }
 }
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

 uint32_t left_w=cTensor_Left.Size_W;
 uint32_t right_w=cTensor_Right.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_left=w%left_w;
  uint32_t w_right=w%right_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t l=cTensor_Left.GetElement(w_left,z,y,x);
     type_t r=cTensor_Right.GetElement(w_right,z,y,x);
     type_t o=(left_scale*l)/(right_scale*r);
     cTensor_Output.SetElement(w_output,z,y,x,o);
    }
   }
  }
 }
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
  throw "CTensor::Add(CTensor &cTensor_Output,const CTensor &cTensor_Left,const CTensor &cTensor_Right): Размерности тензоров не совпадают!";
 }

 uint32_t left_w=cTensor_Left.Size_W;
 uint32_t right_w=cTensor_Right.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_left=w%left_w;
  uint32_t w_right=w%right_w;
  uint32_t w_output=w%output_w;
  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t l=cTensor_Left.GetElement(w_left,z,y,x);
     type_t r=cTensor_Right.GetElement(w_right,z,y,x);
     type_t o=(left_scale*l)+(right_scale*r);
     cTensor_Output.SetElement(w_output,z,y,x,o);
    }
   }
  }
 }
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

 for(uint32_t z=0;z<cTensor_Left.Size_Z;z++)
 {
  for(uint32_t y=0;y<cTensor_Left.Size_Y;y++)
  {
   for(uint32_t x=0;x<cTensor_Left.Size_X;x++)
   {
    type_t summ_left=0;
    type_t summ_right=0;
    for(uint32_t w=0;w<cTensor_Left.Size_W;w++)
    {
     type_t v=cTensor_Left.GetElement(w,z,y,x);
     summ_left+=v;
    }
    summ_left*=left_scale;
    for(uint32_t w=0;w<cTensor_Right.Size_W;w++)
    {
     type_t v=cTensor_Right.GetElement(w,z,y,x);
     summ_right+=v;
    }
    summ_right*=right_scale;
    type_t summ_output=summ_left+summ_right;
    cTensor_Output.SetElement(0,z,y,x,summ_output);//помещаем сумму в нулевой слой w
    for(uint32_t w=1;w<cTensor_Output.Size_W;w++) cTensor_Output.SetElement(w,z,y,x,0);//все остальные слои w обнулены
   }
  }
 }
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
  throw "CTensor::Sub(CTensor &cTensor_Output,const CTensor &cTensor_Left,const CTensor &cTensor_Right): Размерности тензоров не совпадают!";
 }

 uint32_t left_w=cTensor_Left.Size_W;
 uint32_t right_w=cTensor_Right.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_left=w%left_w;
  uint32_t w_right=w%right_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t l=cTensor_Left.GetElement(w_left,z,y,x);
     type_t r=cTensor_Right.GetElement(w_right,z,y,x);
     type_t o=(left_scale*l)-(right_scale*r);
     cTensor_Output.SetElement(w_output,z,y,x,o);
    }
   }
  }
 }
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

 uint32_t working_w=cTensor_Working.Size_W;
 uint32_t bias_w=cTensor_Bias.Size_W;

 for(uint32_t w=0;w<cTensor_Working.Size_W;w++)
 {
  uint32_t w_working=w%working_w;
  uint32_t w_bias=w%bias_w;

  for(uint32_t z=0;z<cTensor_Working.Size_Z;z++)
  {
   type_t b=cTensor_Bias.GetElement(w_bias,z,0,0);
   for(uint32_t y=0;y<cTensor_Working.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Working.Size_X;x++)
    {
     type_t s=cTensor_Working.GetElement(w_working,z,y,x);
     s+=b;
     cTensor_Working.SetElement(w_working,z,y,x,s);
    }
   }
  }
 }
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

 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t i=cTensor_Input.GetElement(w_input,z,y,x);
     type_t o=i*i*scale;
     cTensor_Output.SetElement(w_output,z,y,x,o);
    }
   }
  }
 }
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

 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t i=cTensor_Input.GetElement(w_input,z,y,x);
     type_t o=(sqrt(i+add_sqrt_value))*scale;
     cTensor_Output.SetElement(w_output,z,y,x,o);
    }
   }
  }
 }
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

 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;
  for(uint32_t z=0;z<cTensor_Input.Size_Z;z++)
  {
   type_t summ=0;
   for(uint32_t y=0;y<cTensor_Input.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Input.Size_X;x++)
    {
     type_t e=cTensor_Input.GetElement(w_input,z,y,x);
     summ+=e;
    }
   }
   summ*=scale;
   cTensor_Output.SetElement(w_output,z,0,0,summ);
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

 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;
 uint32_t unit_w=cTensor_UnitWZ.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;
  uint32_t w_unit=w%unit_w;
  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   type_t d=cTensor_UnitWZ.GetElement(w_unit,z,0,0);
   d*=scale_unit_wz;
   type_t *d_xin=cTensor_Input.GetColumnPtr(w_input,z,0);
   type_t *d_xout=cTensor_Output.GetColumnPtr(w_output,z,0);
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++,d_xin++,d_xout++)
    {
     type_t v=(*d_xin);
     v*=scale_input,
     (*d_xout)=v+d;
    }
   }
  }
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

 for(int32_t w=0;w<cTensor_Input.Size_W;w++)
 {
  uint32_t w_in=Mod(w,cTensor_Input.Size_W);
  uint32_t w_out=Mod(w,cTensor_Output.Size_W);
  for(int32_t z=0;z<cTensor_Input.Size_Z;z++)
  {
   for(int32_t y=0;y<cTensor_Input.Size_Y;y++)
   {
    type_t *d_xin=cTensor_Input.GetTensorDataPtr(w_in,z)+y*cTensor_Input.Size_X;
    type_t *d_xout=cTensor_Output.GetTensorDataPtr(w_out,z)+y*cTensor_Input.Size_X;
    type_t *d_valuex=cTensor_ValueX.GetTensorDataPtr(w_in,z);
    for(uint32_t x=0;x<cTensor_Input.Size_X;x++,d_xin++,d_xout++,d_valuex++)
    {
     type_t v=(*d_xin);
     v+=(*d_valuex);
     *d_xout=v;
    }
   }
  }
 }
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

 for(int32_t wo=0;wo<cTensor_Output.Size_W;wo++)
 {
  int32_t wi=wo+w;
  for(int32_t zo=0;zo<cTensor_Output.Size_Z;zo++)
  {
   int32_t zi=zo+z;
   for(int32_t yo=0;yo<cTensor_Output.Size_Y;yo++)
   {
    type_t yi=yo+y;
    for(int32_t xo=0;xo<cTensor_Output.Size_X;xo++)
    {
     type_t xi=xo+x;
     type_t value=cTensor_Input.GetElement(wi,zi,yi,xi);
     cTensor_Output.SetElement(wo,zo,yo,xo,value);
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//разделить тензор на Q,K,V
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SplitKQVTensor(CTensor<type_t> &cTensor_Q,CTensor<type_t> &cTensor_K,CTensor<type_t> &cTensor_V,const CTensor<type_t> &cTensor_QKV,int32_t num_total_tokens,int32_t head_dim,int32_t arch_dim,int32_t q_split_index,int32_t k_split_index,int32_t v_split_index)
{
 for(int32_t w=0;w<cTensor_K.Size_W;w++)
 {
  uint32_t wp=w;
  for(int32_t z=0;z<cTensor_K.Size_Z;z++)
  {
   uint32_t zp=z;
   for(int32_t y=0;y<cTensor_K.Size_Y;y++)
   {
    uint32_t yp=y;
    for(uint32_t x=0;x<cTensor_K.Size_X;x++)
    {
     uint32_t xp=x;
     type_t q=0;
     type_t k=0;
     type_t v=0;
     if (yp<num_total_tokens)
     {
      q=cTensor_QKV.GetElement(wp,0,yp,q_split_index*arch_dim+zp*head_dim+xp);
      k=cTensor_QKV.GetElement(wp,0,yp,k_split_index*arch_dim+zp*head_dim+xp);
      v=cTensor_QKV.GetElement(wp,0,yp,v_split_index*arch_dim+zp*head_dim+xp);
     }
     cTensor_Q.SetElement(wp,zp,yp,xp,q);
     cTensor_K.SetElement(wp,zp,yp,xp,k);
     cTensor_V.SetElement(wp,zp,yp,xp,v);
    }
   }
  }
 }
}


/*
//----------------------------------------------------------------------------------------------------
//умножить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Left.Size_X!=cTensor_Right.Size_Y  || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Left.Size_Y || cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Mul(CTensor &cTensor_Output,const CTensor &cTensor_Left,const CTensor &cTensor_Right): Размерности тензоров не совпадают!";
 }

 uint32_t left_w=cTensor_Left.Size_W;
 uint32_t right_w=cTensor_Right.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_left=w%left_w;
  uint32_t w_right=w%right_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Left.Size_Z;z++)
  {
   type_t *m=cTensor_Output.GetColumnPtr(w_output,z,0);
   for(uint32_t y=0;y<cTensor_Left.Size_Y;y++)
   {
    const type_t *m1_begin=cTensor_Left.GetColumnPtr(w_left,z,0)+y*cTensor_Left.Size_X;
    for(uint32_t x=0;x<cTensor_Right.Size_X;x++,m++)
    {
     type_t s=0;
     const type_t *m2=cTensor_Right.GetColumnPtr(w_right,z,0)+x;
     const type_t *m1=m1_begin;
     for(uint32_t n=0;n<cTensor_Left.Size_X;n++,m1++,m2+=cTensor_Right.Size_X) s+=(*m1)*(*m2);
     *m=s;
    }
   }
  }
 }
*/

//----------------------------------------------------------------------------------------------------
//умножить тензоры (оптимизированная версия с блокировкой кэша и OpenMP)
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Left.Size_X!=cTensor_Right.Size_Y  || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Left.Size_Y || cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Mul: Размерности тензоров не совпадают!";
 }

 uint32_t LY = cTensor_Left.Size_Y;
 uint32_t LX = cTensor_Left.Size_X;
 uint32_t RX = cTensor_Right.Size_X;
 uint32_t Z = cTensor_Left.Size_Z;
 uint32_t W = cTensor_Output.Size_W;

 // Обязательно обнуляем выходной тензор перед накоплением суммы
 cTensor_Output.Zero();

 // Размер блока для кэширования. 32 - эмпирически хороший выбор для L1 кэша процессора
 const uint32_t BLOCK_SIZE = 32;

 for(uint32_t w=0;w<W;w++)
 {
  for(uint32_t z=0;z<Z;z++)
  {
   uint32_t w_left = w % cTensor_Left.Size_W;
   uint32_t w_right = w % cTensor_Right.Size_W;

   // Получаем сырые указатели на начало срезов матриц для конкретных w и z
   const type_t* left_base = cTensor_Left.GetColumnPtr(w_left, z, 0);
   const type_t* right_base = cTensor_Right.GetColumnPtr(w_right, z, 0);
   type_t* out_base = cTensor_Output.GetColumnPtr(w, z, 0);

   // Разбиваем умножение на блоки для оптимизации работы с кэшем
   for(uint32_t yy = 0; yy < LY; yy += BLOCK_SIZE)
   {
    for(uint32_t kk = 0; kk < LX; kk += BLOCK_SIZE)
    {
     for(uint32_t xx = 0; xx < RX; xx += BLOCK_SIZE)
     {
      // Определяем границы текущего блока (с учетом неполных блоков по краям)
      uint32_t y_end = std::min(yy + BLOCK_SIZE, LY);
      uint32_t k_end = std::min(kk + BLOCK_SIZE, LX);
      uint32_t x_end = std::min(xx + BLOCK_SIZE, RX);

      // Вычисляем блок матрицы
      for(uint32_t y = yy; y < y_end; y++)
      {
       for(uint32_t k = kk; k < k_end; k++)
       {
        // Значение левой матрицы вычисляется один раз и мультиплексируется по X
        type_t left_val = left_base[y * LX + k];

        // Внутренний цикл идет строго по памяти последовательно для правой и выходной матрицы
        for(uint32_t x = xx; x < x_end; x++)
        {
         out_base[y * RX + x] += left_val * right_base[k * RX + x];
        }
       }
      }
     }
    }
   }
  }
 }
}
/*
//----------------------------------------------------------------------------------------------------
//умножить транспонированный левый тензор на правый
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::TransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Left.Size_Y!=cTensor_Right.Size_Y  || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Left.Size_X || cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::TransposeMul(CTensor &cTensor_Output,const CTensor &cTensor_Left,const CTensor &cTensor_Right): Размерности тензоров не совпадают!";
 }

 uint32_t left_w=cTensor_Left.Size_W;
 uint32_t right_w=cTensor_Right.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_left=w%left_w;
  uint32_t w_right=w%right_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Left.Size_Z;z++)
  {
   type_t *m=cTensor_Output.GetColumnPtr(w_output,z,0);
   for(uint32_t y=0;y<cTensor_Left.Size_X;y++)
   {
    const type_t *m1_begin=cTensor_Left.GetColumnPtr(w_left,z,0)+y;
    for(uint32_t x=0;x<cTensor_Right.Size_X;x++,m++)
    {
     type_t s=0;
     const type_t *m2=cTensor_Right.GetColumnPtr(w_right,z,0)+x;
     const type_t *m1=m1_begin;
     for(uint32_t n=0;n<cTensor_Left.Size_Y;n++,m1+=cTensor_Left.Size_X,m2+=cTensor_Right.Size_X) s+=(*m1)*(*m2);
     *m=s;
    }
   }
  }
 }
}
*/

//----------------------------------------------------------------------------------------------------
//умножить транспонированный левый тензор на правый (оптимизированная версия)
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::TransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Left.Size_Y!=cTensor_Right.Size_Y  || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Left.Size_X || cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::TransposeMul: Размерности тензоров не совпадают!";
 }

 uint32_t K_DIM = cTensor_Left.Size_Y; // Общая размерность (внутренняя)
 uint32_t L_X = cTensor_Left.Size_X;   // Выходная Y (строки транспонированной левой)
 uint32_t R_X = cTensor_Right.Size_X;  // Выходная X (столбцы правой)
 uint32_t Z = cTensor_Left.Size_Z;
 uint32_t W = cTensor_Output.Size_W;

 // Обязательно обнуляем выходной тензор перед накоплением суммы
 cTensor_Output.Zero();

 // Размер блока для кэширования
 const uint32_t BLOCK_SIZE = 32;

 #ifdef GOFAST
 #pragma omp parallel for collapse(2) schedule(static)
 #endif
 for(uint32_t w=0;w<W;w++)
 {
  for(uint32_t z=0;z<Z;z++)
  {
   uint32_t w_left = w % cTensor_Left.Size_W;
   uint32_t w_right = w % cTensor_Right.Size_W;

   // Получаем сырые указатели на начало срезов матриц
   const type_t* left_base = cTensor_Left.GetColumnPtr(w_left, z, 0);
   const type_t* right_base = cTensor_Right.GetColumnPtr(w_right, z, 0);
   type_t* out_base = cTensor_Output.GetColumnPtr(w, z, 0);

   // Разбиваем умножение на блоки для оптимизации работы с кэшем
   for(uint32_t kk = 0; kk < K_DIM; kk += BLOCK_SIZE)
   {
    for(uint32_t yy = 0; yy < L_X; yy += BLOCK_SIZE)
    {
     for(uint32_t xx = 0; xx < R_X; xx += BLOCK_SIZE)
     {
      // Определяем границы текущего блока (с учетом неполных блоков по краям)
      uint32_t k_end = std::min(kk + BLOCK_SIZE, K_DIM);
      uint32_t y_end = std::min(yy + BLOCK_SIZE, L_X);
      uint32_t x_end = std::min(xx + BLOCK_SIZE, R_X);

      // Вычисляем блок матрицы
      // Порядок K -> Y -> X оптимален для A^T * B, так как обеспечивает
      // последовательное чтение элементов из левой матрицы (по столбцу исходной = по строке транспонированной)
      for(uint32_t k = kk; k < k_end; k++)
      {
       for(uint32_t y = yy; y < y_end; y++)
       {
        // Читаем элемент левой матрицы. При переходе по Y мы двигаемся по памяти последовательно!
        type_t left_val = left_base[k * L_X + y];

        // Внутренний цикл идет строго последовательно по памяти для правой и выходной матрицы
        for(uint32_t x = xx; x < x_end; x++)
        {
         out_base[y * R_X + x] += left_val * right_base[k * R_X + x];
        }
       }
      }
     }
    }
   }
  }
 }
}




//----------------------------------------------------------------------------------------------------
//умножить тензор на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const type_t &value_right)
{
 if (cTensor_Output.Size_X!=cTensor_Left.Size_X || cTensor_Output.Size_Y!=cTensor_Left.Size_Y || cTensor_Output.Size_Z!=cTensor_Left.Size_Z)
 {
  throw "CTensor::Mul(CTensor &cTensor_Output,const CTensor &cTensor_Left,const type_t &value_right): Размерности тензоров не совпадают!";
 }

 uint32_t left_w=cTensor_Left.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_left=w%left_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t l=cTensor_Left.GetElement(w_left,z,y,x);
     type_t o=l*value_right;
     cTensor_Output.SetElement(w_output,z,y,x,o);
    }
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
//умножить тензор на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const type_t &value_left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Mul(CTensor &cTensor_Output,const type_t &value_left,const CTensor &cTensor_Right): Размерности тензоров не совпадают!";
 }

 uint32_t right_w=cTensor_Right.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_right=w%right_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t r=cTensor_Right.GetElement(w_right,z,y,x);
     type_t o=r*value_left;
     cTensor_Output.SetElement(w_output,z,y,x,o);
    }
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
//транспонировать тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Transpose(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Output.Size_Y!=cTensor_Input.Size_X || cTensor_Output.Size_X!=cTensor_Input.Size_Y || cTensor_Output.Size_Z!=cTensor_Input.Size_Z || cTensor_Output.Size_W!=cTensor_Input.Size_W)
 {
  throw "void CTensor::Transpose(CTensor &cTensor_Output,const CTensor &cTensor_Input): Размерности матриц не совпадают!";
 }

 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Input.Size_Z;z++)
  {
   const type_t *i_ptr=cTensor_Input.GetColumnPtr(w_input,z,0);
   type_t *o_ptr=cTensor_Output.GetColumnPtr(w_output,z,0);
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
}

//----------------------------------------------------------------------------------------------------
//поэлементное произведение тензора на тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::TensorItemProduction(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Left,CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Right.Size_X!=cTensor_Left.Size_X || cTensor_Right.Size_Y!=cTensor_Left.Size_Y || cTensor_Right.Size_Z!=cTensor_Left.Size_Z) throw("Ошибка поэлементного умножения тензора на тензор");
 if (cTensor_Right.Size_X!=cTensor_Output.Size_X || cTensor_Right.Size_Y!=cTensor_Output.Size_Y || cTensor_Right.Size_Z!=cTensor_Output.Size_Z) throw("Ошибка поэлементного умножения тензора на тензор");

 uint32_t left_w=cTensor_Left.Size_W;
 uint32_t right_w=cTensor_Right.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_left=w%left_w;
  uint32_t w_right=w%right_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     type_t l=cTensor_Left.GetElement(w_left,z,y,x);
     type_t r=cTensor_Right.GetElement(w_right,z,y,x);
     type_t o=l*r;
     cTensor_Output.SetElement(w_output,z,y,x,o);
    }
   }
  }
 }
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
///!увеличение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X/upsampling_x || cTensor_Input.Size_Y!=cTensor_Output.Size_Y/upsampling_y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::UpSampling: Размерности тензоров не совпадают!";
 }

 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    uint32_t iy=y/upsampling_y;
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     uint32_t ix=x/upsampling_x;
 	 if (ix>=cTensor_Input.Size_X || iy>=cTensor_Input.Size_Y)
	 {
      cTensor_Output.SetElement(w_output,z,y,x,0);
	  continue;
	 }
     type_t item=cTensor_Input.GetElement(w_input,z,iy,ix);
     cTensor_Output.SetElement(w_output,z,y,x,item);
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
///!уменьшение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y)
{
 if (cTensor_Input.Size_X/downsampling_x!=cTensor_Output.Size_X || cTensor_Input.Size_Y/downsampling_y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Input.Size_W!=cTensor_Output.Size_W)
 {
  throw "CTensor::DownSampling: Размерности тензоров не совпадают!";
 }

 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;

  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     uint32_t ix=x*downsampling_x;
     uint32_t iy=y*downsampling_y;
 	 type_t summ=0;
     for(uint32_t dx=0;dx<downsampling_x;dx++)
	 {
      uint32_t xp=ix+dx;
	  if (xp>=cTensor_Input.Size_X) continue;
      for(uint32_t dy=0;dy<downsampling_y;dy++)
	  {
       uint32_t yp=iy+dy;
  	   if (yp>=cTensor_Input.Size_Y) continue;
	   summ+=cTensor_Input.GetElement(w_input,z,yp,xp);
	  }
	 }
	 summ/=static_cast<type_t>(downsampling_x*downsampling_y);
     cTensor_Output.SetElement(w_output,z,y,x,summ);
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
///!увеличение разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPooling(CTensor<type_t> &cTensor_Output,CTensor<SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y)
{
 if ((cTensor_Input.Size_X/pooling_x)!=cTensor_Output.Size_X || (cTensor_Input.Size_Y/pooling_y)!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Input.Size_W!=cTensor_Output.Size_W)
 {
  throw "CTensor::MaxPooling: Размерности тензоров не совпадают!";
 }

 uint32_t output_x=cTensor_Output.Size_X;
 uint32_t output_y=cTensor_Output.Size_Y;
 uint32_t output_z=cTensor_Output.Size_Z;

 uint32_t input_x=cTensor_Input.Size_X;
 uint32_t input_y=cTensor_Input.Size_Y;
 uint32_t input_z=cTensor_Input.Size_Z;

 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;
 uint32_t position_w=cTensor_Position.Size_W;

 for(uint32_t w=0;w<output_w;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;
  uint32_t w_position=w%position_w;

  for(uint32_t z=0;z<output_z;z++)
  {
   for(uint32_t y=0;y<output_y;y++)
   {
    for(uint32_t x=0;x<output_x;x++)
    {
     uint32_t ix=x*pooling_x;
     uint32_t iy=y*pooling_y;
     type_t max=cTensor_Input.GetElement(w_input,z,iy,ix);
     uint32_t max_x=ix;
     uint32_t max_y=iy;
     for(uint32_t py=0;py<pooling_y;py++)
     {
      for(uint32_t px=0;px<pooling_x;px++)
      {
       type_t e=cTensor_Input.GetElement(w_input,z,iy+py,ix+px);
       if (e>max)
       {
        max=e;
        max_x=ix+px;
        max_y=iy+py;
       }
      }
     }
     cTensor_Output.SetElement(w_output,z,y,x,max);
     SPos sPos;
     sPos.X=max_x;
     sPos.Y=max_y;
     cTensor_Position.SetElement(w_position,z,y,x,sPos);
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
///!обратный проход при увеличении разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y)
{
 if (cTensor_Input.Size_X!=(cTensor_Output.Size_X/pooling_x) || cTensor_Input.Size_Y!=(cTensor_Output.Size_Y/pooling_y) || cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Input.Size_W!=cTensor_Output.Size_W)
 {
  throw "CTensor::MaxPooling: Размерности тензоров не совпадают!";
 }

 uint32_t input_x=cTensor_Input.Size_X;
 uint32_t input_y=cTensor_Input.Size_Y;
 uint32_t input_z=cTensor_Input.Size_Z;

 CTensorMath<type_t>::Fill(cTensor_Output,0);

 uint32_t input_w=cTensor_Input.Size_W;
 uint32_t output_w=cTensor_Output.Size_W;
 uint32_t position_w=cTensor_Position.Size_W;

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  uint32_t w_input=w%input_w;
  uint32_t w_output=w%output_w;
  uint32_t w_position=w%position_w;

  for(uint32_t z=0;z<input_z;z++)
  {
   for(uint32_t y=0;y<input_y;y++)
   {
    for(uint32_t x=0;x<input_x;x++)
    {
     type_t delta=cTensor_Input.GetElement(w_input,z,y,x);
     SPos sPos=cTensor_Position.GetElement(w_position,z,y,x);
     cTensor_Output.SetElement(w_output,z,sPos.Y,sPos.X,delta);
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
///!выполнить отсечку значений тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value)
{
 type_t *item_ptr=cTensor.GetColumnPtr(0,0,0);

 uint32_t size_y=cTensor.Size_Y;
 uint32_t size_x=cTensor.Size_X;
 uint32_t size_z=cTensor.Size_Z;
 uint32_t size_w=cTensor.Size_W;

 for(uint32_t w=0;w<size_w;w++)
 {
  for(uint32_t z=0;z<size_z;z++)
  {
   for(uint32_t y=0;y<size_y;y++)
   {
    for(uint32_t x=0;x<size_x;x++,item_ptr++)
    {
     if (*item_ptr<min_value) *item_ptr=min_value;
     if (*item_ptr>max_value) *item_ptr=max_value;
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
///!ввыставить знак элементам тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Signum(CTensor<type_t> &cTensor)
{
 type_t *item_ptr=cTensor.GetColumnPtr(0,0,0);

 uint32_t size_y=cTensor.Size_Y;
 uint32_t size_x=cTensor.Size_X;
 uint32_t size_z=cTensor.Size_Z;
 uint32_t size_w=cTensor.Size_W;

 for(uint32_t w=0;w<size_w;w++)
 {
  for(uint32_t z=0;z<size_z;z++)
  {
   for(uint32_t y=0;y<size_y;y++)
   {
    for(uint32_t x=0;x<size_x;x++,item_ptr++)
    {
     if (*item_ptr<0) *item_ptr=-1;
     if (*item_ptr>0) *item_ptr=1;
    }
   }
  }
 }
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

 uint32_t size_y=cTensor_Weight.Size_Y;
 uint32_t size_x=cTensor_Weight.Size_X;
 uint32_t size_z=cTensor_Weight.Size_Z;
 uint32_t size_w=cTensor_Weight.Size_W;

 for(uint32_t w=0;w<size_w;w++)
 {
  for(uint32_t z=0;z<size_z;z++)
  {
   for(uint32_t y=0;y<size_y;y++)
   {
    for(uint32_t x=0;x<size_x;x++)
    {

     type_t dw=cTensor_dWeight.GetElement(w,z,y,x);
     type_t m=cTensor_M.GetElement(w,z,y,x);
     type_t v=cTensor_V.GetElement(w,z,y,x);

     dw/=static_cast<type_t>(batch_size);

     m=beta1*m+(1.0-beta1)*dw;
     v=beta2*v+(1.0-beta2)*dw*dw;

     type_t mc=m/(1.0-pow(beta1,iteration));
     type_t vc=v/(1.0-pow(beta2,iteration));

     dw=speed*mc/(sqrt(vc)+epsilon);

     cTensor_M.SetElement(w,z,y,x,m);
     cTensor_V.SetElement(w,z,y,x,v);
     //корректируем веса
     type_t wc=cTensor_Weight.GetElement(w,z,y,x);
     cTensor_Weight.SetElement(w,z,y,x,wc-dw);
    }
   }
  }
 }
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

 const uint32_t C=cTensor_Input.Size_Z;//число каналов
 for(uint32_t w=0;w<cTensor_Input.Size_Z;w++)
 {
  //шаг времени для данного элемента пакета (w — индекс картинки в батче)
  type_t time_step=static_cast<type_t>(cTensor_TimeStep.GetElement(w,0,0,0));
  for(uint32_t z=0;z<cTensor_Input.Size_Z;z++)
  {
   //Временное кодирование: зависит только от z и t
   const uint32_t i=z/2;//индекс "полупары" sin/cos
   const type_t exponent=static_cast<type_t>(2*i)/static_cast<type_t>(C);
   const type_t freq=pow(10000.0f,exponent);
   const type_t angle=time_step/freq;
   for(uint32_t y=0;y<cTensor_Input.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Input.Size_X;x++)
    {
     type_t emb_z;
     if ((z & 0x01)==0) emb_z=sin(angle)*scale;//чётный канал
                   else emb_z=cos(angle)*scale;//нечётный канал
     //добавление одинакового значения ко всем пикселям канала z
     type_t value=cTensor_Input.GetElement(w,z,y,x);
     value+=emb_z;
     cTensor_Output.SetElement(w,z,y,x,value);
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//!создать матрицу исключения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::CreateDropOutMatrix(CTensor<type_t> &cTensor_Output,type_t drop_out)
{
 //создаём матрицу исключения
 Fill(cTensor_Output,0);
 type_t mult=static_cast<type_t>(1.0/(1.0-drop_out));
 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++)
    {
     if (CRandom<type_t>::GetRandValue(1)>=drop_out) cTensor_Output.SetElement(w,z,y,x,mult);
    }
   }
  }
 }
}


//----------------------------------------------------------------------------------------------------
//!задать тензор случайными значениями с нормальным распределением
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SetNormalNoise(CTensor<type_t> &cTensor_Output)
{
 throw("Функция SetNormalNoise из CTensorMath не должна вызываться при компиляции для CPU. Отключите макрос активации заполнения тензора на GPU!");
}
//----------------------------------------------------------------------------------------------------
//!заполнить тензоры зашумлённым изображением и шумом
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::GetNoiseImageAndNoise(CTensor<type_t> &cTensor_NoisyImage,CTensor<type_t> &cTensor_Noise,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_SqrtAlphaBar,const CTensor<type_t> &cTensor_SqrtOneMinusAlphaBar)
{
 throw("Функция GetNoiseImageAndNoise из CTensorMath не должна вызываться при компиляции для CPU. Отключите макрос активации заполнения тензора на GPU!");
}

//----------------------------------------------------------------------------------------------------
//!ограничить тензор по норме XY
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::ClipByNormXY(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold)
{
 if (cTensor_Input.Size_W!=cTensor_Output.Size_W || cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::ClipByNorm: Размерности тензоров не совпадают!";
 }
 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  for(uint32_t z=0;z<cTensor_Output.Size_Z;z++)
  {
   //суммируем по X и Y
   const type_t *d_xin=cTensor_Input.GetColumnPtr(w,z,0);
   type_t *d_xout=cTensor_Output.GetColumnPtr(w,z,0);
   const type_t *d_xin_local=d_xin;
   type_t summ=0;
   for(uint32_t y=0;y<cTensor_Output.Size_Y;y++)
   {
    for(uint32_t x=0;x<cTensor_Output.Size_X;x++,d_xin_local++)
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
    for(uint32_t y=0;y<cTensor_Input.Size_Y;y++)
    {
     for(uint32_t x=0;x<cTensor_Input.Size_X;x++,d_xin++,d_xout++)
     {
      type_t v=*d_xin;
      v*=k;
      *d_xout=v;
     }
    }
   }
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
  throw "CTensor::ClipByNorm: Размерности тензоров не совпадают!";
 }

 for(uint32_t w=0;w<cTensor_Input.Size_W;w++)
 {
  uint32_t w_in=w;
  uint32_t w_out=w;
  for(uint32_t z=0;z<cTensor_Input.Size_Z;z++)
  {
   for(uint32_t y=0;y<cTensor_Input.Size_Y;y++)
   {
    //суммируем по X
    const type_t *d_xin=cTensor_Input.GetColumnPtr(w,z,y);
    type_t *d_xout=cTensor_Output.GetColumnPtr(w,z,y);
    const type_t *d_xin_local=d_xin;
    type_t summ=0;
    for(uint32_t x=0;x<cTensor_Input.Size_X;x++,d_xin_local++)
    {
     type_t v=*d_xin_local;
     summ+=v*v;
    }
    summ=sqrt(summ);
    //нормируем, если нужно
    if (summ>threshold)
    {
     type_t k=threshold/summ;
     for(uint32_t x=0;x<cTensor_Input.Size_X;x++,d_xin++,d_xout++)
     {
      type_t v=*d_xin;
      v*=k;
      *d_xout=v;
     }
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//!выполнить прямой проход GroupNorm
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::GroupNormForward(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_Gamma,CTensor<type_t> &cTensor_Beta,CTensor<type_t> &cTensor_XHAT,CTensor<type_t> &cTensor_InvStd,uint32_t num_groups,uint32_t channels_per_group,type_t epsilon)
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
void CTensorMath<type_t>::GroupNormBackward(CTensor<type_t> &cTensor_Delta_Array,CTensor<type_t> &cTensor_XHAT_Array,CTensor<type_t> &cTensor_Gamma,CTensor<type_t> &cTensor_InvStd_Array,CTensor<type_t> &cTensor_PrevLayerError_Array,CTensor<type_t> &cTensor_dGamma,CTensor<type_t> &cTensor_dBeta,uint32_t num_groups,uint32_t channels_per_group,bool calc_weights)
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
