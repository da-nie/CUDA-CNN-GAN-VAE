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
  static void Sub(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<вычесть тензоры
  static void AddBias(CTensor<type_t> &cTensor_Working,const CTensor<type_t> &cTensor_Bias);///<добавить смещения к элементам тензора (смещения одинаковы для x и y, но по z смещения разные)
  static void Pow2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale=1);///<возведение элементов тензора в квадрат
  static void SQRT(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_sqrt_value);//вычисление квадратного корня из элементов тензора
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

 uint32_t LY=cTensor_Left.Size_Y;
 uint32_t LX=cTensor_Left.Size_X;
 uint32_t RX=cTensor_Right.Size_X;
 uint32_t Z=cTensor_Left.Size_Z;
 uint32_t W=cTensor_Output.Size_W;

 // Обязательно обнуляем выходной тензор перед накоплением суммы
 cTensor_Output.Zero();

 // Размер блока для кэширования. 32 - эмпирически хороший выбор для L1 кэша процессора
 const uint32_t BLOCK_SIZE=32;

 for(uint32_t w=0;w<W;w++)
 {
  for(uint32_t z=0;z<Z;z++)
  {
   uint32_t w_left=w % cTensor_Left.Size_W;
   uint32_t w_right=w % cTensor_Right.Size_W;

   // Получаем сырые указатели на начало срезов матриц для конкретных w и z
   const type_t* left_base=cTensor_Left.GetColumnPtr(w_left, z, 0);
   const type_t* right_base=cTensor_Right.GetColumnPtr(w_right, z, 0);
   type_t* out_base=cTensor_Output.GetColumnPtr(w, z, 0);

   // Разбиваем умножение на блоки для оптимизации работы с кэшем
   for(uint32_t yy=0; yy<LY; yy+=BLOCK_SIZE)
   {
    for(uint32_t kk=0; kk<LX; kk+=BLOCK_SIZE)
    {
     for(uint32_t xx=0; xx<RX; xx+=BLOCK_SIZE)
     {
      // Определяем границы текущего блока (с учетом неполных блоков по краям)
      uint32_t y_end=std::min(yy+BLOCK_SIZE, LY);
      uint32_t k_end=std::min(kk+BLOCK_SIZE, LX);
      uint32_t x_end=std::min(xx+BLOCK_SIZE, RX);

      // Вычисляем блок матрицы
      for(uint32_t y=yy; y<y_end; y++)
      {
       for(uint32_t k=kk; k<k_end; k++)
       {
        // Значение левой матрицы вычисляется один раз и мультиплексируется по X
        type_t left_val=left_base[y*LX+k];

        // Внутренний цикл идет строго по памяти последовательно для правой и выходной матрицы
        for(uint32_t x=xx; x<x_end; x++)
        {
         out_base[y*RX+x]+=left_val*right_base[k*RX+x];
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
void CTensorMath<type_t>::LeftTransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Left.Size_Y!=cTensor_Right.Size_Y  || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Left.Size_X || cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::LeftTransposeMul(CTensor &cTensor_Output,const CTensor &cTensor_Left,const CTensor &cTensor_Right): Размерности тензоров не совпадают!";
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
void CTensorMath<type_t>::LeftTransposeMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Left.Size_Y!=cTensor_Right.Size_Y  || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Left.Size_X || cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::LeftTransposeMul: Размерности тензоров не совпадают!";
 }

 uint32_t K_DIM=cTensor_Left.Size_Y; // Общая размерность (внутренняя)
 uint32_t L_X=cTensor_Left.Size_X;   // Выходная Y (строки транспонированной левой)
 uint32_t R_X=cTensor_Right.Size_X;  // Выходная X (столбцы правой)
 uint32_t Z=cTensor_Left.Size_Z;
 uint32_t W=cTensor_Output.Size_W;

 // Обязательно обнуляем выходной тензор перед накоплением суммы
 cTensor_Output.Zero();

 // Размер блока для кэширования
 const uint32_t BLOCK_SIZE=32;

 #ifdef GOFAST
 #pragma omp parallel for collapse(2) schedule(static)
 #endif
 for(uint32_t w=0;w<W;w++)
 {
  for(uint32_t z=0;z<Z;z++)
  {
   uint32_t w_left=w%cTensor_Left.Size_W;
   uint32_t w_right=w%cTensor_Right.Size_W;

   //получаем сырые указатели на начало срезов матриц
   const type_t* left_base=cTensor_Left.GetColumnPtr(w_left,z,0);
   const type_t* right_base=cTensor_Right.GetColumnPtr(w_right,z,0);
   type_t* out_base=cTensor_Output.GetColumnPtr(w,z,0);

   //разбиваем умножение на блоки для оптимизации работы с кэшем
   for(uint32_t kk=0;kk<K_DIM;kk+=BLOCK_SIZE)
   {
    for(uint32_t yy=0;yy<L_X;yy+=BLOCK_SIZE)
    {
     for(uint32_t xx=0;xx<R_X;xx+=BLOCK_SIZE)
     {
      //определяем границы текущего блока (с учетом неполных блоков по краям)
      uint32_t k_end=std::min(kk+BLOCK_SIZE,K_DIM);
      uint32_t y_end=std::min(yy+BLOCK_SIZE,L_X);
      uint32_t x_end=std::min(xx+BLOCK_SIZE,R_X);

      // Вычисляем блок матрицы
      // Порядок K -> Y -> X оптимален для A^T*B, так как обеспечивает
      // последовательное чтение элементов из левой матрицы (по столбцу исходной=по строке транспонированной)
      for(uint32_t k=kk;k<k_end;k++)
      {
       for(uint32_t y=yy;y<y_end;y++)
       {
        // Читаем элемент левой матрицы. При переходе по Y мы двигаемся по памяти последовательно!
        type_t left_val=left_base[k*L_X+y];

        // Внутренний цикл идет строго последовательно по памяти для правой и выходной матрицы
        for(uint32_t x=xx;x<x_end;x++)
        {
         out_base[y*R_X+x]+=left_val*right_base[k*R_X+x];
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


#endif
