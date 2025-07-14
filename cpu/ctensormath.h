#ifndef C_TENSOR_MATH_H
#define C_TENSOR_MATH_H

//****************************************************************************************************
//Операции над тензорами произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "ctensor.h"

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
 size_t X;
 size_t Y;
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
  static void Inv(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<вычислить обратный тензор
  static void Div(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<поделить тензоры
  static void Add(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<сложить тензоры
  static void Sub(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<вычесть тензоры
  static void AddBias(CTensor<type_t> &cTensor_Working,const CTensor<type_t> &cTensor_Bias);///<добавить смещения к элементам тензора (смещения одинаковы для x и y, но по z смещения разные)
  static void Pow2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale=1);///<возведение элементов тензора в квадрат
  static void SQRT(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_sqrt_value);//вычисление квадратного корня из элементов тензора
  static void SummXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input);///<вычислить сумму элементов по X и Y для каждого Z

  template<class kernel_output_t,class kernel_left_t,class kernel_right_t>
  static void MulAbstract(CTensor<type_t> &cTensor_Output,kernel_output_t &sTensorKernel_Output,const CTensor<type_t> &cTensor_Left,kernel_left_t &sTensorKernel_Left,const CTensor<type_t> &cTensor_Right,kernel_right_t &sTensorKernel_Right);///<умножить тензоры

  static void Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить тензоры
  static void TransponseMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<умножить транспонированный левый тензор на правый
  static void Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const type_t &value_right);///<умножить тензор на число
  static void Mul(CTensor<type_t> &cTensor_Output,const type_t &value_left,const CTensor<type_t> &cTensor_Right);///<умножить тензор на число
  static void Transponse(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<транспонировать тензор
  static void TensorItemProduction(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Left,CTensor<type_t> &cTensor_Right);///<поэлементное произведение тензора на тензор
  static CTensor<type_t> Transpose(const CTensor<type_t> &cTensor_Input);///<получить транспонированный тензор

  static void UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,size_t upsampling_x,size_t upsampling_y);///<увеличение разрешения тензора
  static void DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,size_t downsampling_x,size_t downsampling_y);///<уменьшение разрешения тензора

  static void MaxPooling(CTensor<type_t> &cTensor_Output,CTensor<SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,size_t pooling_x,size_t pooling_y);///<уменьшение разрешения тензора выборкой большего элемента
  static void MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,size_t pooling_x,size_t pooling_y);///<обратный проход при увеличении разрешения тензора выборкой большего элемента
  static void Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value);///<выполнить отсечку значений тензора

  static void Adam(CTensor<type_t> &cTensor_Weight,CTensor<type_t> &cTensor_dWeight,CTensor<type_t> &cTensor_M,CTensor<type_t> &cTensor_V,double speed,double beta1,double beta2,double epsilon,double iteration);///<выполнить алгоритм Adam к весовому тензору
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
 CTensor<type_t> cTensor(cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Left.Size_X);
 CTensorMath<type_t>::Add(cTensor,cTensor_Left,cTensor_Right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "-"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator-(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Left.Size_X);
 CTensorMath<type_t>::Sub(cTensor,cTensor_Left,cTensor_Right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator*(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Right.Size_X);
 CTensorMath<type_t>::Mul(cTensor,cTensor_Left,cTensor_Right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator*(const CTensor<type_t> &cTensor_Left,const type_t &value_right)
{
 CTensor<type_t> cTensor(cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Left.Size_X);
 CTensorMath<type_t>::Mul(cTensor,cTensor_Left,value_right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator*(const type_t &value_left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Right.Size_Z,cTensor_Right.Size_Y,cTensor_Right.Size_X);
 CTensorMath<type_t>::Mul(cTensor,value_left,cTensor_Right);
 return(cTensor);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

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
 const type_t *i_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,i_ptr++)
   {
    *o_ptr=1.0/(*i_ptr);
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
 const type_t *left_ptr=&cTensor_Left.Item[0];
 const type_t *right_ptr=&cTensor_Right.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Left.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Left.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Left.Size_X;x++,o_ptr++,left_ptr++,right_ptr++)
   {
    *o_ptr=left_scale*(*left_ptr)/(right_scale*(*right_ptr));
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
 const type_t *left_ptr=&cTensor_Left.Item[0];
 const type_t *right_ptr=&cTensor_Right.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Left.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Left.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Left.Size_X;x++,o_ptr++,left_ptr++,right_ptr++)
   {
    *o_ptr=left_scale*(*left_ptr)+right_scale*(*right_ptr);
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
 const type_t *left_ptr=&cTensor_Left.Item[0];
 const type_t *right_ptr=&cTensor_Right.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Left.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Left.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Left.Size_X;x++,o_ptr++,left_ptr++,right_ptr++)
   {
    *o_ptr=left_scale*(*left_ptr)-right_scale*(*right_ptr);
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

 type_t *work_ptr=&cTensor_Working.Item[0];
 const type_t *bias_ptr=&cTensor_Bias.Item[0];

 for(size_t z=0;z<cTensor_Working.Size_Z;z++,bias_ptr++)
 {
  type_t bias=*bias_ptr;

  for(size_t y=0;y<cTensor_Working.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Working.Size_X;x++,work_ptr++)
   {
    *work_ptr+=bias;
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

 const type_t *i_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,i_ptr++)
   {
    type_t e=(*i_ptr);
    *o_ptr=scale*e*e;
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

 const type_t *i_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,i_ptr++)
   {
    type_t e=(*i_ptr);
    *o_ptr=(sqrt(e+add_sqrt_value))*scale;
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//вычислить сумму элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SummXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::SummXY: Размерности тензоров не совпадают!";
 }
 cTensor_Input.CopyToDevice();

 type_t *output_ptr=&cTensor_Output.Item[0];
 type_t *input_ptr=&cTensor_Input.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++,output_ptr++)
 {
  type_t summ=0;

  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,input_ptr++)
   {
    summ+=*input_ptr;
   }
  }
  *output_ptr=summ;
 }
}

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
 for(size_t z=0;z<cTensor_Left.Size_Z;z++)
 {
  type_t *m=cTensor_Output.GetColumnPtr(z,0);
  for(size_t y=0;y<cTensor_Left.Size_Y;y++)
  {
   const type_t *m1_begin=cTensor_Left.GetColumnPtr(z,0)+y*cTensor_Left.Size_X;
   for(size_t x=0;x<cTensor_Right.Size_X;x++,m++)
   {
    type_t s=0;
    const type_t *m2=cTensor_Right.GetColumnPtr(z,0)+x;
    const type_t *m1=m1_begin;
    for(size_t n=0;n<cTensor_Left.Size_X;n++,m1++,m2+=cTensor_Right.Size_X) s+=(*m1)*(*m2);
    *m=s;
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
//умножить транспонированный левый тензор на правый
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::TransponseMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Left.Size_Y!=cTensor_Right.Size_Y  || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Left.Size_X || cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::TransponseMul(CTensor &cTensor_Output,const CTensor &cTensor_Left,const CTensor &cTensor_Right): Размерности тензоров не совпадают!";
 }
 for(size_t z=0;z<cTensor_Left.Size_Z;z++)
 {
  type_t *m=cTensor_Output.GetColumnPtr(z,0);
  for(size_t y=0;y<cTensor_Left.Size_X;y++)
  {
   const type_t *m1_begin=cTensor_Left.GetColumnPtr(z,0)+y;
   for(size_t x=0;x<cTensor_Right.Size_X;x++,m++)
   {
    type_t s=0;
    const type_t *m2=cTensor_Right.GetColumnPtr(z,0)+x;
    const type_t *m1=m1_begin;
    for(size_t n=0;n<cTensor_Left.Size_Y;n++,m1+=cTensor_Left.Size_X,m2+=cTensor_Right.Size_X) s+=(*m1)*(*m2);
    *m=s;
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

 const type_t *left_ptr=&cTensor_Left.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Left.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Left.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Left.Size_X;x++,o_ptr++,left_ptr++)
   {
    *o_ptr=(*left_ptr)*value_right;
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
 if (cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y)
 {
  throw "CTensor::Mul(CTensor &cTensor_Output,const type_t &value_left,const CTensor &cTensor_Right): Размерности тензоров не совпадают!";
 }

 const type_t *right_ptr=&cTensor_Right.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Right.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Right.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Right.Size_X;x++,o_ptr++,right_ptr++)
   {
    *o_ptr=(*right_ptr)*value_left;
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
//транспонировать тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Transponse(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Output.Size_Y!=cTensor_Input.Size_X || cTensor_Output.Size_X!=cTensor_Input.Size_Y || cTensor_Output.Size_Z!=cTensor_Input.Size_Z)
 {
  throw "void CTensor::Transponse(CTensor &cTensor_Output,const CTensor &cTensor_Input): Размерности матриц не совпадают!";
 }
 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  const type_t *i_ptr=cTensor_Input.GetColumnPtr(z,0);
  type_t *o_ptr=cTensor_Output.GetColumnPtr(z,0);
  for(size_t y=0;y<cTensor_Input.Size_Y;y++,o_ptr++)
  {
   type_t *o_ptr_local=o_ptr;
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr_local+=cTensor_Input.Size_Y,i_ptr++)
   {
    *o_ptr_local=*i_ptr;
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

 type_t *right_ptr=cTensor_Right.GetColumnPtr(0,0);
 type_t *output_ptr=cTensor_Output.GetColumnPtr(0,0);
 type_t *left_ptr=cTensor_Left.GetColumnPtr(0,0);

 size_t size_y=cTensor_Left.GetSizeY();
 size_t size_x=cTensor_Left.GetSizeX();
 size_t size_z=cTensor_Left.GetSizeZ();
 for(size_t z=0;z<size_z;z++)
 {
  for(size_t y=0;y<size_y;y++)
  {
   for(size_t x=0;x<size_x;x++,left_ptr++,right_ptr++,output_ptr++)
   {
    type_t a=*left_ptr;
    type_t b=*right_ptr;
   *output_ptr=a*b;
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
 CTensor<type_t> cTensor(cTensor_Input.Size_Z,cTensor_Input.Size_X,cTensor_Input.Size_Y);
 Transponse(cTensor,cTensor_Input);
 return(cTensor);
}

//----------------------------------------------------------------------------------------------------
///!увеличение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,size_t upsampling_x,size_t upsampling_y)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X/upsampling_x || cTensor_Input.Size_Y!=cTensor_Output.Size_Y/upsampling_y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::UpSampling: Размерности тензоров не совпадают!";
 }

 type_t *output_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Output.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Output.Size_Y;y++)
  {
   size_t iy=y/upsampling_y;
   for(size_t x=0;x<cTensor_Output.Size_X;x++,output_ptr++)
   {
    size_t ix=x/upsampling_x;
	if (ix>=cTensor_Input.Size_X || iy>=cTensor_Input.Size_Y)
	{
     *output_ptr=0;
	 continue;
	}
    type_t item=cTensor_Input.GetElement(z,iy,ix);
    *output_ptr=item;
   }
  }
 }

}

//----------------------------------------------------------------------------------------------------
///!уменьшение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,size_t downsampling_x,size_t downsampling_y)
{
 if (cTensor_Input.Size_X/downsampling_x!=cTensor_Output.Size_X || cTensor_Input.Size_Y/downsampling_y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::DownSampling: Размерности тензоров не совпадают!";
 }

 type_t *output_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Output.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Output.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Output.Size_X;x++,output_ptr++)
   {
    size_t ix=x*downsampling_x;
    size_t iy=y*downsampling_y;
	type_t summ=0;
    for(size_t dx=0;dx<downsampling_x;dx++)
	{
     size_t xp=ix+dx;
	 if (xp>=cTensor_Input.Size_X) continue;
     for(size_t dy=0;dy<downsampling_y;dy++)
	 {
      size_t yp=iy+dy;
  	  if (yp>=cTensor_Input.Size_Y) continue;
	  summ+=cTensor_Input.GetElement(z,yp,xp);
	 }
	}
	summ/=static_cast<type_t>(downsampling_x*downsampling_y);
    *output_ptr=summ;
   }
  }
 }

}

//----------------------------------------------------------------------------------------------------
///!увеличение разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPooling(CTensor<type_t> &cTensor_Output,CTensor<SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,size_t pooling_x,size_t pooling_y)
{
 if ((cTensor_Input.Size_X/pooling_x)!=cTensor_Output.Size_X || (cTensor_Input.Size_Y/pooling_y)!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::MaxPooling: Размерности тензоров не совпадают!";
 }

 size_t output_x=cTensor_Output.GetSizeX();
 size_t output_y=cTensor_Output.GetSizeY();
 size_t output_z=cTensor_Output.GetSizeZ();

 size_t input_x=cTensor_Input.GetSizeX();
 size_t input_y=cTensor_Input.GetSizeY();
 size_t input_z=cTensor_Input.GetSizeZ();

 for(size_t z=0;z<output_z;z++)
 {
  for(size_t y=0;y<output_y;y++)
  {
   for(size_t x=0;x<output_x;x++)
   {
    size_t ix=x*pooling_x;
    size_t iy=y*pooling_y;
    type_t max=cTensor_Input.GetElement(z,iy,ix);
    size_t max_x=ix;
    size_t max_y=iy;
    for(size_t py=0;py<pooling_y;py++)
    {
     for(size_t px=0;px<pooling_x;px++)
     {
      type_t e=cTensor_Input.GetElement(z,iy+py,ix+px);
      if (e>max)
      {
       max=e;
       max_x=ix+px;
       max_y=iy+py;
      }
     }
    }
    cTensor_Output.SetElement(z,y,x,max);
    SPos sPos;
    sPos.X=max_x;
    sPos.Y=max_y;
    cTensor_Position.SetElement(z,y,x,sPos);
   }
  }
 }

}

//----------------------------------------------------------------------------------------------------
///!обратный проход при увеличении разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,size_t pooling_x,size_t pooling_y)
{
 if (cTensor_Input.Size_X!=(cTensor_Output.Size_X/pooling_x) || cTensor_Input.Size_Y!=(cTensor_Output.Size_Y/pooling_y) || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::MaxPooling: Размерности тензоров не совпадают!";
 }

 size_t input_x=cTensor_Input.GetSizeX();
 size_t input_y=cTensor_Input.GetSizeY();
 size_t input_z=cTensor_Input.GetSizeZ();

 cTensor_Output.Zero();

 for(size_t z=0;z<input_z;z++)
 {
  for(size_t y=0;y<input_y;y++)
  {
   for(size_t x=0;x<input_x;x++)
   {
    type_t delta=cTensor_Input.GetElement(z,y,x);
    SPos sPos=cTensor_Position.GetElement(z,y,x);
    cTensor_Output.SetElement(z,sPos.Y,sPos.X,delta);
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
 type_t *item_ptr=cTensor.GetColumnPtr(0,0);

 size_t size_y=cTensor.GetSizeY();
 size_t size_x=cTensor.GetSizeX();
 size_t size_z=cTensor.GetSizeZ();

 for(size_t z=0;z<size_z;z++)
 {
  for(size_t y=0;y<size_y;y++)
  {
   for(size_t x=0;x<size_x;x++,item_ptr++)
   {
    if (*item_ptr<min_value) *item_ptr=min_value;
    if (*item_ptr>max_value) *item_ptr=max_value;
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//!выполнить алгоритм Adam к весовому тензору
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Adam(CTensor<type_t> &cTensor_Weight,CTensor<type_t> &cTensor_dWeight,CTensor<type_t> &cTensor_M,CTensor<type_t> &cTensor_V,double speed,double beta1,double beta2,double epsilon,double iteration)
{
 if (cTensor_Weight.Size_X!=cTensor_dWeight.Size_X || cTensor_Weight.Size_Y!=cTensor_dWeight.Size_Y || cTensor_Weight.Size_Z!=cTensor_dWeight.Size_Z ||
     cTensor_Weight.Size_X!=cTensor_M.Size_X || cTensor_Weight.Size_Y!=cTensor_M.Size_Y || cTensor_Weight.Size_Z!=cTensor_M.Size_Z ||
     cTensor_Weight.Size_X!=cTensor_V.Size_X || cTensor_Weight.Size_Y!=cTensor_V.Size_Y || cTensor_Weight.Size_Z!=cTensor_V.Size_Z)
 {
  throw "CTensor::Adam: Размерности тензоров не совпадают!";
 }

 size_t size_y=cTensor_Weight.GetSizeY();
 size_t size_x=cTensor_Weight.GetSizeX();
 size_t size_z=cTensor_Weight.GetSizeZ();

 for(size_t z=0;z<size_z;z++)
 {
  for(size_t y=0;y<size_y;y++)
  {
   for(size_t x=0;x<size_x;x++)
   {

    type_t dw=cTensor_dWeight.GetElement(z,y,x);
    type_t m=cTensor_M.GetElement(z,y,x);
    type_t v=cTensor_V.GetElement(z,y,x);

    m=beta1*m+(1.0-beta1)*dw;
    v=beta2*v+(1.0-beta2)*dw*dw;

    type_t mc=m/(1.0-pow(beta1,iteration));
    type_t vc=v/(1.0-pow(beta2,iteration));

    dw=speed*mc/(sqrt(vc)+epsilon);

    cTensor_M.SetElement(z,y,x,m);
    cTensor_V.SetElement(z,y,x,v);
    //корректируем веса
    type_t w=cTensor_Weight.GetElement(z,y,x);
    cTensor_Weight.SetElement(z,y,x,w-dw);
   }
  }
 }
}


#endif
