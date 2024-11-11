#ifndef C_TENSOR_APPLY_FUNC_H
#define C_TENSOR_APPLY_FUNC_H

//****************************************************************************************************
//Применение функций к тензорам произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "ctensor.h"
#include "ctensormath.h"

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
  static type_t Sigmoid(type_t v);///<сигмоид
  static type_t ReLU(type_t v);///<ReLU
  static type_t LeakyReLU(type_t v);///<Leaky ReLU
  static type_t Linear(type_t v);///<линейная
  static type_t Tangence(type_t v);///<гиперболический тангенс

  static type_t dSigmoid(type_t v);///<производная сигмоида
  static type_t dReLU(type_t v);///<производная ReLU
  static type_t dLeakyReLU(type_t v);///<производная Leaky ReLU
  static type_t dLinear(type_t v);///<производная линейной функции
  static type_t dTangence(type_t v);///<производная гиперболического тангенса

  static void ApplySigmoid(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию сигмоид
  static void ApplyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию ReLU
  static void ApplyLeakyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию Leaky ReLU
  static void ApplyLinear(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить линейную функцию
  static void ApplyTangence(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию гиперболический тангенс
   //static void ApplySoftMax(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию softmax

  static void ApplyDifferentialSigmoid(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию  производной от сигмоида
  static void ApplyDifferentialReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<применить функцию производной от ReLU
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
///!сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::Sigmoid(type_t v)
{
 if (v>30) v=29.99999;
 if (v<-30) v=-29.99999;

 return(1.0/(1.0+exp(-v)));
}
//----------------------------------------------------------------------------------------------------
///!ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::ReLU(type_t v)
{
 if (v>0) return(v);
 return(0);
}
//----------------------------------------------------------------------------------------------------
///!Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::LeakyReLU(type_t v)
{
 if (v>0) return(v);
 return(0.1*v);
}
//----------------------------------------------------------------------------------------------------
///!линейная
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::Linear(type_t v)
{
 return(v);
}
//----------------------------------------------------------------------------------------------------
///!гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::Tangence(type_t v)
{
 if (v>30) v=29.99999;
 if (v<-30) v=-29.99999;

 //if (v>20) return(1);
 //if (v<-20) return(-1);

 type_t ep=exp(2*v);
 type_t en=exp(-2*v);
 return((ep-en)/(ep+en));
}
//----------------------------------------------------------------------------------------------------
///!производная сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::dSigmoid(type_t v)
{
 type_t s=Sigmoid(v);
 return((1.0-s)*s);
}
//----------------------------------------------------------------------------------------------------
///!производная ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::dReLU(type_t v)
{
 if (v>=0) return(1);
 return(0);
}
//----------------------------------------------------------------------------------------------------
///!производная Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::dLeakyReLU(type_t v)
{
 if (v>=0) return(1);
 return(0.1);
}
//----------------------------------------------------------------------------------------------------
///!производная линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::dLinear(type_t v)
{
 return(1);
}
//----------------------------------------------------------------------------------------------------
///!производная гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensorApplyFunc<type_t>::dTangence(type_t v)
{
 type_t t=Tangence(v);
 return(1-t*t);
}

//----------------------------------------------------------------------------------------------------
///!применить функцию сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplySigmoid(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=Sigmoid(*input_ptr);
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
///!применить функцию ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=ReLU(*input_ptr);
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
///!применить функцию Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyLeakyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=LeakyReLU(*input_ptr);
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
///!применить линейную функцию
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyLinear(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=Linear(*input_ptr);
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
///!применить функцию гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyTangence(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=Tangence(*input_ptr);
   }
  }
 }
}
/*
//----------------------------------------------------------------------------------------------------
///!применить функцию softmax
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplySoftMax(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=SoftMax(*input_ptr);
   }
  }
 }
}
*/

//----------------------------------------------------------------------------------------------------
///!применить функцию производной от сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyDifferentialSigmoid(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=dSigmoid(*input_ptr);
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyDifferentialReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=dReLU(*input_ptr);
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyDifferentialLeakyReLU(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=dLeakyReLU(*input_ptr);
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
///!применить производную линейной функций
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyDifferentialLinear(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=dLinear(*input_ptr);
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyDifferentialTangence(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=dTangence(*input_ptr);
   }
  }
 }
}
/*
//----------------------------------------------------------------------------------------------------
///!применить функцию производной от softmax
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorApplyFunc<type_t>::ApplyDifferentialSoftMax(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y)
 {
  throw "CTensorApplyFunc: Размерности тензоров не совпадают!";
 }

 const type_t *input_ptr=&cTensor_Input.Item[0];
 type_t *o_ptr=&cTensor_Output.Item[0];

 for(size_t z=0;z<cTensor_Input.Size_Z;z++)
 {
  for(size_t y=0;y<cTensor_Input.Size_Y;y++)
  {
   for(size_t x=0;x<cTensor_Input.Size_X;x++,o_ptr++,input_ptr++)
   {
    *o_ptr=dSoftMax(*input_ptr);
   }
  }
 }
}
*/

#endif
