#ifndef C_TENSOR_CLIP_H
#define C_TENSOR_CLIP_H

//****************************************************************************************************
//Операции ограничения над тензорами произвольной размерности
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
///!Операции ограничения над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorClip
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
  static void Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value);///<выполнить отсечку значений тензора
  static void ClipByNormXY(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold);///<ограничить тензор по норме XY
  static void ClipByNormX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold);///<ограничить тензор по норме X
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
///!выполнить отсечку значений тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorClip<type_t>::Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value)
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
//!ограничить тензор по норме XY
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorClip<type_t>::ClipByNormXY(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold)
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
void CTensorClip<type_t>::ClipByNormX(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t threshold)
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


#endif
