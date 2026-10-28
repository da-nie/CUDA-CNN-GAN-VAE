#ifndef C_TENSOR_POOLING_H
#define C_TENSOR_POOLING_H

//****************************************************************************************************
//Операции изменения размеров тензора над тензорами произвольной размерности
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
///!Операции изменения размеров тензора над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorPooling
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  struct SPos
  {
   uint32_t X;
   uint32_t Y;
  };
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
 public:
  //-конструктор----------------------------------------------------------------------------------------
  //-конструктор копирования----------------------------------------------------------------------------
  //-деструктор-----------------------------------------------------------------------------------------
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  static void MaxPooling(CTensor<type_t> &cTensor_Output,const CTensor<CTensorPooling<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y);///<уменьшение разрешения тензора выборкой большего элемента
  static void MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<CTensorPooling<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y);///<обратный проход при увеличении разрешения тензора выборкой большего элемента
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
///!увеличение разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorPooling<type_t>::MaxPooling(CTensor<type_t> &cTensor_Output,const CTensor<CTensorPooling<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y)
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
void CTensorPooling<type_t>::MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<CTensorPooling<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y)
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

#endif
