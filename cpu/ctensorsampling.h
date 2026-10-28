#ifndef C_TENSOR_SAMPLING_H
#define C_TENSOR_SAMPLING_H

//****************************************************************************************************
//Операции увеличения/уменьшения размерности над тензорами произвольной размерности
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
///!Операции увеличения/уменьшения размерности над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorSampling
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
  static void UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y);///<увеличение разрешения тензора
  static void DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y);///<уменьшение разрешения тензора
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
///!увеличение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorSampling<type_t>::UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y)
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
void CTensorSampling<type_t>::DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y)
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

#endif
