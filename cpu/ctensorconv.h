#ifndef C_TENSOR_CONV_H
#define C_TENSOR_CONV_H

//****************************************************************************************************
//Операции свёртки над тензорами произвольной размерности
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
///!Операции свёртки над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorConv
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
  static void ForwardConvolution(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Kernel,int32_t kernel_x,int32_t kernel_y,int32_t kernel_z,size_t kernel_amount,const CTensor<type_t> &cTensor_Bias,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y);///<прямая свёртка
  static void BackwardConvolution(CTensor<type_t> &cTensor_OutputDelta,const CTensor<type_t> &cTensor_Delta,const CTensor<type_t> &cTensor_Kernel,int32_t kernel_x,int32_t kernel_y,int32_t kernel_z,size_t kernel_amount,const CTensor<type_t> &cTensor_Bias,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y);///<обратная свёртка
  static void CreateDeltaWeightAndBias(CTensor<type_t> &cTensor_dKernel,int32_t dkernel_x,int32_t dkernel_y,int32_t dkernel_z,size_t dkernel_amount,CTensor<type_t> &cTensor_dBias,const CTensor<type_t> &cTensor_Image,CTensor<type_t> &cTensor_Delta,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y);///<вычисление поправок весов и смещений
  static void CreateBackDeltaWeightAndBias(CTensor<type_t> &cTensor_dKernel,int32_t dkernel_x,int32_t dkernel_y,int32_t dkernel_z,size_t dkernel_amount,CTensor<type_t> &cTensor_dBias,CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y);///<создание поправок весов и смещений для обратной свёртки
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
/*!прямая свёртка
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::ForwardConvolution(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Kernel,int32_t kernel_x,int32_t kernel_y,int32_t kernel_z,size_t kernel_amount,const CTensor<type_t> &cTensor_Bias,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y)
{
 //вычисляем размеры выходного тензора
 int32_t output_z=kernel_amount;
 if (output_z==0) throw("Для прямой свёртки требуется хотя бы одно ядро свёртки");
 if (output_z!=cTensor_Bias.GetSizeZ()) throw("Для прямой свёртки требуется чтобы количество ядер и смещений совпадало");

 int32_t input_y=cTensor_Image.Size_Y;
 int32_t input_x=cTensor_Image.Size_X;
 int32_t input_z=cTensor_Image.Size_Z;

 int32_t output_x=(input_x-kernel_x+2*padding_x)/stride_x+1;
 int32_t output_y=(input_y-kernel_y+2*padding_y)/stride_y+1;

 if (cTensor_Output.Size_X!=output_x || cTensor_Output.Size_Y!=output_y || cTensor_Output.Size_Z!=output_z) throw("Ошибочная размерность выходного тензора для свёртки");

 if (input_z!=kernel_z) throw("Для прямой свёртки требуется чтобы глубина фильтров и входного тензора совпадали");

 //выполняем свёртку

 //странно, но работает быстро
 //для каждого фильтра
 for(size_t k=0;k<output_z;k++)
 {
  for(int32_t y=0;y<output_y;y++)
  {
   for(int32_t x=0;x<output_x;x++)
   {
    type_t sum=cTensor_Bias.GetElement(k,0,0);//сразу прибавляем смещение
    //применяем фильтр
    for(int32_t ky=0;ky<kernel_y;ky++)
    {
     int32_t y0=stride_y*y+ky-padding_y;
     if (y0<0 || y0>=input_y) continue;
     for(int32_t kx=0;kx<kernel_x;kx++)
     {
      int32_t x0=stride_x*x+kx-padding_x;
      //игнорируем элементы вне границ входного тензора
      if (x0<0 || x0>=input_x) continue;
      //проходимся по всей глубине тензора и считаем сумму
      for(int32_t z=0;z<input_z;z++)
      {
       type_t kernel=cTensor_Kernel.GetElement(0,k,z*kernel_x*kernel_y+ky*kernel_x+kx);
       type_t image=cTensor_Image.GetElement(z,y0,x0);
       sum+=kernel*image;
      }
     }
    }
    cTensor_Output.SetElement(k,y,x,sum);
   }
  }
 }

 /*
 //для каждого ядра
 for(size_t k=0;k<output_z;k++)
 {
  for(int32_t y=0;y<output_y;y++)
  {
   for(int32_t x=0;x<output_x;x++)
   {
    type_t sum=bias[k];//прибавляем смещение
    const type_t *input_ptr=cTensor_Image.GetColumnPtr(0,0);
    const type_t *kernel_ptr=cTensor_Kernel[k].GetColumnPtr(0,0);
    size_t offset_kernel_ptr=(y*output_x+x);
    type_t *output_ptr=cTensor_Output.GetColumnPtr(k,0)+offset_kernel_ptr;

    for(size_t ky=0;ky<kernel_y;ky++)
    {
     int32_t y0=static_cast<int32_t>(stride_y*y+ky);
     y0-=static_cast<int32_t>(padding_y);
     if (y0<0 || y0>=input_y) continue;
     for(size_t kx=0;kx<kernel_x;kx++)
     {
      int32_t x0=static_cast<int32_t>(stride_x*x+kx);
      x0-=static_cast<int32_t>(padding_x);
      if (x0<0 || x0>=input_x) continue;
      size_t offset_i_ptr=y0*input_x+x0;
      size_t offset_d_ptr=ky*kernel_x+kx;
      for(size_t d=0;d<input_z;d++)
      {
       const type_t *i_ptr=input_ptr+d*input_x*input_y+offset_i_ptr;
       const type_t *k_ptr=kernel_ptr+d*kernel_x*kernel_y+offset_d_ptr;
	   sum+=(*i_ptr)*(*k_ptr);
      }
     }
    }
    *output_ptr=sum;//записываем результат свёртки в выходной тензор
   }
  }
 }*/
}


//----------------------------------------------------------------------------------------------------
/*!обратная свёртка
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::BackwardConvolution(CTensor<type_t> &cTensor_OutputDelta,const CTensor<type_t> &cTensor_Delta,const CTensor<type_t> &cTensor_Kernel,int32_t kernel_x,int32_t kernel_y,int32_t kernel_z,size_t kernel_amount,const CTensor<type_t> &cTensor_Bias,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y)
{
 //вычисляем размеры выходного тензора
 if (kernel_amount==0) throw("Для обратной свёртки требуется хотя бы одно ядро свёртки");
 if (kernel_amount!=cTensor_Bias.GetSizeZ()) throw("Для обратной свёртки требуется чтобы количество ядер и смещений совпадало");

 int32_t input_y=cTensor_Delta.Size_Y;
 int32_t input_x=cTensor_Delta.Size_X;
 int32_t input_z=cTensor_Delta.Size_Z;

 //обратная свёртка делается с ядрами, повёрнутыми на 180
 padding_x=kernel_x-1-padding_x;
 padding_y=kernel_y-1-padding_y;

 int32_t output_x=cTensor_OutputDelta.Size_X;
 int32_t output_y=cTensor_OutputDelta.Size_Y;

 int32_t output_z=kernel_z;

 if (cTensor_OutputDelta.Size_X!=output_x || cTensor_OutputDelta.Size_Y!=output_y || cTensor_OutputDelta.Size_Z!=output_z) throw("Ошибочная размерность выходного тензора для обратной свёртки");
 if (input_z!=kernel_amount) throw("Для обратной свёртки требуется чтобы количество фильтров и глубина входного тензора совпадали");

 //странно, но эта нативная реализация работает быстрее
 for(int32_t y=0;y<output_y;y++)
 {
  for(int32_t x=0;x<output_x;x++)
  {
   for(int32_t z=0;z<output_z;z++)
   {
    type_t summ=0;
    //идём по всем весовым коэффициентам фильтров
    for(int32_t ky=0;ky<kernel_y;ky++)
    {
     int32_t y0=y+ky-padding_y;
     if (y0%stride_y!=0) continue;//delta=0
     int32_t delta_y0=y0/stride_y;
     if (delta_y0<0 || delta_y0>=input_y) continue;
     for(int32_t kx=0;kx<kernel_x;kx++)
     {
      int32_t x0=x+kx-padding_x;
      //игнорируем выходящие за границы элементы
	  if (x0%stride_x!=0) continue;//delta=0
	  int32_t delta_x0=x0/stride_x;
	  if (delta_x0<0 || delta_x0>=input_x) continue;
      //суммируем по всем фильтрам
      for(int32_t f=0;f<kernel_amount;f++)
      {
       summ+=cTensor_Bias.GetElement(f,0,0);//TODO: надо выяснить, как прибавлять смещения
       type_t k=cTensor_Kernel.GetElement(0,f,z*kernel_x*kernel_y+(kernel_y-1-ky)*kernel_x+(kernel_x-1-kx));
       type_t d=cTensor_Delta.GetElement(f,delta_y0,delta_x0);
       summ+=k*d;
      }
     }
    }
    cTensor_OutputDelta.SetElement(z,y,x,summ);
   }
  }
 }

 /*
 for(int32_t y=0;y<output_y;y++)
 {
  for(int32_t x=0;x<output_x;x++)
  {
   for(int32_t z=0;z<output_z;z++)
   {
    const type_t *delta_ptr=cTensor_Delta.GetColumnPtr(0,0);
    type_t *output_ptr=cTensor_OutputDelta.GetColumnPtr(0,0)+z*output_x*output_y+y*output_x+x;
    size_t kernel_depth_offset=z*kernel_x*kernel_y;
    type_t sum=0;//сумма для градиента
    //идём по всем весовым коэффициентам фильтров
    for(size_t ky=0;ky<kernel_y;ky++)
    {
     int32_t y0=static_cast<int32_t>(y+ky);
     y0-=static_cast<int32_t>(padding_y);
     if (y0%stride_y!=0) continue;//delta=0
     int32_t delta_y0=y0/stride_y;
     if (delta_y0<0 || delta_y0>=input_y) continue;
     for(size_t kx=0;kx<kernel_x;kx++)
     {
      int32_t x0=static_cast<int32_t>(x+kx);
      x0-=static_cast<int32_t>(padding_x);
      //игнорируем выходящие за границы элементы
	  if (x0%stride_x!=0) continue;//delta=0
	  int32_t delta_x0=x0/stride_x;
	  if (delta_x0<0 || delta_x0>=input_x) continue;
      //суммируем по всем ядрам
      size_t offset_k_ptr=(kernel_y-1-ky)*kernel_x+(kernel_x-1-kx)+kernel_depth_offset;
      size_t offset_d_ptr=delta_y0*input_x+delta_x0;
      for(size_t k=0;k<kernel_amount;k++)
      {
       sum+=bias[k];//TODO: надо выяснить, как прибавлять смещения
       const type_t *d_ptr=delta_ptr+k*input_x*input_y+offset_d_ptr;
	   const type_t *kernel_ptr=cTensor_Kernel[k].GetColumnPtr(0,0);
       const type_t *k_ptr=kernel_ptr+offset_k_ptr;
       sum+=(*k_ptr)*(*d_ptr);//добавляем произведение повёрнутых фильтров на дельты
      }
     }
    }
    *output_ptr=sum;//записываем результат в тензор градиента
   }
  }
 }*/
}

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::CreateDeltaWeightAndBias(CTensor<type_t> &cTensor_dKernel,int32_t dkernel_x,int32_t dkernel_y,int32_t dkernel_z,size_t dkernel_amount,CTensor<type_t> &cTensor_dBias,const CTensor<type_t> &cTensor_Image,CTensor<type_t> &cTensor_Delta,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y)
{
 if (dkernel_amount==0) throw("Для создания поправок весов и смещений требуется не пустой вектор поправок к ядрам");
 if (cTensor_dBias.GetSizeZ()!=dkernel_amount) throw("Для создания поправок весов и смещений требуется чтобы количество поправок фильтров и поправок сдвигов совпадало");

 int32_t image_x=cTensor_Image.Size_X;
 int32_t image_y=cTensor_Image.Size_Y;
 int32_t image_z=cTensor_Image.Size_Z;

 int32_t delta_x=cTensor_Delta.Size_X;
 int32_t delta_y=cTensor_Delta.Size_Y;
 int32_t delta_z=cTensor_Delta.Size_Z;

 if (dkernel_z!=image_z) throw("Неверные размеры тензора поправок к ядрам для обновления весов и смещений");
 if (delta_z!=dkernel_amount) throw("Для создания поправок весов и смещений требуется чтобы глубина тензора дельт совпадала с количеством ядер");

 //новая дельта
 int32_t new_delta_x=stride_x*(delta_x-1)+1;
 int32_t new_delta_y=stride_y*(delta_y-1)+1;

 for(int32_t f=0;f<dkernel_amount;f++)
 {
  for(int32_t y=0;y<new_delta_y;y++)
  {
   if (y%stride_y!=0) continue;
   for(int32_t x=0;x<new_delta_x;x++)
   {
	if (x%stride_x!=0) continue;
    type_t delta=cTensor_Delta.GetElement(f,y/stride_y,x/stride_x);//запоминаем значение градиента
    for(int32_t i=0;i<dkernel_y;i++)
    {
     int32_t i0=i+y-padding_y;
	 if (i0<0 || i0>=image_y) continue;
     for(int32_t j=0;j<dkernel_x;j++)
     {
      int32_t j0=j+x-padding_x;
      //игнорируем выходящие за границы элементы
      if (j0<0 || j0>=image_x) continue;
      //наращиваем градиент фильтра
      for(int32_t c=0;c<dkernel_z;c++)
      {
       type_t dk=cTensor_dKernel.GetElement(0,f,c*dkernel_x*dkernel_y+i*dkernel_x+j);
       dk+=delta*cTensor_Image.GetElement(c,i0,j0);
       cTensor_dKernel.SetElement(0,f,c*dkernel_x*dkernel_y+i*dkernel_x+j,dk);
      }
     }
    }
	type_t b=cTensor_dBias.GetElement(f,0,0);
	b+=delta;
    cTensor_dBias.SetElement(f,0,0,b);
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений для обратной свёртки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::CreateBackDeltaWeightAndBias(CTensor<type_t> &cTensor_dKernel,int32_t dkernel_x,int32_t dkernel_y,int32_t dkernel_z,size_t dkernel_amount,CTensor<type_t> &cTensor_dBias,CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y)
{
 CTensorConv<type_t>::CreateDeltaWeightAndBias(cTensor_dKernel,dkernel_x,dkernel_y,dkernel_z,dkernel_amount,cTensor_dBias,cTensor_Delta,cTensor_Image,stride_x,stride_y,padding_x,padding_y);
 CTensorMath<type_t>::Fill(cTensor_dBias,0);//TODO: неясно, нужно ли использовать эти поправки
}

#endif
