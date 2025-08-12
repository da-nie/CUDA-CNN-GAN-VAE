#ifndef C_TENSOR_CONV_CU_H
#define C_TENSOR_CONV_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensorconv.h"

#endif


#ifndef USE_CPU


//****************************************************************************************************
//Операции свёртки над тензорами произвольной размерности
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

//****************************************************************************************************
///!структура ядра тензора входного изображения для прямой свёртки
//****************************************************************************************************
template<class type_t>
struct STensorKernel_ForwardConvolution_Image
{
 STensorKernel<type_t> sTensorKernel_Image;///<исходный тензор

 int32_t Conv_Kernel_X;
 int32_t Conv_Kernel_Y;
 int32_t Conv_Stride_X;
 int32_t Conv_Stride_Y;
 int32_t Conv_Padding_X;
 int32_t Conv_Padding_Y;
 int32_t Size_X;
 int32_t Size_Y;
 int32_t Size_Z;
 int32_t Stride_X;
 int32_t Basic_Size_X;
 int32_t Basic_Size_Y;
 int32_t Offset_X;
 int32_t Offset_Y;

 int32_t Dst_X;
 int32_t Dst_Y;

 __host__ __device__ STensorKernel_ForwardConvolution_Image()///<конструктор
 {
 }
 __host__ __device__ STensorKernel_ForwardConvolution_Image(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)///<конструктор
 {
  Set(cTensor,input_y,input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);
 }

 __forceinline__ __host__ __device__ type_t* GetTensorDataPtr(size_t z)///<получить указатель на элементы с глубиной z
 {
  return(NULL);
 }

 __forceinline__ __host__ __device__ void SelectZ(size_t z)///<выбрать слой Z
 {
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t z,size_t y,size_t x)
 {
  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t tmp=pos;
  tmp/=Dst_X;
  int32_t dx=pos-tmp*Dst_X;
  pos=tmp;
  tmp/=Dst_Y;
  int32_t dy=pos-tmp*Dst_Y;
  pos=tmp;
  tmp/=Conv_Kernel_X;
  int32_t kx=pos-tmp*Conv_Kernel_X;
  pos=tmp;
  tmp/=Conv_Kernel_Y;
  int32_t ky=pos-tmp*Conv_Kernel_Y;
  int32_t sz=tmp;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  return(sTensorKernel_Image.GetElement(sz,sy,sx));
 }

 __forceinline__ __host__ __device__ void SetElement(size_t z,size_t y,size_t x,type_t value)
 {
  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t tmp=pos;
  tmp/=Dst_X;
  int32_t dx=pos-tmp*Dst_X;
  pos=tmp;
  tmp/=Dst_Y;
  int32_t dy=pos-tmp*Dst_Y;
  pos=tmp;
  tmp/=Conv_Kernel_X;
  int32_t kx=pos-tmp*Conv_Kernel_X;
  pos=tmp;
  tmp/=Conv_Kernel_Y;
  int32_t ky=pos-tmp*Conv_Kernel_Y;
  int32_t sz=tmp;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  sTensorKernel_Image.SetElement(sz,sy,sx,value);
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t y,size_t x)
 {
  return(GetElement(0,y,x));
 }
 __forceinline__ __host__ __device__ void SetElement(size_t y,size_t x,type_t value)
 {
  SetElement(0,y,x,value);
 }


 __forceinline__ __host__ __device__ size_t GetSizeX(void) const
 {
  return(Size_X);
 }

 __forceinline__ __host__ __device__ size_t GetSizeY(void) const
 {
  return(Size_Y);
 }

 __forceinline__ __host__ __device__ size_t GetSizeZ(void) const
 {
  return(Size_Z);
 }

 __host__ __device__ void Reset(void)
 {
  sTensorKernel_Image.Reset();
  Conv_Kernel_X=0;
  Conv_Kernel_Y=0;
  Conv_Stride_X=0;
  Conv_Stride_Y=0;
  Conv_Padding_X=0;
  Conv_Padding_Y=0;
  Size_X=0;
  Size_Y=0;
  Size_Z=0;
  Basic_Size_X=0;
  Basic_Size_Y=0;
  Offset_X=0;
  Offset_Y=0;
  Dst_X=0;
  Dst_Y=0;
 }

 __host__ __device__ void Set(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)
 {
  sTensorKernel_Image.Set(cTensor);

  Conv_Kernel_X=kernel_x;
  Conv_Kernel_Y=kernel_y;
  Conv_Stride_X=stride_x;
  Conv_Stride_Y=stride_y;
  Conv_Padding_X=padding_x;
  Conv_Padding_Y=padding_y;
  Size_X=input_x;
  Size_Y=input_y;
  Size_Z=1;
  Basic_Size_X=Size_X;
  Basic_Size_Y=Size_Y;
  Offset_X=0;
  Offset_Y=0;
  Dst_X=dst_x;
  Dst_Y=dst_y;
 }
};

//----------------------------------------------------------------------------------------------------
/*!прямая свёртка через умножение матриц
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

 //перестроим входной тензор для выполнения умножения
 int32_t dst_y=output_y;
 int32_t dst_x=output_x;

 //int32_t new_input_z=1;
 int32_t new_input_y=kernel_y*kernel_x*kernel_z;
 int32_t new_input_x=dst_x*dst_y;

 /*
 //перестроим тензоры ядер в строку
 CTensor<type_t> cTensor_NewKernel(1,output_z,kernel_x*kernel_y*kernel_z);

 for(size_t k=0;k<cTensor_Kernel.size();k++)
 {
  cTensor_Kernel[k].CopyFromDevice();
  const type_t *s_ptr=cTensor_Kernel[k].GetColumnPtr(0,0);
  type_t *d_ptr=cTensor_NewKernel.GetColumnPtr(0,k);
  for(size_t n=0;n<kernel_x*kernel_y*kernel_z;n++,s_ptr++,d_ptr++) *d_ptr=*s_ptr;
 }
 cTensor_NewKernel.SetHostOnChange();

 //перестроим смещения
 CTensor<type_t> cTensor_Bias(bias.size(),1,1);
 for(size_t b=0;b<bias.size();b++)
 {
  cTensor_Bias.SetElement(b,0,0,bias[b]);
 }
 */

 //умножаем матрицы
 cTensor_Output.ReinterpretSize(1,output_z,new_input_x);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Kernel(cTensor_Kernel);
 STensorKernel_ForwardConvolution_Image<type_t> sTensorKernel_Image(cTensor_Image,new_input_y,new_input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);

 CTensorMath<type_t>::MulAbstract(cTensor_Output,sTensorKernel_Output,cTensor_Kernel,sTensorKernel_Kernel,cTensor_Image,sTensorKernel_Image);
 cTensor_Output.ReinterpretSize(output_z,output_y,output_x);

 CTensorMath<type_t>::AddBias(cTensor_Output,cTensor_Bias);
}


//****************************************************************************************************
///!структура ядра тензора для свёрточных ядер для обратной  свёртки
//****************************************************************************************************
template<class type_t>
struct STensorKernel_BackwardConvolution_Kernel
{
 STensorKernel<type_t> sTensorKernel_Kernel;///<исходный тензорор ядра

 size_t Size_X;///<размер по x
 size_t Size_Y;///<размер по y
 size_t Size_Z;///<размер по z
 int32_t Kernel_X;
 int32_t Kernel_Y;
 int32_t Kernel_Z;
 int32_t Offset_X;
 int32_t Offset_Y;
 int32_t Kernel_Amount;

 __host__ __device__ STensorKernel_BackwardConvolution_Kernel(void)///<конструктор
 {
 }
 __host__ __device__ STensorKernel_BackwardConvolution_Kernel(const CTensor<type_t> &cTensor_Kernel,size_t kernel_x,size_t kernel_y,size_t kernel_z,size_t kernel_amount)///<конструктор
 {
  Set(cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount);
 }

 __forceinline__ __host__ __device__ type_t* GetTensorDataPtr(size_t z)///<получить указатель на элементы с глубиной z
 {
  return(NULL);
 }

 __forceinline__ __host__ __device__ void SelectZ(size_t z)///<выбрать слой Z
 {
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t z,size_t y,size_t x)
 {
  int32_t sx=x+Offset_X;
  int32_t sy=y+Offset_Y;

  if (sy<0 || sy>=Kernel_Z) return(0);
  if (sx<0 || sx>=Kernel_X*Kernel_Y*Kernel_Amount) return(0);
  int32_t ki=sx/(Kernel_X*Kernel_Y);
  sx-=ki*(Kernel_X*Kernel_Y);
  int32_t ky=sx/Kernel_X;
  sx-=ky*Kernel_X;
  int32_t kx=sx;
  int32_t kz=sy;

  ky=Kernel_Y-1-ky;
  kx=Kernel_X-1-kx;

  return(sTensorKernel_Kernel.GetElement(ki,kx+ky*Kernel_X+kz*Kernel_X*Kernel_Y));
 }

 __forceinline__ __host__ __device__ void SetElement(size_t z,size_t y,size_t x,type_t value)
 {
  int32_t sx=x+Offset_X;
  int32_t sy=y+Offset_Y;

  if (sy<0 || sy>=Kernel_Amount) return;
  if (sx<0 || sx>=Kernel_X*Kernel_Y*Kernel_Z) return;
  int32_t kz=sx/(Kernel_X*Kernel_Y);
  sx-=kz*(Kernel_X*Kernel_Y);
  int32_t ky=sx/Kernel_X;
  sx-=ky*Kernel_X;
  int32_t kx=sx;

  ky=Kernel_Y-1-ky;
  kx=Kernel_X-1-kx;

  sTensorKernel_Kernel.SetElement(sy,kx+ky*Kernel_X+kz*Kernel_X*Kernel_Y,value);
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t y,size_t x)
 {
  return(GetElement(0,y,x));
 }
 __forceinline__ __host__ __device__ void SetElement(size_t y,size_t x,type_t value)
 {
  SetElement(0,y,x,value);
 }

 __forceinline__ __host__ __device__ size_t GetSizeX(void) const
 {
  return(Size_X);
 }

 __forceinline__ __host__ __device__ size_t GetSizeY(void) const
 {
  return(Size_Y);
 }

 __forceinline__ __host__ __device__ size_t GetSizeZ(void) const
 {
  return(Size_Z);
 }

 __host__ __device__ void Reset(void)
 {
  Size_X=0;
  Size_Y=0;
  Size_Z=0;
  Kernel_X=0;
  Kernel_Y=0;
  Kernel_Z=0;
  Offset_X=0;
  Offset_Y=0;
  Kernel_Amount=0;
 }

 __host__ __device__ void Set(const CTensor<type_t> &cTensor_Kernel,size_t kernel_x,size_t kernel_y,size_t kernel_z,size_t kernel_amount)
 {
  sTensorKernel_Kernel.Set(cTensor_Kernel);
  sTensorKernel_Kernel.SelectZ(0);

  Kernel_X=kernel_x;
  Kernel_Y=kernel_y;
  Kernel_Z=kernel_z;

  Size_Z=1;
  Size_Y=kernel_z;
  Size_X=kernel_x*kernel_y*kernel_amount;

  Offset_X=0;
  Offset_Y=0;

  Kernel_Amount=kernel_amount;
 }
};



//****************************************************************************************************
///!структура ядра тензора входного изображения для обратной свёртки
//****************************************************************************************************
template<class type_t>
struct STensorKernel_BackwardConvolution_Delta
{
 STensorKernel<type_t> sTensorKernel_Delta;///<исходный тензор

 int32_t Conv_Kernel_X;
 int32_t Conv_Kernel_Y;
 int32_t Conv_Stride_X;
 int32_t Conv_Stride_Y;
 int32_t Conv_Padding_X;
 int32_t Conv_Padding_Y;
 int32_t Size_X;
 int32_t Size_Y;
 int32_t Size_Z;
 int32_t Stride_X;
 int32_t Basic_Size_X;
 int32_t Basic_Size_Y;
 int32_t Offset_X;
 int32_t Offset_Y;

 int32_t Dst_X;
 int32_t Dst_Y;

 __host__ __device__ STensorKernel_BackwardConvolution_Delta()///<конструктор
 {
 }
 __host__ __device__ STensorKernel_BackwardConvolution_Delta(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)///<конструктор
 {
  Set(cTensor,input_y,input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);
 }

 __forceinline__ __host__ __device__ type_t* GetTensorDataPtr(size_t z)///<получить указатель на элементы с глубиной z
 {
  return(NULL);
 }

 __forceinline__ __host__ __device__ void SelectZ(size_t z)///<выбрать слой Z
 {
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t z,size_t y,size_t x)
 {
  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t tmp=pos;
  tmp/=Dst_X;
  int32_t dx=pos-tmp*Dst_X;
  pos=tmp;
  tmp/=Dst_Y;
  int32_t dy=pos-tmp*Dst_Y;
  pos=tmp;
  tmp/=Conv_Kernel_X;
  int32_t kx=pos-tmp*Conv_Kernel_X;
  pos=tmp;
  tmp/=Conv_Kernel_Y;
  int32_t ky=pos-tmp*Conv_Kernel_Y;
  int32_t sz=tmp;

  int32_t sy=dy+ky-Conv_Padding_Y;
  int32_t sx=dx+kx-Conv_Padding_X;

  if (sx<0 || sy<0 || sz<0) return(0);

  int32_t sy_s=sy/Conv_Stride_Y;
  if (sy!=sy_s*Conv_Stride_Y) return(0);

  int32_t sx_s=sx/Conv_Stride_X;
  if (sx!=sx_s*Conv_Stride_X) return(0);

  return(sTensorKernel_Delta.GetElement(sz,sy_s,sx_s));
 }
 __forceinline__ __host__ __device__ void SetElement(size_t z,size_t y,size_t x,type_t value)
 {
  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t tmp=pos;
  tmp/=Dst_X;
  int32_t dx=pos-tmp*Dst_X;
  pos=tmp;
  tmp/=Dst_Y;
  int32_t dy=pos-tmp*Dst_Y;
  pos=tmp;
  tmp/=Conv_Kernel_X;
  int32_t kx=pos-tmp*Conv_Kernel_X;
  pos=tmp;
  tmp/=Conv_Kernel_Y;
  int32_t ky=pos-tmp*Conv_Kernel_Y;
  int32_t sz=tmp;

  int32_t sy=dy+ky-Conv_Padding_Y;
  int32_t sx=dx+kx-Conv_Padding_X;

  if (sx<0 || sy<0 || sz<0) return;

  int32_t sy_s=sy/Conv_Stride_Y;
  int32_t sx_s=sx/Conv_Stride_X;

  if ((sy!=sy_s*Conv_Stride_Y) || (sx!=sx_s*Conv_Stride_X)) value=0;

  sTensorKernel_Delta.SetElement(sz,sy_s,sx_s,value);
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t y,size_t x)
 {
  return(GetElement(0,y,x));
 }
 __forceinline__ __host__ __device__ void SetElement(size_t y,size_t x,type_t value)
 {
  SetElement(0,y,x,value);
 }

 __forceinline__ __host__ __device__ size_t GetSizeX(void) const
 {
  return(Size_X);
 }

 __forceinline__ __host__ __device__ size_t GetSizeY(void) const
 {
  return(Size_Y);
 }

 __forceinline__ __host__ __device__ size_t GetSizeZ(void) const
 {
  return(Size_Z);
 }

 __host__ __device__ void Reset(void)
 {
  sTensorKernel_Delta.Reset();
  Conv_Kernel_X=0;
  Conv_Kernel_Y=0;
  Conv_Stride_X=0;
  Conv_Stride_Y=0;
  Conv_Padding_X=0;
  Conv_Padding_Y=0;
  Size_X=0;
  Size_Y=0;
  Size_Z=0;
  Basic_Size_X=0;
  Basic_Size_Y=0;
  Offset_X=0;
  Offset_Y=0;
  Dst_X=0;
  Dst_Y=0;
 }

 __host__ __device__ void Set(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)
 {
  sTensorKernel_Delta.Set(cTensor);

  Conv_Kernel_X=kernel_x;
  Conv_Kernel_Y=kernel_y;
  Conv_Stride_X=stride_x;
  Conv_Stride_Y=stride_y;
  Conv_Padding_X=padding_x;
  Conv_Padding_Y=padding_y;
  Size_X=input_x;
  Size_Y=input_y;
  Size_Z=1;
  Basic_Size_X=Size_X;
  Basic_Size_Y=Size_Y;
  Offset_X=0;
  Offset_Y=0;
  Dst_X=dst_x;
  Dst_Y=dst_y;
 }
};

//----------------------------------------------------------------------------------------------------
/*!обратная свёртка через умножение матриц
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

 //перестроим входной тензор для выполнения умножения
 int32_t dst_y=output_y;
 int32_t dst_x=output_x;

 //int32_t new_input_z=1;
 int32_t new_input_y=kernel_y*kernel_x*input_z;
 int32_t new_input_x=dst_x*dst_y;

 //умножаем матрицы
 cTensor_OutputDelta.ReinterpretSize(1,output_z,new_input_x);

 STensorKernel<type_t> sTensorKernel_OutputDelta(cTensor_OutputDelta);
 STensorKernel_BackwardConvolution_Kernel<type_t> sTensorKernel_Kernel(cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount);
 STensorKernel_BackwardConvolution_Delta<type_t> sTensorKernel_Delta(cTensor_Delta,new_input_y,new_input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);

 CTensorMath<type_t>::MulAbstract(cTensor_OutputDelta,sTensorKernel_OutputDelta,cTensor_Kernel,sTensorKernel_Kernel,cTensor_Delta,sTensorKernel_Delta);

 cTensor_OutputDelta.RestoreSize();
 cTensor_OutputDelta.SetDeviceOnChange();
}




//****************************************************************************************************
///!структура ядра тензора входного изображения для вычисления поправок
//****************************************************************************************************
template<class type_t>
struct STensorKernel_DeltaWeightAndBias_Image
{
 STensorKernel<type_t> sTensorKernel_Image;///<исходный тензор

 int32_t Conv_Kernel_X;
 int32_t Conv_Kernel_Y;
 int32_t Conv_Stride_X;
 int32_t Conv_Stride_Y;
 int32_t Conv_Padding_X;
 int32_t Conv_Padding_Y;
 int32_t Size_X;
 int32_t Size_Y;
 int32_t Size_Z;
 int32_t Stride_X;
 int32_t Basic_Size_X;
 int32_t Basic_Size_Y;
 int32_t Offset_X;
 int32_t Offset_Y;

 int32_t Dst_X;
 int32_t Dst_Y;
 int32_t Input_Z;

 __host__ __device__ STensorKernel_DeltaWeightAndBias_Image()///<конструктор
 {
 }
 __host__ __device__ STensorKernel_DeltaWeightAndBias_Image(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)///<конструктор
 {
  Set(cTensor,input_y,input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);
 }

 __forceinline__ __host__ __device__ type_t* GetTensorDataPtr(size_t z)///<получить указатель на элементы с глубиной z
 {
  return(NULL);
 }

 __forceinline__ __host__ __device__ void SelectZ(size_t z)///<выбрать слой Z
 {
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t z,size_t y,size_t x)
 {
  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t tmp=pos;
  tmp/=Dst_X;
  int32_t dx=pos-tmp*Dst_X;
  pos=tmp;
  tmp/=Dst_Y;
  int32_t dy=pos-tmp*Dst_Y;
  pos=tmp;
  tmp/=Input_Z;
  int32_t sz=pos-tmp*Input_Z;
  pos=tmp;
  tmp/=Conv_Kernel_X;
  int32_t kx=pos-tmp*Conv_Kernel_X;
  int32_t ky=tmp;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  return(sTensorKernel_Image.GetElement(sz,sy,sx));
 }
 __forceinline__ __host__ __device__ void SetElement(size_t z,size_t y,size_t x,type_t value)
 {
  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t tmp=pos;
  tmp/=Dst_X;
  int32_t dx=pos-tmp*Dst_X;
  pos=tmp;
  tmp/=Dst_Y;
  int32_t dy=pos-tmp*Dst_Y;
  pos=tmp;
  tmp/=Input_Z;
  int32_t sz=pos-tmp*Input_Z;
  pos=tmp;
  tmp/=Conv_Kernel_X;
  int32_t kx=pos-tmp*Conv_Kernel_X;
  int32_t ky=tmp;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  sTensorKernel_Image.SetElement(sz,sy,sx,value);
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t y,size_t x)
 {
  return(GetElement(0,y,x));
 }
 __forceinline__ __host__ __device__ void SetElement(size_t y,size_t x,type_t value)
 {
  return(SetElement(0,y,x,value));
 }

 __forceinline__ __host__ __device__ size_t GetSizeX(void) const
 {
  return(Size_X);
 }

 __forceinline__ __host__ __device__ size_t GetSizeY(void) const
 {
  return(Size_Y);
 }

 __forceinline__ __host__ __device__ size_t GetSizeZ(void) const
 {
  return(Size_Z);
 }

 __host__ __device__ void Reset(void)
 {
  sTensorKernel_Image.Reset();
  Conv_Kernel_X=0;
  Conv_Kernel_Y=0;
  Conv_Stride_X=0;
  Conv_Stride_Y=0;
  Conv_Padding_X=0;
  Conv_Padding_Y=0;
  Size_X=0;
  Size_Y=0;
  Size_Z=0;
  Basic_Size_X=0;
  Basic_Size_Y=0;
  Offset_X=0;
  Offset_Y=0;
  Dst_X=0;
  Dst_Y=0;
  Input_Z=0;
 }

 __host__ __device__ void Set(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)
 {
  sTensorKernel_Image.Set(cTensor);

  Input_Z=cTensor.GetSizeZ();
  Conv_Kernel_X=kernel_x;
  Conv_Kernel_Y=kernel_y;
  Conv_Stride_X=stride_x;
  Conv_Stride_Y=stride_y;
  Conv_Padding_X=padding_x;
  Conv_Padding_Y=padding_y;
  Size_X=input_x;
  Size_Y=input_y;
  Size_Z=1;
  Basic_Size_X=Size_X;
  Basic_Size_Y=Size_Y;
  Offset_X=0;
  Offset_Y=0;
  Dst_X=dst_x;
  Dst_Y=dst_y;
 }
};

//****************************************************************************************************
///!структура ядра тензора дельт для вычисления поправок
//****************************************************************************************************
template<class type_t>
struct STensorKernel_DeltaWeightAndBias_Delta
{
 STensorKernel<type_t> sTensorKernel_Delta;///<исходный тензор

 int32_t Size_X;
 int32_t Size_Y;
 int32_t Size_Z;
 int32_t Offset_X;
 int32_t Offset_Y;
 int32_t Stride_X;
 int32_t Stride_Y;
 int32_t NewDelta_X;
 int32_t NewDelta_Y;

 __host__ __device__ STensorKernel_DeltaWeightAndBias_Delta()///<конструктор
 {
 }
 __host__ __device__ STensorKernel_DeltaWeightAndBias_Delta(const CTensor<type_t> &cTensor,int32_t new_delta_y,int32_t new_delta_x,int32_t stride_y,int32_t stride_x)///<конструктор
 {
  Set(cTensor,new_delta_y,new_delta_x,stride_y,stride_x);
 }

 __forceinline__ __host__ __device__ type_t* GetTensorDataPtr(size_t z)///<получить указатель на элементы с глубиной z
 {
  return(NULL);
 }

 __forceinline__ __host__ __device__ void SelectZ(size_t z)///<выбрать слой Z
 {
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t z,size_t y,size_t x)
 {
  int32_t sx=x+Offset_X;
  int32_t sy=y+Offset_Y;

  int32_t dy=sx/NewDelta_X;
  int32_t dx=sx-dy*NewDelta_X;

  //пересчитываем в новые размеры матрицы поправок
  int32_t dx_s=dx/Stride_X;
  int32_t dy_s=dy/Stride_Y;
  if (dx==dx_s*Stride_X && dy==dy_s*Stride_Y) return(sTensorKernel_Delta.GetElement(sy,dy_s,dx_s));
  return(0);
 }
 __forceinline__ __host__ __device__ void SetElement(size_t z,size_t y,size_t x,type_t value)
 {
  int32_t sx=x+Offset_X;
  int32_t sy=y+Offset_Y;

  int32_t dy=sx/NewDelta_X;
  int32_t dx=sx-dy*NewDelta_X;

  //пересчитываем в новые размеры матрицы поправок
  int32_t dx_s=dx/Stride_X;
  int32_t dy_s=dy/Stride_Y;
  if (dx==dx_s*Stride_X && dy==dy_s*Stride_Y) sTensorKernel_Delta.SetElement(sy,dy_s,dx_s,value);
 }

 __forceinline__ __host__ __device__ type_t GetElement(size_t y,size_t x)
 {
  return(GetElement(0,y,x));
 }
 __forceinline__ __host__ __device__ void SetElement(size_t y,size_t x,type_t value)
 {
  return(SetElement(0,y,x,value));
 }

 __forceinline__ __host__ __device__ size_t GetSizeX(void) const
 {
  return(Size_X);
 }

 __forceinline__ __host__ __device__ size_t GetSizeY(void) const
 {
  return(Size_Y);
 }

 __forceinline__ __host__ __device__ size_t GetSizeZ(void) const
 {
  return(Size_Z);
 }

 __host__ __device__ void Reset(void)
 {
  sTensorKernel_Delta.Reset();
  Size_X=0;
  Size_Y=0;
  Size_Z=0;
  Offset_X=0;
  Offset_Y=0;
  Stride_X=0;
  Stride_Y=0;
  NewDelta_X=0;
  NewDelta_Y=0;
 }

 __host__ __device__ void Set(const CTensor<type_t> &cTensor,int32_t new_delta_y,int32_t new_delta_x,int32_t stride_y,int32_t stride_x)
 {
  sTensorKernel_Delta.Set(cTensor);

  Size_Z=1;
  Size_Y=cTensor.GetSizeZ();
  Size_X=new_delta_x*new_delta_y;
  Offset_X=0;
  Offset_Y=0;
  Stride_X=stride_x;
  Stride_Y=stride_y;
  NewDelta_X=new_delta_x;
  NewDelta_Y=new_delta_y;
 }
};

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений для прямой свёртки через умножение матриц
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

 //перестроим входной тензор для выполнения умножения
 int32_t dst_y=dkernel_y;
 int32_t dst_x=dkernel_x;

 int32_t new_input_z=1;
 int32_t new_input_y=new_delta_y*new_delta_x;
 int32_t new_input_x=dst_x*dst_y*image_z;

 //умножаем матрицы
 CTensor<type_t> cTensor_Output(1,delta_z,new_input_x);

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel_DeltaWeightAndBias_Delta<type_t> sTensorKernel_Delta(cTensor_Delta,new_delta_y,new_delta_x,stride_y,stride_x);
 STensorKernel_DeltaWeightAndBias_Image<type_t> sTensorKernel_Image(cTensor_Image,new_input_y,new_input_x,new_delta_y,new_delta_x,1,1,padding_y,padding_x,dst_y,dst_x);

 CTensorMath<type_t>::MulAbstract(cTensor_Output,sTensorKernel_Output,cTensor_Delta,sTensorKernel_Delta,cTensor_Image,sTensorKernel_Image);

 cTensor_Output.CopyFromDevice();
 cTensor_Output.ReinterpretSize(dkernel_z*delta_z,dkernel_y,dkernel_x);

 //копируем результаты в поправки ядер (прибавляем к имеющимся)
 cTensor_dKernel.CopyFromDevice();
 for(size_t k=0;k<dkernel_amount;k++)
 {
  const type_t *s_ptr=cTensor_Output.GetColumnPtr(k*dkernel_z,0);
  type_t *d_ptr=cTensor_dKernel.GetColumnPtr(0,k);
  for(size_t n=0;n<dkernel_x*dkernel_y*dkernel_z;n++,s_ptr++,d_ptr++) *d_ptr+=*s_ptr;
 }
 cTensor_dKernel.SetHostOnChange();

 CTensor<type_t> dB=cTensor_dBias;

 cTensor_dBias.CopyToDevice();
 CTensorMath<type_t>::SummXY(dB,cTensor_Delta);
 CTensorMath<type_t>::Add(cTensor_dBias,cTensor_dBias,dB,1,1);
 cTensor_dBias.CopyFromDevice();
}

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений для обратной свёртки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::CreateBackDeltaWeightAndBias(CTensor<type_t> &cTensor_dKernel,int32_t dkernel_x,int32_t dkernel_y,int32_t dkernel_z,size_t dkernel_amount,CTensor<type_t> &cTensor_dBias,CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y)
{
 CTensorConv<type_t>::CreateDeltaWeightAndBias(cTensor_dKernel,dkernel_x,dkernel_y,dkernel_z,dkernel_amount,cTensor_dBias,cTensor_Delta,cTensor_Image,stride_x,stride_y,padding_x,padding_y);
 cTensor_dBias.Zero();//TODO: неясно, нужно ли использовать эти поправки
}

#endif

#endif
