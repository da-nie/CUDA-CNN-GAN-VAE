#ifndef C_TENSOR_CONV_H
#define C_TENSOR_CONV_H

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
  static void ForwardConvolution(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Image,const std::vector<CTensor<type_t> > &cTensor_Kernel,const std::vector<type_t> &bias,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y);///<прямая свёртка
  static void BackwardConvolution(CTensor<type_t> &cTensor_OutputDelta,const CTensor<type_t> &cTensor_Delta,const std::vector<CTensor<type_t> > &cTensor_Kernel,const std::vector<type_t> &bias,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y);///<обратная свёртка
  static void CreateDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,const CTensor<type_t> &cTensor_Image,CTensor<type_t> &cTensor_Delta,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y);///<вычисление поправок весов и смещений
  static void CreateBackDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y);///<создание поправок весов и смещений для обратной свёртки
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};

static const size_t CONV_TENSOR_OPERATION_BLOCK_SIZE=16;

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

 STensorKernel_ForwardConvolution_Image()///<конструктор
 {
 }
 STensorKernel_ForwardConvolution_Image(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)///<конструктор
 {
  Set(cTensor,input_y,input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);
 }

 __host__ __device__ type_t* GetTensorDataPtr(size_t z)///<получить указатель на элементы с глубиной z
 {
  return(NULL);
 }

 __host__ __device__ STensorKernel_ForwardConvolution_Image<type_t> GetSubTensor(size_t z,size_t y,size_t x)///<получить подтензор с глубиной 1 (x,y - координаты блока)
 {
  if (z>=Size_Z) z=0;
  STensorKernel_ForwardConvolution_Image<type_t> sub_tensor;
  sub_tensor.Size_X=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Size_Y=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Size_Z=1;
  sub_tensor.Dst_X=Dst_X;
  sub_tensor.Dst_Y=Dst_Y;
  sub_tensor.Basic_Size_X=Basic_Size_X;
  sub_tensor.Basic_Size_Y=Basic_Size_Y;
  sub_tensor.Conv_Kernel_X=Conv_Kernel_X;
  sub_tensor.Conv_Kernel_Y=Conv_Kernel_Y;
  sub_tensor.Conv_Stride_X=Conv_Stride_X;
  sub_tensor.Conv_Stride_Y=Conv_Stride_Y;
  sub_tensor.Conv_Padding_X=Conv_Padding_X;
  sub_tensor.Conv_Padding_Y=Conv_Padding_Y;
  sub_tensor.sTensorKernel_Image=sTensorKernel_Image;
  sub_tensor.Offset_X=Offset_X+x*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Offset_Y=Offset_Y+y*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;

  //условие не строгое, так как последний блок для матриц не кратных блоку гарантировано будет превышать размер матрицы.
  if ((x+1)*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE>Size_X) sub_tensor.Size_X=Size_X%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  if ((y+1)*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE>Size_Y) sub_tensor.Size_Y=Size_Y%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;

  return(sub_tensor);
 }

 __host__ __device__ type_t GetElement(size_t z,size_t y,size_t x)
 {
  if (x>=Size_X || y>=Size_Y) return(0);
  if (z>=Size_Z) return(0);

  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t dx=pos%Dst_X;
  pos/=Dst_X;
  int32_t dy=pos%Dst_Y;
  pos/=Dst_Y;
  int32_t kx=pos%Conv_Kernel_X;
  pos/=Conv_Kernel_X;
  int32_t ky=pos%Conv_Kernel_Y;
  pos/=Conv_Kernel_Y;
  int32_t sz=pos;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  return(sTensorKernel_Image.TensorData_Ptr[sz*sTensorKernel_Image.StrideZ+sy*sTensorKernel_Image.StrideX+sx]);

  //return(sTensorKernel_Image.GetElement(sz,sy,sx));
 }
 __host__ __device__ void SetElement(size_t z,size_t y,size_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y) return;
  if (z>=Size_Z) return;

  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t dx=pos%Dst_X;
  pos/=Dst_X;
  int32_t dy=pos%Dst_Y;
  pos/=Dst_Y;
  int32_t kx=pos%Conv_Kernel_X;
  pos/=Conv_Kernel_X;
  int32_t ky=pos%Conv_Kernel_Y;
  pos/=Conv_Kernel_Y;
  int32_t sz=pos;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  //sTensorKernel_Image.SetElement(sz,sy,sx,value);
  sTensorKernel_Image.TensorData_Ptr[sz*sTensorKernel_Image.StrideZ+sy*sTensorKernel_Image.StrideX+sx]=value;
 }

 __host__ __device__ size_t GetSizeX(void)
 {
  return(Size_X);
 }

 __host__ __device__ size_t GetSizeY(void)
 {
  return(Size_Y);
 }

 __host__ __device__ size_t GetSizeZ(void)
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

 __host__ void Set(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)
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
void CTensorConv<type_t>::ForwardConvolution(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Image,const std::vector<CTensor<type_t> > &cTensor_Kernel,const std::vector<type_t> &bias,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y)
{
 //вычисляем размеры выходного тензора
 int32_t output_z=cTensor_Kernel.size();
 if (output_z==0) throw("Для прямой свёртки требуется хотя бы одно ядро свёртки");
 if (output_z!=bias.size()) throw("Для прямой свёртки требуется чтобы количество ядер и смещений совпадало");

 int32_t kernel_x=cTensor_Kernel[0].Size_X;
 int32_t kernel_y=cTensor_Kernel[0].Size_Y;
 int32_t kernel_z=cTensor_Kernel[0].Size_Z;

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

 int32_t new_input_z=1;
 int32_t new_input_y=kernel_y*kernel_x*kernel_z;
 int32_t new_input_x=dst_x*dst_y;

 //перестроим тензоры ядер в строку
 CTensor<type_t> cTensor_NewKernel(1,output_z,kernel_x*kernel_y*kernel_z);
 for(size_t k=0;k<cTensor_Kernel.size();k++)
 {
  cTensor_Kernel[k].CopyFromDevice();
  const type_t *s_ptr=cTensor_Kernel[k].GetColumnPtr(0,0);
  type_t *d_ptr=cTensor_NewKernel.GetColumnPtr(0,k);
  for(size_t n=0;n<kernel_x*kernel_y*kernel_z;n++,s_ptr++,d_ptr++) *d_ptr=*s_ptr;
 }
 //перестроим смещения
 CTensor<type_t> cTensor_Bias(bias.size(),1,1);
 for(size_t b=0;b<bias.size();b++)
 {
  cTensor_Bias.SetElement(b,0,0,bias[b]);
 }

 //умножаем матрицы
 cTensor_Output.ReinterpretSize(1,output_z,new_input_x);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_NewKernel(cTensor_NewKernel);
 STensorKernel_ForwardConvolution_Image<type_t> sTensorKernel_Image(cTensor_Image,new_input_y,new_input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);

 CTensorMath<type_t>::MulAbstract(cTensor_Output,sTensorKernel_Output,cTensor_NewKernel,sTensorKernel_NewKernel,cTensor_Image,sTensorKernel_Image);
 cTensor_Output.RestoreSize();

 CTensorMath<type_t>::AddBias(cTensor_Output,cTensor_Bias);
}



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

 STensorKernel_BackwardConvolution_Delta()///<конструктор
 {
 }
 STensorKernel_BackwardConvolution_Delta(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)///<конструктор
 {
  Set(cTensor,input_y,input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);
 }

 __host__ __device__ type_t* GetTensorDataPtr(size_t z)///<получить указатель на элементы с глубиной z
 {
  return(NULL);
 }

 __host__ __device__ STensorKernel_BackwardConvolution_Delta<type_t> GetSubTensor(size_t z,size_t y,size_t x)///<получить подтензор с глубиной 1 (x,y - координаты блока)
 {
  if (z>=Size_Z) z=0;
  STensorKernel_BackwardConvolution_Delta<type_t> sub_tensor;
  sub_tensor.Size_X=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Size_Y=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Size_Z=1;
  sub_tensor.Dst_X=Dst_X;
  sub_tensor.Dst_Y=Dst_Y;
  sub_tensor.Basic_Size_X=Basic_Size_X;
  sub_tensor.Basic_Size_Y=Basic_Size_Y;
  sub_tensor.Conv_Kernel_X=Conv_Kernel_X;
  sub_tensor.Conv_Kernel_Y=Conv_Kernel_Y;
  sub_tensor.Conv_Stride_X=Conv_Stride_X;
  sub_tensor.Conv_Stride_Y=Conv_Stride_Y;
  sub_tensor.Conv_Padding_X=Conv_Padding_X;
  sub_tensor.Conv_Padding_Y=Conv_Padding_Y;
  sub_tensor.sTensorKernel_Delta=sTensorKernel_Delta;
  sub_tensor.Offset_X=Offset_X+x*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Offset_Y=Offset_Y+y*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;

  //условие не строгое, так как последний блок для матриц не кратных блоку гарантировано будет превышать размер матрицы.
  if ((x+1)*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE>Size_X) sub_tensor.Size_X=Size_X%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  if ((y+1)*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE>Size_Y) sub_tensor.Size_Y=Size_Y%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;

  return(sub_tensor);
 }

 __host__ __device__ type_t GetElement(size_t z,size_t y,size_t x)
 {
  if (x>=Size_X || y>=Size_Y) return(0);
  if (z>=Size_Z) return(0);

  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t dx=pos%Dst_X;
  pos/=Dst_X;
  int32_t dy=pos%Dst_Y;
  pos/=Dst_Y;
  int32_t kx=pos%Conv_Kernel_X;
  pos/=Conv_Kernel_X;
  int32_t ky=pos%Conv_Kernel_Y;
  pos/=Conv_Kernel_Y;
  int32_t sz=pos;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  return(sTensorKernel_Delta.GetElement(sz,sy,sx));
 }
 __host__ __device__ void SetElement(size_t z,size_t y,size_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y) return;
  if (z>=Size_Z) return;

  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t dx=pos%Dst_X;
  pos/=Dst_X;
  int32_t dy=pos%Dst_Y;
  pos/=Dst_Y;
  int32_t kx=pos%Conv_Kernel_X;
  pos/=Conv_Kernel_X;
  int32_t ky=pos%Conv_Kernel_Y;
  pos/=Conv_Kernel_Y;
  int32_t sz=pos;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  sTensorKernel_Delta.SetElement(sz,sy,sx,value);
 }

 __host__ __device__ size_t GetSizeX(void)
 {
  return(Size_X);
 }

 __host__ __device__ size_t GetSizeY(void)
 {
  return(Size_Y);
 }

 __host__ __device__ size_t GetSizeZ(void)
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

 __host__ void Set(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)
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
void CTensorConv<type_t>::BackwardConvolution(CTensor<type_t> &cTensor_OutputDelta,const CTensor<type_t> &cTensor_Delta,const std::vector<CTensor<type_t> > &cTensor_Kernel,const std::vector<type_t> &bias,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y)
{
/* int32_t padding_x=0;//дополнение нулями
 int32_t padding_y=0;//дополнение нулями
 int32_t stride_x=1;//шаг свёртки
 int32_t stride_y=1;//шаг свёртки
 */

 //вычисляем размеры выходного тензора
 int32_t kernel_amount=cTensor_Kernel.size();
 if (kernel_amount==0) throw("Для обратной свёртки требуется хотя бы одно ядро свёртки");
 if (kernel_amount!=bias.size()) throw("Для обратной свёртки требуется чтобы количество ядер и смещений совпадало");

 int32_t kernel_x=cTensor_Kernel[0].Size_X;
 int32_t kernel_y=cTensor_Kernel[0].Size_Y;
 int32_t kernel_z=cTensor_Kernel[0].Size_Z;

 int32_t input_y=cTensor_Delta.Size_Y;
 int32_t input_x=cTensor_Delta.Size_X;
 int32_t input_z=cTensor_Delta.Size_Z;

 //обратная свёртка делается с ядрами, повёрнутыми на 180
 padding_x=((cTensor_OutputDelta.Size_X-1)*stride_x+kernel_x-input_x)/2;
 padding_y=((cTensor_OutputDelta.Size_Y-1)*stride_y+kernel_y-input_y)/2;

 int32_t output_x=(input_x-kernel_x+2*padding_x)/stride_x+1;
 int32_t output_y=(input_y-kernel_y+2*padding_y)/stride_y+1;
 int32_t output_z=kernel_z;

 if (cTensor_OutputDelta.Size_X!=output_x || cTensor_OutputDelta.Size_Y!=output_y || cTensor_OutputDelta.Size_Z!=output_z) throw("Ошибочная размерность выходного тензора для обратной свёртки");
 if (input_z!=kernel_amount) throw("Для обратной свёртки требуется чтобы количество фильтров и глубина входного тензора совпадали");

 //перестроим входной тензор для выполнения умножения
 int32_t dst_y=output_y;
 int32_t dst_x=output_x;

 int32_t new_input_z=1;
 int32_t new_input_y=kernel_y*kernel_x*input_z;
 int32_t new_input_x=dst_x*dst_y;

 //перестроим тензоры ядер в строку по глубине и переворачиваем их на 180
 CTensor<type_t> cTensor_NewKernel(1,kernel_z,kernel_x*kernel_y*input_z);
 ///исключение фрагмента даёт 200 мс
 for(size_t z=0;z<kernel_z;z++)
 {
  type_t *d_ptr=cTensor_NewKernel.GetColumnPtr(0,z);
  for(size_t k=0;k<cTensor_Kernel.size();k++)
  {
   cTensor_Kernel[k].CopyFromDevice();
   const type_t *s_ptr=cTensor_Kernel[k].GetColumnPtr(z,0);
   for(size_t ky=0;ky<kernel_y;ky++)
   {
    for(size_t kx=0;kx<kernel_x;kx++,d_ptr++)
	{
     int32_t nky=kernel_y-1-ky;
	 int32_t nkx=kernel_x-1-kx;
	 *d_ptr=s_ptr[nkx+nky*kernel_x];
	}
   }
  }
 }

 //умножаем матрицы
 cTensor_OutputDelta.ReinterpretSize(1,output_z,new_input_x);

 STensorKernel<type_t> sTensorKernel_OutputDelta(cTensor_OutputDelta);
 STensorKernel<type_t> sTensorKernel_NewKernel(cTensor_NewKernel);
 STensorKernel_BackwardConvolution_Delta<type_t> sTensorKernel_Delta(cTensor_Delta,new_input_y,new_input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);

 CTensorMath<type_t>::MulAbstract(cTensor_OutputDelta,sTensorKernel_OutputDelta,cTensor_NewKernel,sTensorKernel_NewKernel,cTensor_Delta,sTensorKernel_Delta);
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

 STensorKernel_DeltaWeightAndBias_Image()///<конструктор
 {
 }
 STensorKernel_DeltaWeightAndBias_Image(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)///<конструктор
 {
  Set(cTensor,input_y,input_x,kernel_y,kernel_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);
 }

 __host__ __device__ type_t* GetTensorDataPtr(size_t z)///<получить указатель на элементы с глубиной z
 {
  return(NULL);
 }

 __host__ __device__ STensorKernel_DeltaWeightAndBias_Image<type_t> GetSubTensor(size_t z,size_t y,size_t x)///<получить подтензор с глубиной 1 (x,y - координаты блока)
 {
  if (z>=Size_Z) z=0;
  STensorKernel_DeltaWeightAndBias_Image<type_t> sub_tensor;
  sub_tensor.Size_X=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Size_Y=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Size_Z=1;
  sub_tensor.Dst_X=Dst_X;
  sub_tensor.Dst_Y=Dst_Y;
  sub_tensor.Input_Z=Input_Z;
  sub_tensor.Basic_Size_X=Basic_Size_X;
  sub_tensor.Basic_Size_Y=Basic_Size_Y;
  sub_tensor.Conv_Kernel_X=Conv_Kernel_X;
  sub_tensor.Conv_Kernel_Y=Conv_Kernel_Y;
  sub_tensor.Conv_Stride_X=Conv_Stride_X;
  sub_tensor.Conv_Stride_Y=Conv_Stride_Y;
  sub_tensor.Conv_Padding_X=Conv_Padding_X;
  sub_tensor.Conv_Padding_Y=Conv_Padding_Y;
  sub_tensor.sTensorKernel_Image=sTensorKernel_Image;
  sub_tensor.Offset_X=Offset_X+x*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Offset_Y=Offset_Y+y*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;

  //условие не строгое, так как последний блок для матриц не кратных блоку гарантировано будет превышать размер матрицы.
  if ((x+1)*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE>Size_X) sub_tensor.Size_X=Size_X%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  if ((y+1)*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE>Size_Y) sub_tensor.Size_Y=Size_Y%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;

  return(sub_tensor);
 }

 __host__ __device__ type_t GetElement(size_t z,size_t y,size_t x)
 {
  if (x>=Size_X || y>=Size_Y) return(0);
  if (z>=Size_Z) return(0);

  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата
  int32_t dx=pos%Dst_X;
  pos/=Dst_X;
  int32_t dy=pos%Dst_Y;
  pos/=Dst_Y;
  int32_t sz=pos%Input_Z;
  pos/=Input_Z;
  int32_t kx=pos%Conv_Kernel_X;
  pos/=Conv_Kernel_X;
  int32_t ky=pos;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  return(sTensorKernel_Image.GetElement(sz,sy,sx));
 }
 __host__ __device__ void SetElement(size_t z,size_t y,size_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y) return;
  if (z>=Size_Z) return;

  int32_t pos=(x+Offset_X)+(y+Offset_Y)*Basic_Size_X;//линейная координата

  int32_t dx=pos%Dst_X;
  pos/=Dst_X;
  int32_t dy=pos%Dst_Y;
  pos/=Dst_Y;
  int32_t sz=pos%Input_Z;
  pos/=Input_Z;
  int32_t kx=pos%Conv_Kernel_X;
  pos/=Conv_Kernel_X;
  int32_t ky=pos;

  int32_t sy=dy*Conv_Stride_Y+ky-Conv_Padding_Y;
  int32_t sx=dx*Conv_Stride_X+kx-Conv_Padding_X;

  sTensorKernel_Image.SetElement(sz,sy,sx,value);
 }

 __host__ __device__ size_t GetSizeX(void)
 {
  return(Size_X);
 }

 __host__ __device__ size_t GetSizeY(void)
 {
  return(Size_Y);
 }

 __host__ __device__ size_t GetSizeZ(void)
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

 __host__ void Set(const CTensor<type_t> &cTensor,int32_t input_y,int32_t input_x,int32_t kernel_y,int32_t kernel_x,int32_t stride_y,int32_t stride_x,int32_t padding_y,int32_t padding_x,int32_t dst_y,int32_t dst_x)
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

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений для прямой свёртки через умножение матриц
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::CreateDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,const CTensor<type_t> &cTensor_Image,CTensor<type_t> &cTensor_Delta,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y)
{
 int32_t dkernel_amount=cTensor_dKernel.size();
 if (dkernel_amount==0) throw("Для создания поправок весов и смещений требуется не пустой вектор поправок к ядрам");
 if (dbias.size()!=dkernel_amount) throw("Для создания поправок весов и смещений требуется чтобы количество поправок фильтров и поправок сдвигов совпадало");

 int32_t image_x=cTensor_Image.Size_X;
 int32_t image_y=cTensor_Image.Size_Y;
 int32_t image_z=cTensor_Image.Size_Z;

 int32_t delta_x=cTensor_Delta.Size_X;
 int32_t delta_y=cTensor_Delta.Size_Y;
 int32_t delta_z=cTensor_Delta.Size_Z;

 int32_t dkernel_x=(image_x-delta_x+2*padding_x)/stride_x+1;
 int32_t dkernel_y=(image_y-delta_y+2*padding_y)/stride_y+1;

 int32_t dkernel_z=image_z;
 if (dkernel_x!=cTensor_dKernel[0].Size_X || dkernel_y!=cTensor_dKernel[0].Size_Y || dkernel_z!=cTensor_dKernel[0].Size_Z) throw("Неверные размеры тензора поправок к ядрам для обновления весов и смещений");
 if (delta_z!=dkernel_amount) throw("Для создания поправок весов и смещений требуется чтобы глубина тензора дельт совпадала с количеством ядер");

 //перестроим входной тензор для выполнения умножения
 int32_t dst_y=dkernel_y;
 int32_t dst_x=dkernel_x;

 int32_t new_input_z=1;
 int32_t new_input_y=delta_y*delta_x;
 int32_t new_input_x=dst_x*dst_y*image_z;

 //перестроим тензоры дельт в строку
 cTensor_Delta.ReinterpretSize(1,delta_z,delta_x*delta_y);
 //умножаем матрицы
 CTensor<type_t> cTensor_Output(1,delta_z,new_input_x);

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Delta(cTensor_Delta);
 STensorKernel_DeltaWeightAndBias_Image<type_t> sTensorKernel_Image(cTensor_Image,new_input_y,new_input_x,delta_y,delta_x,stride_y,stride_x,padding_y,padding_x,dst_y,dst_x);

 CTensorMath<type_t>::MulAbstract(cTensor_Output,sTensorKernel_Output,cTensor_Delta,sTensorKernel_Delta,cTensor_Image,sTensorKernel_Image);

 cTensor_Delta.ReinterpretSize(delta_z,delta_y,delta_x);
 cTensor_Output.CopyFromDevice();
 cTensor_Output.ReinterpretSize(dkernel_z*delta_z,dkernel_y,dkernel_x);

 //копируем результаты в поправки ядер
 for(size_t k=0;k<dkernel_amount;k++)
 {
  cTensor_dKernel[k].CopyFromDevice();
  const type_t *s_ptr=cTensor_Output.GetColumnPtr(k*dkernel_z,0);
  type_t *d_ptr=cTensor_dKernel[k].GetColumnPtr(0,0);
  for(size_t n=0;n<dkernel_x*dkernel_y*dkernel_z;n++,s_ptr++,d_ptr++) *d_ptr+=*s_ptr;
  cTensor_dKernel[k].SetHostOnChange();
  //считаем поправку к вектору сдвига
  /*
  const type_t *ds_ptr=cTensor_Delta.GetColumnPtr(k,0);
  type_t summ=0;
  for(int32_t n=0;n<delta_y*delta_x;n++,ds_ptr++) summ+=*ds_ptr;
  dbias[k]+=summ;
  */
 }

 //считаем поправку к вектору сдвига
 CTensor<type_t> cTensor_dBias(dbias.size(),1,1);
 CTensorMath<type_t>::SummXY(cTensor_dBias,cTensor_Delta);
 for(size_t b=0;b<dbias.size();b++)
 {
  dbias[b]+=cTensor_dBias.GetElement(b,0,0);
 }
}

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений для обратной свёртки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::CreateBackDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y)
{
 CTensorConv<type_t>::CreateDeltaWeightAndBias(cTensor_dKernel,dbias,cTensor_Delta,cTensor_Image,stride_x,stride_y,padding_x,padding_y);
}

#endif
