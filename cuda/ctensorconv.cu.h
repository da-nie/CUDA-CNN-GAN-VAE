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
  static void ForwardConvolution(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Image,const std::vector<CTensor<type_t> > &cTensor_Kernel,const std::vector<type_t> &bias,uint32_t step_y,uint32_t step_x,uint32_t padding_y,uint32_t padding_x);///<прямая свёртка
  static void BackwardConvolution(CTensor<type_t> &cTensor_OutputDelta,const CTensor<type_t> &cTensor_Delta,const std::vector<CTensor<type_t> > &cTensor_Kernel,const std::vector<type_t> &bias);///<обратная свёртка
  static void CreateDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,const CTensor<type_t> &cTensor_Image,CTensor<type_t> &cTensor_Delta);///<вычисление поправок весов и смещений
  static void CreateBackDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta);///<создание поправок весов и смещений для обратной свёртки
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


//----------------------------------------------------------------------------------------------------
//функция CUDA переупорядочения входного тензора в матрицу для свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAForwardConvolutionConvertInputFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,int32_t dst_x,int32_t dst_y,int32_t kernel_x,int32_t kernel_y,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;

 x+=blockCol*CONV_TENSOR_OPERATION_BLOCK_SIZE;
 y+=blockRow*CONV_TENSOR_OPERATION_BLOCK_SIZE;

 int32_t output_x=tensor_output.GetSizeX();
 int32_t output_y=tensor_output.GetSizeY();

 int32_t input_x=tensor_input.GetSizeX();
 int32_t input_y=tensor_input.GetSizeY();

 if (x>=output_x) return;
 if (y>=output_y) return;

 //копируем значение в выходной тензор

 int32_t pos=x+y*output_x;//линейная координата
 int32_t dx=pos%dst_x;
 pos/=dst_x;
 int32_t dy=pos%dst_y;
 pos/=dst_y;
 int32_t kx=pos%kernel_x;
 pos/=kernel_x;
 int32_t ky=pos%kernel_y;
 pos/=kernel_y;
 int32_t z=pos;

 int32_t sy=dy*step_y+ky-padding_y;
 int32_t sx=dx*step_x+kx-padding_x;
 if (sy>=0 && sy<input_y && sx>=0 && sx<input_x)
 {
  type_t e=tensor_input.GetElement(z,sy,sx);
  tensor_output.SetElement(0,y,x,e);
 }
 else tensor_output.SetElement(0,y,x,0);
}


//----------------------------------------------------------------------------------------------------
/*!прямая свёртка через умножение матриц
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::ForwardConvolution(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Image,const std::vector<CTensor<type_t> > &cTensor_Kernel,const std::vector<type_t> &bias,uint32_t step_y,uint32_t step_x,uint32_t padding_y,uint32_t padding_x)
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

 int32_t output_x=(input_x-kernel_x+2*padding_x)/step_x+1;
 int32_t output_y=(input_y-kernel_y+2*padding_y)/step_y+1;

 if (cTensor_Output.Size_X!=output_x || cTensor_Output.Size_Y!=output_y || cTensor_Output.Size_Z!=output_z) throw("Ошибочная размерность выходного тензора для свёртки");

 if (input_z!=kernel_z) throw("Для прямой свёртки требуется чтобы глубина фильтров и входного тензора совпадали");

 //перестроим входной тензор для выполнения умножения
 int32_t dst_y=(input_y+padding_y*2-(kernel_y-1)-1)/step_y+1;
 int32_t dst_x=(input_x+padding_x*2-(kernel_x-1)-1)/step_x+1;

 int32_t new_input_z=1;
 int32_t new_input_y=kernel_y*kernel_x*kernel_z;
 int32_t new_input_x=dst_x*dst_y;
 CTensor<type_t> &cTensor_NewInput=cTensor_Image.GetCUDATmpTensorA(new_input_z,new_input_y,new_input_x);
 /*
 cTensor_Image.CopyFromDevice();

 type_t *d_ptr=cTensor_NewInput.GetColumnPtr(0,0);
 for(int32_t z=0;z<input_z;z++)
 {
  const type_t *i_ptr=cTensor_Image.GetColumnPtr(z,0);
  for(int32_t ky=0;ky<kernel_y;ky++)
  {
   for(int32_t kx=0;kx<kernel_x;kx++)
   {
    for(int32_t dy=0;dy<dst_y;dy++)
    {
     for (int32_t dx=0;dx<dst_x;dx++,d_ptr++)
     {
      int32_t sy=dy*step_y+ky-padding_y;
      int32_t sx=dx*step_x+kx-padding_x;
      if (sy>=0 && sy<input_y && sx>=0 && sx<input_x) *d_ptr=i_ptr[sy*input_x+sx];
                                                 else *d_ptr=0;
     }
    }
   }
  }
 }
 */

 cTensor_Image.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Input(cTensor_Image);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_NewInput);

 dim3 thread(CONV_TENSOR_OPERATION_BLOCK_SIZE,CONV_TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=new_input_x/thread.x;
 if (new_input_x%thread.x) block_x++;
 size_t block_y=new_input_y/thread.y;
 if (new_input_y%thread.y) block_y++;
 size_t block_z=1;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDAForwardConvolutionConvertInputFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,dst_x,dst_y,kernel_x,kernel_y,padding_x,padding_y,step_x,step_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
 cTensor_NewInput.SetDeviceOnChange();

 //перестроим тензоры ядер в строку
 CTensor<type_t> cTensor_NewKernel(1,output_z,kernel_x*kernel_y*kernel_z);
 for(size_t k=0;k<cTensor_Kernel.size();k++)
 {
  cTensor_Kernel[k].CopyFromDevice();
  const type_t *s_ptr=cTensor_Kernel[k].GetColumnPtr(0,0);
  type_t *d_ptr=cTensor_NewKernel.GetColumnPtr(0,k);
  for(size_t n=0;n<kernel_x*kernel_y*kernel_z;n++,s_ptr++,d_ptr++) *d_ptr=*s_ptr;
 }

 //умножаем матрицы
 cTensor_Output.ReinterpretSize(1,output_z,new_input_x);
 CTensorMath<type_t>::Mul(cTensor_Output,cTensor_NewKernel,cTensor_NewInput);
 cTensor_Output.RestoreSize();
 cTensor_Output.CopyFromDevice(true);
 //добавляем смещения
 type_t *o_ptr=cTensor_Output.GetColumnPtr(0,0);
 for(size_t z=0;z<output_z;z++)
 {
  type_t b=bias[z];
  for(size_t n=0;n<output_y*output_x;n++,o_ptr++) *o_ptr+=b;
 }
 cTensor_Output.SetHostOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA переупорядочения входного тензора в матрицу для обратной свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDABackwardConvolutionConvertInputFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,int32_t dst_x,int32_t dst_y,int32_t kernel_x,int32_t kernel_y,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;

 x+=blockCol*CONV_TENSOR_OPERATION_BLOCK_SIZE;
 y+=blockRow*CONV_TENSOR_OPERATION_BLOCK_SIZE;

 int32_t output_x=tensor_output.GetSizeX();
 int32_t output_y=tensor_output.GetSizeY();

 int32_t input_x=tensor_input.GetSizeX();
 int32_t input_y=tensor_input.GetSizeY();

 if (x>=output_x) return;
 if (y>=output_y) return;

 //копируем значение в выходной тензор
 int32_t pos=x+y*output_x;//линейная координата
 int32_t dx=pos%dst_x;
 pos/=dst_x;
 int32_t dy=pos%dst_y;
 pos/=dst_y;
 int32_t kx=pos%kernel_x;
 pos/=kernel_x;
 int32_t ky=pos%kernel_y;
 pos/=kernel_y;
 int32_t z=pos;

 int32_t sy=dy*step_y+ky-padding_y;
 int32_t sx=dx*step_x+kx-padding_x;
 if (sy>=0 && sy<input_y && sx>=0 && sx<input_x)
 {
  type_t e=tensor_input.GetElement(z,sy,sx);
  tensor_output.SetElement(0,y,x,e);
 }
 else tensor_output.SetElement(0,y,x,0);
}

//----------------------------------------------------------------------------------------------------
/*!обратная свёртка через умножение матриц
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::BackwardConvolution(CTensor<type_t> &cTensor_OutputDelta,const CTensor<type_t> &cTensor_Delta,const std::vector<CTensor<type_t> > &cTensor_Kernel,const std::vector<type_t> &bias)
{
 int32_t padding_x=0;//дополнение нулями
 int32_t padding_y=0;//дополнение нулями
 int32_t step_x=1;//шаг свёртки
 int32_t step_y=1;//шаг свёртки

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
 int32_t output_x=step_x*(input_x-1)+kernel_x-2*padding_x;
 int32_t output_y=step_y*(input_y-1)+kernel_y-2*padding_y;
 int32_t output_z=kernel_z;

 padding_x=kernel_x-1-padding_x;
 padding_y=kernel_y-1-padding_y;

 if (cTensor_OutputDelta.Size_X!=output_x || cTensor_OutputDelta.Size_Y!=output_y || cTensor_OutputDelta.Size_Z!=output_z) throw("Ошибочная размерность выходного тензора для обратной свёртки");
 if (input_z!=kernel_amount) throw("Для обратной свёртки требуется чтобы количество фильтров и глубина входного тензора совпадали");

 //перестроим входной тензор для выполнения умножения
 int32_t dst_y=(input_y+padding_y*2-(kernel_y-1)-1)/step_y+1;
 int32_t dst_x=(input_x+padding_x*2-(kernel_x-1)-1)/step_x+1;

 int32_t new_input_z=1;
 int32_t new_input_y=kernel_y*kernel_x*input_z;
 int32_t new_input_x=dst_x*dst_y;
 CTensor<type_t> &cTensor_NewInput=cTensor_Delta.GetCUDATmpTensorB(new_input_z,new_input_y,new_input_x);

 cTensor_Delta.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Delta);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_NewInput);

 dim3 thread(CONV_TENSOR_OPERATION_BLOCK_SIZE,CONV_TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=new_input_x/thread.x;
 if (new_input_x%thread.x) block_x++;
 size_t block_y=new_input_y/thread.y;
 if (new_input_y%thread.y) block_y++;
 size_t block_z=1;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDABackwardConvolutionConvertInputFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,dst_x,dst_y,kernel_x,kernel_y,padding_x,padding_y,step_x,step_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_NewInput.SetDeviceOnChange();

 /*
 cTensor_Delta.CopyFromDevice();

 type_t *d_ptr=cTensor_NewInput.GetColumnPtr(0,0);
 for(int32_t z=0;z<input_z;z++)
 {
  const type_t *i_ptr=cTensor_Delta.GetColumnPtr(z,0);
  for(int32_t ky=0;ky<kernel_y;ky++)
  {
   for(int32_t kx=0;kx<kernel_x;kx++)
   {
    for(int32_t dy=0;dy<dst_y;dy++)
    {
     for (int32_t dx=0;dx<dst_x;dx++,d_ptr++)
     {
      int32_t sy=dy*step_y+ky-padding_y;
      int32_t sx=dx*step_x+kx-padding_x;
      if (sy>=0 && sy<input_y && sx>=0 && sx<input_x) *d_ptr=i_ptr[sy*input_x+sx];
                                                 else *d_ptr=0;
     }
    }
   }
  }
 }
 */

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
   //for(size_t n=0;n<kernel_x*kernel_y;n++,s_ptr++,d_ptr++) *d_ptr=*s_ptr;
  }
 }

 //умножаем матрицы
 cTensor_OutputDelta.ReinterpretSize(1,output_z,new_input_x);
 CTensorMath<type_t>::Mul(cTensor_OutputDelta,cTensor_NewKernel,cTensor_NewInput);
 cTensor_OutputDelta.RestoreSize();
 cTensor_OutputDelta.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA переупорядочения входного тензора в матрицу для создания поправок весов для прямой свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDACreateDeltaWeightAndBiasConvertInputFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,int32_t dst_x,int32_t dst_y,int32_t kernel_x,int32_t kernel_y,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;

 x+=blockCol*CONV_TENSOR_OPERATION_BLOCK_SIZE;
 y+=blockRow*CONV_TENSOR_OPERATION_BLOCK_SIZE;

 int32_t output_x=tensor_output.GetSizeX();
 int32_t output_y=tensor_output.GetSizeY();

 int32_t input_x=tensor_input.GetSizeX();
 int32_t input_y=tensor_input.GetSizeY();
 int32_t input_z=tensor_input.GetSizeZ();

 if (x>=output_x) return;
 if (y>=output_y) return;

 //копируем значение в выходной тензор
 int32_t pos=x+y*output_x;//линейная координата
 int32_t dx=pos%dst_x;
 pos/=dst_x;
 int32_t dy=pos%dst_y;
 pos/=dst_y;
 int32_t z=pos%input_z;
 pos/=input_z;
 int32_t kx=pos%kernel_x;
 pos/=kernel_x;
 int32_t ky=pos;

 int32_t sy=dy*step_y+ky-padding_y;
 int32_t sx=dx*step_x+kx-padding_x;
 if (sy>=0 && sy<input_y && sx>=0 && sx<input_x)
 {
  type_t e=tensor_input.GetElement(z,sy,sx);
  tensor_output.SetElement(0,y,x,e);
 }
 else tensor_output.SetElement(0,y,x,0);


/*
  for(int32_t ky=0;ky<delta_y;ky++)
 {
  for(int32_t kx=0;kx<delta_x;kx++)
  {
   for(int32_t z=0;z<image_z;z++)
   {
    const type_t *i_ptr=cTensor_Image.GetColumnPtr(z,0);
    for(int32_t dy=0;dy<dst_y;dy++)
    {
     for (int32_t dx=0;dx<dst_x;dx++)
     {
      int32_t sy=dy*step_y+ky-padding_y;
      int32_t sx=dx*step_x+kx-padding_x;
      if (sy>=0 && sy<image_y && sx>=0 && sx<image_x) *d_ptr=i_ptr[sy*image_x+sx];
                                                 else *d_ptr=0;
      d_ptr++;
     }
    }
   }
  }
*/
}

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений для прямой свёртки через умножение матриц
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::CreateDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,const CTensor<type_t> &cTensor_Image,CTensor<type_t> &cTensor_Delta)
{
 int32_t padding_x=0;//дополнение нулями
 int32_t padding_y=0;//дополнение нулями
 int32_t step_x=1;//шаг свёртки
 int32_t step_y=1;//шаг свёртки

 int32_t dkernel_amount=cTensor_dKernel.size();
 if (dkernel_amount==0) throw("Для создания поправок весов и смещений требуется не пустой вектор поправок к ядрам");
 if (dbias.size()!=dkernel_amount) throw("Для создания поправок весов и смещений требуется чтобы количество поправок фильтров и поправок сдвигов совпадало");

 int32_t image_x=cTensor_Image.Size_X;
 int32_t image_y=cTensor_Image.Size_Y;
 int32_t image_z=cTensor_Image.Size_Z;

 int32_t delta_x=cTensor_Delta.Size_X;
 int32_t delta_y=cTensor_Delta.Size_Y;
 int32_t delta_z=cTensor_Delta.Size_Z;

 int32_t dkernel_x=image_x-delta_x+1;
 int32_t dkernel_y=image_y-delta_y+1;
 int32_t dkernel_z=image_z;

 if (dkernel_x!=cTensor_dKernel[0].Size_X || dkernel_y!=cTensor_dKernel[0].Size_Y || dkernel_z!=cTensor_dKernel[0].Size_Z) throw("Неверные размеры тензора поправок к ядрам для обновления весов и смещений");
 if (delta_z!=dkernel_amount) throw("Для создания поправок весов и смещений требуется чтобы глубина тензора дельт совпадала с количеством ядер");

 //перестроим входной тензор для выполнения умножения
 int32_t dst_y=(image_y+padding_y*2-(delta_y-1)-1)/step_y+1;
 int32_t dst_x=(image_x+padding_x*2-(delta_x-1)-1)/step_x+1;

 int32_t new_input_z=1;
 int32_t new_input_y=delta_y*delta_x;
 int32_t new_input_x=dst_x*dst_y*image_z;

 CTensor<type_t> &cTensor_NewInput=cTensor_Image.GetCUDATmpTensorC(new_input_z,new_input_y,new_input_x);

 cTensor_Image.CopyToDevice();
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Image);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_NewInput);

 dim3 thread(CONV_TENSOR_OPERATION_BLOCK_SIZE,CONV_TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=new_input_x/thread.x;
 if (new_input_x%thread.x) block_x++;
 size_t block_y=new_input_y/thread.y;
 if (new_input_y%thread.y) block_y++;
 size_t block_z=1;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDACreateDeltaWeightAndBiasConvertInputFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,dst_x,dst_y,delta_x,delta_y,padding_x,padding_y,step_x,step_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_NewInput.SetDeviceOnChange();

 /*

 cTensor_Image.CopyFromDevice();

 type_t *d_ptr=cTensor_NewInput.GetColumnPtr(0,0);

 for(int32_t ky=0;ky<delta_y;ky++)
 {
  for(int32_t kx=0;kx<delta_x;kx++)
  {
   for(int32_t z=0;z<image_z;z++)
   {
    const type_t *i_ptr=cTensor_Image.GetColumnPtr(z,0);
    for(int32_t dy=0;dy<dst_y;dy++)
    {
     for (int32_t dx=0;dx<dst_x;dx++)
     {
      int32_t sy=dy*step_y+ky-padding_y;
      int32_t sx=dx*step_x+kx-padding_x;
      if (sy>=0 && sy<image_y && sx>=0 && sx<image_x) *d_ptr=i_ptr[sy*image_x+sx];
                                                 else *d_ptr=0;
      d_ptr++;
     }
    }
   }
  }
 }
*/

 //перестроим тензоры дельт в строку
 cTensor_Delta.ReinterpretSize(1,delta_z,delta_x*delta_y);
 //умножаем матрицы
 CTensor<type_t> cTensor_Output(1,delta_z,new_input_x);
 CTensorMath<type_t>::Mul(cTensor_Output,cTensor_Delta,cTensor_NewInput);
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
  const type_t *ds_ptr=cTensor_Delta.GetColumnPtr(k,0);
  for(int32_t n=0;n<delta_y*delta_x;n++,ds_ptr++) dbias[k]+=*ds_ptr;
 }
}

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений для обратной свёртки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::CreateBackDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta)
{
 CTensorConv<type_t>::CreateDeltaWeightAndBias(cTensor_dKernel,dbias,cTensor_Delta,cTensor_Image);
}

#endif
