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
  static void CreateDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta);///<вычисление поправок весов и смещений
  static void CreateBackDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta);///<создание поправок весов и смещений для обратной свёртки
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
//функция CUDA для выполнения прямой свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAForwardConvolutionFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_image,STensorKernel<type_t> tensor_kernel,type_t** kernel_item_ptr_array,type_t* bias_ptr,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 int32_t x=blockIdx.x;
 int32_t y=blockIdx.y;
 int32_t z=threadIdx.x;

 int32_t output_z=z;

 int32_t kernel_x=tensor_kernel.GetSizeX();
 int32_t kernel_y=tensor_kernel.GetSizeY();
 int32_t kernel_z=tensor_kernel.GetSizeZ();

 int32_t input_x=tensor_image.GetSizeX();
 int32_t input_y=tensor_image.GetSizeY();

 //настроим ядро
 tensor_kernel.TensorData_Ptr=kernel_item_ptr_array[output_z];

 type_t sum=bias_ptr[output_z];//сразу прибавляем смещение
 //применяем фильтр
 for(int32_t ky=0;ky<kernel_y;ky++)
 {
  int32_t y0=step_y*y+ky-padding_y;
  if (y0<0 || y0>=input_y) continue;
  for(int32_t kx=0;kx<kernel_x;kx++)
  {
   int32_t x0=step_x*x+kx-padding_x;
   //игнорируем элементы вне границ входного тензора
   if (x0<0 || x0>=input_x) continue;
   //проходимся по всей глубине тензора и считаем сумму
   for(int32_t z=0;z<kernel_z;z++)
   {
    type_t kernel=tensor_kernel.GetElement(z,ky,kx);
    type_t image=tensor_image.GetElement(z,y0,x0);
    sum+=kernel*image;
   }
  }
 }
 tensor_output.SetElement(output_z,y,x,sum);



/*
 int32_t kx=threadIdx.x;
 int32_t ky=threadIdx.y;
 int32_t kz=threadIdx.z;

 int32_t x=blockIdx.x;
 int32_t y=blockIdx.y;

 //применяем фильтр
 int32_t y0=step_y*y+ky-padding_y;
 int32_t x0=step_x*x+kx-padding_x;
 //игнорируем элементы вне границ входного тензора
 if (y0<0 || y0>=tensor_image.GetSizeY() || x0<0 || x0>=tensor_image.GetSizeX())
 {
  summ[kz][ky][kx]=0;
 }
 else
 {
  type_t kernel=tensor_kernel.GetElement(kz,ky,kx);
  type_t image=tensor_image.GetElement(kz,y0,x0);
  summ[kz][ky][kx]=kernel*image;
 }

 __syncthreads();

 //делаем суммирование результата свёртки
 if (kx==0 && ky==0 && kz==0)
 {
  type_t s=0;
  for(int32_t klz=0;klz<tensor_kernel.GetSizeZ();klz++)
  {
   for(int32_t kly=0;kly<tensor_kernel.GetSizeY();kly++)
   {
    for(int32_t klx=0;klx<tensor_kernel.GetSizeX();klx++)
    {
     s+=summ[klz][kly][klx];
    }
   }
  }
  tensor_output.SetElement(output_z,y,x,s);
 }
 __syncthreads();
 */
}


//----------------------------------------------------------------------------------------------------
/*!прямая свёртка
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

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);//выходной тензор
 STensorKernel<type_t> sTensorKernel_Image(cTensor_Image);//входной тензор

 cTensor_Image.CopyToDevice();

 //копируем на видеокарту указатели и смещения
 CCUDADeviceVector<type_t> bias_array(bias.size());
 bias_array.copy_host_to_device(&bias[0],bias.size());

 CCUDADeviceVector<type_t *> kernel_item_ptr_array(cTensor_Kernel.size());
 std::vector<type_t *> item_ptr(cTensor_Kernel.size());
 for(size_t n=0;n<cTensor_Kernel.size();n++)
 {
  cTensor_Kernel[n].CopyToDevice();
  item_ptr[n]=cTensor_Kernel[n].GetDeviceVector().get();
 }
 kernel_item_ptr_array.copy_host_to_device(&item_ptr[0],item_ptr.size());
 //выполняем свёртку
 STensorKernel<type_t> sTensorKernel_Kernel(cTensor_Kernel[0]);
 dim3 thread(cTensor_Kernel.size(),1,1);
 dim3 blocks(output_x,output_y,1);
 CUDAForwardConvolutionFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Image,sTensorKernel_Kernel,kernel_item_ptr_array.get(),bias_array.get(),padding_x,padding_y,step_x,step_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для выполнения обратной свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDABackwardConvolutionFunction(STensorKernel<type_t> tensor_output_delta,STensorKernel<type_t> tensor_delta,STensorKernel<type_t> tensor_kernel,type_t** kernel_item_ptr_array,size_t kernel_amount,type_t* bias_ptr,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 int32_t x=blockIdx.x;
 int32_t y=blockIdx.y;

 int32_t z=threadIdx.x;

 int32_t output_z=z;

 int32_t kernel_x=tensor_kernel.GetSizeX();
 int32_t kernel_y=tensor_kernel.GetSizeY();
 int32_t kernel_z=tensor_kernel.GetSizeZ();

 int32_t input_x=tensor_delta.GetSizeX();
 int32_t input_y=tensor_delta.GetSizeY();

 type_t sum=0;
 //применяем фильтр
 for(int32_t ky=0;ky<kernel_y;ky++)
 {
  int32_t y0=y+ky-padding_y;
  if (y0<0 || y0>=input_y) continue;
  for(int32_t kx=0;kx<kernel_x;kx++)
  {
   int32_t x0=x+kx-padding_x;
   //игнорируем элементы вне границ входного тензора
   if (x0<0 || x0>=input_x) continue;
   //проходимся по всей глубине тензора и считаем сумму
   for(int32_t k=0;k<kernel_amount;k++)
   {
    //настроим ядро
    tensor_kernel.TensorData_Ptr=kernel_item_ptr_array[k];
    //считаем свёртку
    type_t kernel=tensor_kernel.GetElement(output_z,kernel_y-1-ky,kernel_x-1-kx);
    type_t delta=tensor_delta.GetElement(k,y0,x0);

    sum+=kernel*delta;
   }
  }
 }
 tensor_output_delta.SetElement(output_z,y,x,sum);
}

//----------------------------------------------------------------------------------------------------
/*!обратная свёртка
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

 STensorKernel<type_t> sTensorKernel_OutputDelta(cTensor_OutputDelta);//выходной тензор
 STensorKernel<type_t> sTensorKernel_Delta(cTensor_Delta);//входной тензор

 cTensor_Delta.CopyToDevice();

 //копируем на видеокарту указатели и смещения
 CCUDADeviceVector<type_t> bias_array(bias.size());
 bias_array.copy_host_to_device(&bias[0],bias.size());

 CCUDADeviceVector<type_t *> kernel_item_ptr_array(cTensor_Kernel.size());
 std::vector<type_t *> item_ptr(cTensor_Kernel.size());
 for(size_t n=0;n<cTensor_Kernel.size();n++)
 {
  cTensor_Kernel[n].CopyToDevice();
  item_ptr[n]=cTensor_Kernel[n].GetDeviceVector().get();
 }
 kernel_item_ptr_array.copy_host_to_device(&item_ptr[0],item_ptr.size());
 //выполняем свёртку
 STensorKernel<type_t> sTensorKernel_Kernel(cTensor_Kernel[0]);
 dim3 thread(output_z,1,1);
 dim3 blocks(output_x,output_y,1);
 CUDABackwardConvolutionFunction<type_t><<<blocks,thread>>>(sTensorKernel_OutputDelta,sTensorKernel_Delta,sTensorKernel_Kernel,kernel_item_ptr_array.get(),cTensor_Kernel.size(),bias_array.get(),padding_x,padding_y,step_x,step_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_OutputDelta.SetDeviceOnChange();


/*
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
     for(int32_t kx=0;kx<kernel_x;kx++)
     {
      int32_t y0=y+ky-padding_y;
      int32_t x0=x+kx-padding_x;
      //игнорируем выходящие за границы элементы
      if (y0<0 || y0>=input_y) continue;
      if (x0<0 || x0>=input_x) continue;
      //суммируем по всем фильтрам
      for(int32_t f=0;f<kernel_amount;f++)
      {
       type_t k=cTensor_Kernel[f].GetElement(z,kernel_y-1-ky,kernel_x-1-kx);
       type_t d=cTensor_Delta.GetElement(f,y0,x0);
       summ+=k*d;
      }
     }
    }
    cTensor_OutputDelta.SetElement(z,y,x,summ);
   }
  }
 }
 cTensor_OutputDelta.SetHostOnChange();
*/

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
     int32_t y0=static_cast<int32_t>(y*step_y+ky);//TODO: возможно, ошибочно умножать на шаг
     y0-=static_cast<int32_t>(padding_y);
     if (y0<0 || y0>=input_y) continue;
     for(size_t kx=0;kx<kernel_x;kx++)
     {
      int32_t x0=static_cast<int32_t>(x*step_x+kx);//TODO: возможно, ошибочно умножать на шаг
      x0-=static_cast<int32_t>(padding_x);
      //игнорируем выходящие за границы элементы
      if (x0<0 || x0>=input_x) continue;
      //суммируем по всем ядрам
      size_t offset_k_ptr=(kernel_y-1-ky)*kernel_x+(kernel_x-1-kx)+kernel_depth_offset;
      size_t offset_d_ptr=y0*input_x+x0;
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
 }

 cTensor_OutputDelta.SetHostOnChange();
 */

}


//----------------------------------------------------------------------------------------------------
//функция CUDA для выполнения поправок
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADeltaWeightAndBiasFunction(STensorKernel<type_t> tensor_d_kernel,type_t** d_kernel_item_ptr_array,type_t* d_bias_ptr,STensorKernel<type_t> tensor_image,STensorKernel<type_t> tensor_delta,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 int32_t kx=blockIdx.x;
 int32_t ky=blockIdx.y;
 int32_t kz=blockIdx.z;

 int32_t f=threadIdx.x;

 int32_t image_x=tensor_image.GetSizeX();
 int32_t image_y=tensor_image.GetSizeY();

 int32_t delta_x=tensor_delta.GetSizeX();
 int32_t delta_y=tensor_delta.GetSizeY();
 int32_t delta_z=tensor_delta.GetSizeZ();

 //настроим ядро
 tensor_d_kernel.TensorData_Ptr=d_kernel_item_ptr_array[f];

 for(int32_t y=0;y<delta_y;y++)
 {
  for(int32_t x=0;x<delta_x;x++)
  {
   type_t delta=tensor_delta.GetElement(f,y,x);//запоминаем значение градиента
   int32_t i0=ky+y*step_y-padding_y;//TODO: возможно, ошибочно умножать на шаг
   int32_t j0=kx+x*step_x-padding_x;//TODO: возможно, ошибочно умножать на шаг
   if (i0>=0 && i0<image_y && j0>=0 && j0<image_x)//игнорируем выходящие за границы элементы
   {
    //наращиваем градиент фильтра
    type_t dk=tensor_d_kernel.GetElement(kz,ky,kx);
    dk+=delta*tensor_image.GetElement(kz,i0,j0);
    tensor_d_kernel.SetElement(kz,ky,kx,dk);
   }
   if (kx==0 && ky==0 && kz==0) d_bias_ptr[f]+=delta;
  }
 }
}

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::CreateDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta)
{
 int32_t padding_x=0;//дополнение нулями
 int32_t padding_y=0;//дополнение нулями
 int32_t step_x=1;//шаг свёртки
 int32_t step_y=1;//шаг свёртки

 int32_t dkernel_amount=cTensor_dKernel.size();
 if (dkernel_amount==0) throw("Для создания поправок весов и смещений требуется не пустой вектор поправок к ядрам");
 if (dbias.size()!=dkernel_amount) throw("Для создания поправок весов и смещений требуется чтобы количество поправок фильтров и поправок сдвигов совпадало");

 //TODO: считать все тензоры с видеокарты

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

 cTensor_Delta.CopyToDevice();
 cTensor_Image.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Delta(cTensor_Delta);
 STensorKernel<type_t> sTensorKernel_Image(cTensor_Image);

 CCUDADeviceVector<type_t> d_bias_array(dbias.size());
 d_bias_array.copy_host_to_device(&dbias[0],dbias.size());

 CCUDADeviceVector<type_t *> d_kernel_item_ptr_array(cTensor_dKernel.size());
 std::vector<type_t *> item_ptr(cTensor_dKernel.size());
 for(size_t n=0;n<cTensor_dKernel.size();n++)
 {
  cTensor_dKernel[n].CopyToDevice();
  item_ptr[n]=cTensor_dKernel[n].GetDeviceVector().get();
 }
 d_kernel_item_ptr_array.copy_host_to_device(&item_ptr[0],item_ptr.size());
 //выполняем свёртку
 STensorKernel<type_t> sTensorKernel_dKernel(cTensor_dKernel[0]);
 dim3 thread(dkernel_amount,1,1);
 dim3 blocks(dkernel_x,dkernel_y,dkernel_z);
 CUDADeltaWeightAndBiasFunction<type_t><<<blocks,thread>>>(sTensorKernel_dKernel,d_kernel_item_ptr_array.get(),d_bias_array.get(),sTensorKernel_Image,sTensorKernel_Delta,padding_x,padding_y,step_x,step_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 for(size_t n=0;n<cTensor_dKernel.size();n++)
 {
  cTensor_dKernel[n].SetDeviceOnChange();
 }
 d_bias_array.copy_device_to_host(&dbias[0],dbias.size());

/*
 for(int32_t f=0;f<dkernel_amount;f++)
 {
  cTensor_dKernel[f].SetHostOnChange();

  for(int32_t y=0;y<delta_y;y++)
  {
   for(int32_t x=0;x<delta_x;x++)
   {
    type_t delta=cTensor_Delta.GetElement(f,y,x);//запоминаем значение градиента
    for(int32_t i=0;i<dkernel_y;i++)
    {
     for(int32_t j=0;j<dkernel_x;j++)
     {
      int32_t i0=i+y*step_y-padding_y;//TODO: возможно, ошибочно умножать на шаг
      int32_t j0=j+x*step_x-padding_x;//TODO: возможно, ошибочно умножать на шаг
      //игнорируем выходящие за границы элементы
      if (i0<0 || i0>=image_y || j0<0 || j0>=image_x) continue;
      //наращиваем градиент фильтра
      for(int32_t c=0;c<dkernel_z;c++)
      {
       type_t dk=cTensor_dKernel[f].GetElement(c,i,j);
       dk+=delta*cTensor_Image.GetElement(c,i0,j0);
       cTensor_dKernel[f].SetElement(c,i,j,dk);
      }
     }
    }
    dbias[f]+=delta;
   }
  }
 }*/

}

//----------------------------------------------------------------------------------------------------
//функция CUDA для выполнения поправок для обратной свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDABackDeltaWeightAndBiasFunction(STensorKernel<type_t> tensor_d_kernel,type_t** d_kernel_item_ptr_array,type_t* d_bias_ptr,STensorKernel<type_t> tensor_image,STensorKernel<type_t> tensor_delta,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 int32_t kx=blockIdx.x;
 int32_t ky=blockIdx.y;
 int32_t kz=blockIdx.z;

 int32_t f=threadIdx.x;

 int32_t image_x=tensor_image.GetSizeX();
 int32_t image_y=tensor_image.GetSizeY();

 int32_t delta_x=tensor_delta.GetSizeX();
 int32_t delta_y=tensor_delta.GetSizeY();
 int32_t delta_z=tensor_delta.GetSizeZ();

 //настроим ядро
 tensor_d_kernel.TensorData_Ptr=d_kernel_item_ptr_array[f];

 for(int32_t y=0;y<image_y;y++)
 {
  for(int32_t x=0;x<image_x;x++)
  {
   type_t image=tensor_image.GetElement(f,y,x);//запоминаем значение градиента
   int32_t i0=ky+y*step_y-padding_y;//TODO: возможно, ошибочно умножать на шаг
   int32_t j0=kx+x*step_x-padding_x;//TODO: возможно, ошибочно умножать на шаг
   if (i0>=0 && i0<delta_y && j0>=0 && j0<delta_x)//игнорируем выходящие за границы элементы
   {
    //наращиваем градиент фильтра
    type_t dk=tensor_d_kernel.GetElement(kz,ky,kx);
    dk+=image*tensor_delta.GetElement(kz,i0,j0);
    tensor_d_kernel.SetElement(kz,ky,kx,dk);
   }
  }
 }

}

//----------------------------------------------------------------------------------------------------
/*!создание поправок весов и смещений для обратной свёртки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorConv<type_t>::CreateBackDeltaWeightAndBias(std::vector<CTensor<type_t> > &cTensor_dKernel,std::vector<type_t> &dbias,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_Delta)
{
 int32_t padding_x=0;//дополнение нулями
 int32_t padding_y=0;//дополнение нулями
 int32_t step_x=1;//шаг свёртки
 int32_t step_y=1;//шаг свёртки

 int32_t dkernel_amount=cTensor_dKernel.size();
 if (dkernel_amount==0) throw("Для создания поправок весов и смещений требуется не пустой вектор поправок к ядрам");
 if (dbias.size()!=dkernel_amount) throw("Для создания поправок весов и смещений требуется чтобы количество поправок фильтров и поправок сдвигов совпадало");

 //TODO: считать все тензоры с видеокарты

 int32_t delta_x=cTensor_Delta.Size_X;
 int32_t delta_y=cTensor_Delta.Size_Y;
 int32_t delta_z=cTensor_Delta.Size_Z;

 int32_t image_x=cTensor_Image.Size_X;
 int32_t image_y=cTensor_Image.Size_Y;
 int32_t image_z=cTensor_Image.Size_Z;

 int32_t dkernel_x=delta_x-image_x+1;
 int32_t dkernel_y=delta_y-image_y+1;
 int32_t dkernel_z=delta_z;

 if (dkernel_x!=cTensor_dKernel[0].Size_X || dkernel_y!=cTensor_dKernel[0].Size_Y || dkernel_z!=cTensor_dKernel[0].Size_Z) throw("Неверные размеры тензора поправок к ядрам для обновления весов и смещений");

 cTensor_Delta.CopyToDevice();
 cTensor_Image.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Delta(cTensor_Delta);
 STensorKernel<type_t> sTensorKernel_Image(cTensor_Image);

 CCUDADeviceVector<type_t> d_bias_array(dbias.size());
 d_bias_array.copy_host_to_device(&dbias[0],dbias.size());

 CCUDADeviceVector<type_t *> d_kernel_item_ptr_array(cTensor_dKernel.size());
 std::vector<type_t *> item_ptr(cTensor_dKernel.size());
 for(size_t n=0;n<cTensor_dKernel.size();n++)
 {
  cTensor_dKernel[n].CopyToDevice();
  item_ptr[n]=cTensor_dKernel[n].GetDeviceVector().get();
 }
 d_kernel_item_ptr_array.copy_host_to_device(&item_ptr[0],item_ptr.size());
 //выполняем свёртку
 STensorKernel<type_t> sTensorKernel_dKernel(cTensor_dKernel[0]);
 dim3 thread(dkernel_amount,1,1);
 dim3 blocks(dkernel_x,dkernel_y,dkernel_z);
 CUDABackDeltaWeightAndBiasFunction<type_t><<<blocks,thread>>>(sTensorKernel_dKernel,d_kernel_item_ptr_array.get(),d_bias_array.get(),sTensorKernel_Image,sTensorKernel_Delta,padding_x,padding_y,step_x,step_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 for(size_t n=0;n<cTensor_dKernel.size();n++)
 {
  cTensor_dKernel[n].SetDeviceOnChange();
 }
 d_bias_array.copy_device_to_host(&dbias[0],dbias.size());

/*
 for(int32_t f=0;f<dkernel_amount;f++)
 {
  cTensor_dKernel[f].CopyFromDevice();
  cTensor_dKernel[f].SetHostOnChange();

  for(int32_t y=0;y<image_y;y++)
  {
   for(int32_t x=0;x<image_x;x++)
   {
    type_t image=cTensor_Image.GetElement(f,y,x);//запоминаем значение изображения
    for(int32_t i=0;i<dkernel_y;i++)
    {
     for(int32_t j=0;j<dkernel_x;j++)
     {
      int32_t i0=i+y*step_y-padding_y;//TODO: возможно, ошибочно умножать на шаг
      int32_t j0=j+x*step_x-padding_x;//TODO: возможно, ошибочно умножать на шаг
      //игнорируем выходящие за границы элементы
      if (i0<0 || i0>=delta_y || j0<0 || j0>=delta_x) continue;
      //наращиваем градиент фильтра
      for(int32_t c=0;c<dkernel_z;c++)
      {
       type_t dk=cTensor_dKernel[f].GetElement(c,i,j);
       dk+=image*cTensor_Delta.GetElement(c,i0,j0);
       cTensor_dKernel[f].SetElement(c,i,j,dk);
      }
     }
    }
   }
  }
 }*/
}


/*

//Свёртка умножением матрицы Теплица (строятся прямо в процессе работы) - ОЧЕНЬ ТОРМОЗИТ!!!
//----------------------------------------------------------------------------------------------------
//функция CUDA для выполнения быстрой прямой свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAFastForwardConvolutionFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_image,STensorKernel<type_t> tensor_kernel,type_t** kernel_item_ptr_array,type_t* bias_ptr,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 //блок TENSOR_OPERATION_BLOCK_SIZE x TENSOR_OPERATION_BLOCK_SIZE в выходном тензоре
 int32_t blockCol=blockIdx.x;
 int32_t blockRow=blockIdx.y;
 int32_t kernel_index=blockIdx.z;

 //координаты элементов внутри блока в выходном тензоре
 int32_t in_block_x=threadIdx.x;
 int32_t in_block_y=threadIdx.y;

 //координаты элементов в выходном тензоре, представленным вертикальным вектором
 int32_t output_x=in_block_x+blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 int32_t output_y=in_block_y+blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 int32_t output_z=kernel_index;

 //настроим ядро
 tensor_kernel.TensorData_Ptr=kernel_item_ptr_array[kernel_index];
 //расчитаем размерности тензоров
 int32_t kernel_size_x=tensor_kernel.GetSizeX();
 int32_t kernel_size_y=tensor_kernel.GetSizeY();
 int32_t kernel_size_z=tensor_kernel.GetSizeZ();

 int32_t input_size_x=tensor_image.GetSizeX();
 int32_t input_size_y=tensor_image.GetSizeY();

 int32_t local_input_size_x=input_size_x+2*padding_x;
 int32_t local_input_size_y=input_size_y+2*padding_y;
 size_t kernel_tensor_size_x=local_input_size_x*local_input_size_y;
 size_t kernel_tensor_size_y=((local_input_size_y-kernel_size_y)/step_y+1)*((local_input_size_x-kernel_size_x)/step_x+1);

 int32_t output_tensor_size_x=(input_size_x-kernel_size_x+2*padding_x)/step_x+1;
 int32_t output_tensor_size_y=(input_size_y-kernel_size_y+2*padding_y)/step_y+1;

 int32_t m_max=kernel_tensor_size_x/CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 if (kernel_tensor_size_x%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE) m_max++;

 //получаем координаты в выходном прямоугольном тензоре, учитывая, что output_y - это координата разложенного в вертикальный вектор выходного тензора
 int32_t output_in_tensor_x=output_y%output_tensor_size_x;
 int32_t output_in_tensor_y=output_y/output_tensor_size_x;

 int32_t kernel_in_tensor_y=output_y;//строка ядра соответствует строке в выходном тензоре
 //место, откуда начинаются данные ядра в данной строке матрицы Теплица
 int32_t kernel_value_offset=local_input_size_x*output_in_tensor_y*step_y+step_x*output_in_tensor_x;

 type_t summ=bias_ptr[output_z];//сразу прибавляем смещение
 //идём по глубине ядер (и входного изображения)
 for(int32_t z=0;z<kernel_size_z;z++)
 {
  //идём по блокам умножаемой строки и стобца
  for(size_t m=0;m<m_max;m++)
  {
   __shared__ type_t Image[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];//данные блока ядра
   __shared__ type_t Kernel[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];//данные блока изображения

   type_t image=0;//значение изображения в точке (in_block_y,in_block_x)
   type_t kernel=0;//значение ядра в точке (in_block_y,in_block_x)

   //координата в строке матрицы Теплица
   int32_t kernel_teplitz_x=m*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+in_block_x;
   if (kernel_teplitz_x>=kernel_value_offset)
   {
    int32_t kernel_offset=kernel_teplitz_x-kernel_value_offset;//вычитаем смещение позиции данных ядра в матрице Теплица
    int32_t kernel_in_tensor_y=kernel_offset/local_input_size_x;//вычисляем каким координатам внутри тензора ядра соответствует данная точка
	int32_t kernel_in_tensor_x=kernel_offset%local_input_size_x;
	if (kernel_in_tensor_x>=0 && kernel_in_tensor_x<kernel_size_x && kernel_in_tensor_y>=0 && kernel_in_tensor_y<kernel_size_y) kernel=tensor_kernel.GetElement(z,kernel_in_tensor_y,kernel_in_tensor_x);
   }

   //координаты в вертикальном векторе изображения
   int32_t image_vector_y=m*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+in_block_y;
   //вычисляем координаты внутри тензора изображения с учётом дополнения
   int32_t image_padding_in_tensor_y=image_vector_y/local_input_size_x;
   int32_t image_padding_in_tensor_x=image_vector_y%local_input_size_x;
   //координаты внутри тензора изображения без дополнения
   int32_t image_in_tensor_y=image_padding_in_tensor_y-padding_y;
   int32_t image_in_tensor_x=image_padding_in_tensor_x-padding_x;
   if (image_in_tensor_y>=0 && image_in_tensor_y<input_size_y && image_in_tensor_x>=0 && image_in_tensor_x<input_size_x)
   {
	image=tensor_image.GetElement(z,image_in_tensor_y,image_in_tensor_x);
   }

   if (image_vector_y>=local_input_size_x*local_input_size_y || in_block_x!=0) image=0;

   //заполняем блок
   if (output_y>=output_tensor_size_y*output_tensor_size_x) kernel=0;

   Image[in_block_y][in_block_x]=image;
   Kernel[in_block_y][in_block_x]=kernel;

   //синхронизируем потоки
   __syncthreads();
   //умножаем
   if (output_y<output_tensor_size_y*output_tensor_size_x && output_x==0)
   {
    for(size_t e=0;e<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;e++) summ+=Kernel[in_block_y][e]*Image[e][in_block_x];
   }
   __syncthreads();
  }
 }
 //сохраняем результат свёртки
 if (output_y<output_tensor_size_y*output_tensor_size_x && output_x==0) tensor_output.SetElement(output_z,output_in_tensor_y,output_in_tensor_x,summ);
}

//----------------------------------------------------------------------------------------------------
//!прямая свёртка
//
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

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);//выходной тензор
 STensorKernel<type_t> sTensorKernel_Image(cTensor_Image);//входной тензор

 cTensor_Image.CopyToDevice();

 //копируем на видеокарту указатели и смещения
 CCUDADeviceVector<type_t> bias_array(bias.size());
 bias_array.copy_host_to_device(&bias[0],bias.size());

 CCUDADeviceVector<type_t *> kernel_item_ptr_array(cTensor_Kernel.size());
 std::vector<type_t *> item_ptr(cTensor_Kernel.size());
 for(size_t n=0;n<cTensor_Kernel.size();n++)
 {
  cTensor_Kernel[n].CopyToDevice();
  item_ptr[n]=cTensor_Kernel[n].GetDeviceVector().get();
 }
 kernel_item_ptr_array.copy_host_to_device(&item_ptr[0],item_ptr.size());
 //выполняем свёртку
 STensorKernel<type_t> sTensorKernel_Kernel(cTensor_Kernel[0]);

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=1;
 size_t block_y=(output_y*output_x)/thread.y;
 if ((output_y*output_x)%thread.y) block_y++;
 size_t block_z=cTensor_Kernel.size();

 dim3 blocks(block_x,block_y,block_z);
 CUDAFastForwardConvolutionFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Image,sTensorKernel_Kernel,kernel_item_ptr_array.get(),bias_array.get(),padding_x,padding_y,step_x,step_y);
 //Test<type_t>(sTensorKernel_Output,sTensorKernel_Image,sTensorKernel_Kernel,kernel_item_ptr_array.get(),bias_array.get(),padding_x,padding_y,step_x,step_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
 cTensor_Output.SetDeviceOnChange();

}

*/

#endif




/*
//----------------------------------------------------------------------------------------------------
//функция для теста быстрой прямой свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
void Test(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_image,STensorKernel<type_t> tensor_kernel,type_t** kernel_item_ptr_array,type_t* bias_ptr,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 char str[255];

 type_t Image[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];//данные блока ядра
 type_t Kernel[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];//данные блока изображения
 static type_t Output[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];

 type_t image_tensor[3][3]={{1,2,3},{4,5,6},{7,8,9}};
 type_t kernel_tensor[2][2]={{1,2},{3,4}};

 for(size_t I=0;I<2;I++)
 {
  for(size_t X=0;X<16;X++)
  {
   for(size_t Y=0;Y<16;Y++)
   {

 //блок TENSOR_OPERATION_BLOCK_SIZE x TENSOR_OPERATION_BLOCK_SIZE в выходном тензоре
 int32_t blockCol=0;//blockIdx.x;
 int32_t blockRow=0;//blockIdx.y;
 int32_t kernel_index=0;//blockIdx.z;

 //координаты элементов внутри блока в выходном тензоре
 int32_t in_block_x=X;//threadIdx.x;
 int32_t in_block_y=Y;//threadIdx.y;

 //координаты элементов в выходном тензоре, представленным вертикальным вектором
 int32_t output_x=in_block_x+blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 int32_t output_y=in_block_y+blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 int32_t output_z=kernel_index;

 //настроим ядро
 //tensor_kernel.TensorData_Ptr=kernel_item_ptr_array[kernel_index];
 //расчитаем размерности тензоров
 int32_t kernel_size_x=tensor_kernel.GetSizeX();
 int32_t kernel_size_y=tensor_kernel.GetSizeY();
 int32_t kernel_size_z=tensor_kernel.GetSizeZ();

 int32_t input_size_x=tensor_image.GetSizeX();
 int32_t input_size_y=tensor_image.GetSizeY();

 int32_t local_input_size_x=input_size_x+2*padding_x;
 int32_t local_input_size_y=input_size_y+2*padding_y;
 size_t kernel_tensor_size_x=local_input_size_x*local_input_size_y;
 size_t kernel_tensor_size_y=((local_input_size_y-kernel_size_y)/step_y+1)*((local_input_size_x-kernel_size_x)/step_x+1);

 int32_t output_tensor_size_x=(input_size_x-kernel_size_x+2*padding_x)/step_x+1;
 int32_t output_tensor_size_y=(input_size_y-kernel_size_y+2*padding_y)/step_y+1;

 int32_t m_max=kernel_tensor_size_x/CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 if (kernel_tensor_size_x%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE) m_max++;

 //получаем координаты в выходном прямоугольном тензоре, учитывая, что output_y - это координата разложенного в вертикальный вектор выходного тензора
 int32_t output_in_tensor_x=output_y%output_tensor_size_x;
 int32_t output_in_tensor_y=output_y/output_tensor_size_x;

 type_t summ=0;//bias_ptr[output_z];//сразу прибавляем смещение
 //идём по глубине ядер (и входного изображения)
 for(int32_t z=0;z<1+0*kernel_size_z;z++)
 {
  //
  int32_t kernel_in_tensor_y=output_y;//строка ядра соответствует строке в выходном тензоре
  //место, откуда начинаются данные ядра в данной строке матрицы Теплица
  int32_t kernel_value_offset=local_input_size_x*output_in_tensor_y*step_y+step_x*output_in_tensor_x;
  //идём по блокам умножаемой строки и стобца
  for(size_t m=0;m<m_max;m++)
  {

   type_t image=0;//значение изображения в точке (in_block_y,in_block_x)
   type_t kernel=0;//значение ядра в точке (in_block_y,in_block_x)

   //координата в строке матрицы Теплица
   int32_t kernel_teplitz_x=m*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+in_block_x;
   if (kernel_teplitz_x>=kernel_value_offset)
   {
    int32_t kernel_offset=kernel_teplitz_x-kernel_value_offset;//вычитаем смещение позиции данных ядра в матрице Теплица
    int32_t kernel_in_tensor_y=kernel_offset/local_input_size_x;//вычисляем каким координатам внутри тензора ядра соответствует данная точка
	int32_t kernel_in_tensor_x=kernel_offset%local_input_size_x;
	if (kernel_in_tensor_x>=0 && kernel_in_tensor_x<kernel_size_x && kernel_in_tensor_y>=0 && kernel_in_tensor_y<kernel_size_y) kernel=kernel_tensor[kernel_in_tensor_y][kernel_in_tensor_x];
   }

   //координаты в вертикальном векторе изображения
   int32_t image_vector_y=m*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+in_block_y;
   //вычисляем координаты внутри тензора изображения с учётом дополнения
   int32_t image_padding_in_tensor_y=image_vector_y/local_input_size_x;
   int32_t image_padding_in_tensor_x=image_vector_y%local_input_size_x;
   //координаты внутри тензора изображения без дополнения
   int32_t image_in_tensor_y=image_padding_in_tensor_y-padding_y;
   int32_t image_in_tensor_x=image_padding_in_tensor_x-padding_x;
   if (image_in_tensor_y>=0 && image_in_tensor_y<input_size_y && image_in_tensor_x>=0 && image_in_tensor_x<input_size_x)
   {
	image=image_tensor[image_in_tensor_y][image_in_tensor_x];
   }

   if (image_vector_y>=local_input_size_x*local_input_size_y || in_block_x!=0) image=0;

   //заполняем блок
   if (output_y>=output_tensor_size_y*output_tensor_size_x) kernel=0;

   Image[in_block_y][in_block_x]=image;
   Kernel[in_block_y][in_block_x]=kernel;

   //синхронизируем потоки
   if (I==0) continue;
   //умножаем
   for(size_t e=0;e<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;e++) summ+=Kernel[in_block_y][e]*Image[e][in_block_x];
   }
 }
 //сохраняем результат свёртки
 if (I==1 && output_y<output_tensor_size_y*output_tensor_size_x && output_x==0)
 {
  printf("Y:%i X:%i Summ:%f\r\n",output_in_tensor_y,output_in_tensor_x,summ);
  if (output_in_tensor_y>=0 && output_in_tensor_y<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE && output_in_tensor_x>=0 && output_in_tensor_y<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE) Output[output_in_tensor_y][output_in_tensor_x]=summ;
 }

   }
  }
 }
 //выводим тензор результата
 for(size_t ny=0;ny<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;ny++)
 {
  std::string line;
  for(size_t nx=0;nx<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;nx++)
  {
   sprintf(str,"%g ",Output[ny][nx]);
   line+=str;
  }
  SYSTEM::PutMessageToConsole(line.c_str());
 }
 throw("Стоп");

}


//----------------------------------------------------------------------------------------------------
//функция CUDA для выполнения быстрой прямой свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAFastForwardConvolutionFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_image,STensorKernel<type_t> tensor_kernel,type_t** kernel_item_ptr_array,type_t* bias_ptr,int32_t padding_x,int32_t padding_y,int32_t step_x,int32_t step_y)
{
 //блок TENSOR_OPERATION_BLOCK_SIZE x TENSOR_OPERATION_BLOCK_SIZE в выходном тензоре
 int32_t blockCol=blockIdx.x;
 int32_t blockRow=blockIdx.y;
 int32_t kernel_index=blockIdx.z;

 //координаты элементов внутри блока в выходном тензоре
 int32_t in_block_x=threadIdx.x;
 int32_t in_block_y=threadIdx.y;

 //координаты элементов в выходном тензоре, представленным вертикальным вектором
 int32_t output_x=in_block_x+blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 int32_t output_y=in_block_y+blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 int32_t output_z=kernel_index;

 //настроим ядро
 tensor_kernel.TensorData_Ptr=kernel_item_ptr_array[kernel_index];
 //расчитаем размерности тензоров
 int32_t kernel_size_x=tensor_kernel.GetSizeX();
 int32_t kernel_size_y=tensor_kernel.GetSizeY();
 int32_t kernel_size_z=tensor_kernel.GetSizeZ();

 int32_t input_size_x=tensor_image.GetSizeX();
 int32_t input_size_y=tensor_image.GetSizeY();

 int32_t local_input_size_x=input_size_x+2*padding_x;
 int32_t local_input_size_y=input_size_y+2*padding_y;
 size_t kernel_tensor_size_x=local_input_size_x*local_input_size_y;
 size_t kernel_tensor_size_y=((local_input_size_y-kernel_size_y)/step_y+1)*((local_input_size_x-kernel_size_x)/step_x+1);

 int32_t output_tensor_size_x=(input_size_x-kernel_size_x+2*padding_x)/step_x+1;
 int32_t output_tensor_size_y=(input_size_y-kernel_size_y+2*padding_y)/step_y+1;

 int32_t m_max=kernel_tensor_size_x/CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 if (kernel_tensor_size_x%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE) m_max++;

 //получаем координаты в выходном прямоугольном тензоре, учитывая, что output_y - это координата разложенного в вертикальный вектор выходного тензора
 int32_t output_in_tensor_x=output_y%output_tensor_size_x;
 int32_t output_in_tensor_y=output_y/output_tensor_size_x;

 type_t summ=bias_ptr[output_z];//сразу прибавляем смещение
 //идём по глубине ядер (и входного изображения)
 for(int32_t z=0;z<kernel_size_z;z++)
 {
  //
  int32_t kernel_in_tensor_y=output_y;//строка ядра соответствует строке в выходном тензоре
  //место, откуда начинаются данные ядра в данной строке матрицы Теплица
  int32_t kernel_value_offset=local_input_size_x*output_in_tensor_y*step_y+step_x*output_in_tensor_x;
  //идём по блокам умножаемой строки и стобца
  for(size_t m=0;m<m_max;m++)
  {
   __shared__ type_t Image[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];//данные блока ядра
   __shared__ type_t Kernel[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];//данные блока изображения

   type_t image=0;//значение изображения в точке (in_block_y,in_block_x)
   type_t kernel=0;//значение ядра в точке (in_block_y,in_block_x)

   //координата в строке матрицы Теплица
   int32_t kernel_teplitz_x=m*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+in_block_x;
   if (kernel_teplitz_x>=kernel_value_offset)
   {
    int32_t kernel_offset=kernel_teplitz_x-kernel_value_offset;//вычитаем смещение позиции данных ядра в матрице Теплица
    int32_t kernel_in_tensor_y=kernel_offset/local_input_size_x;//вычисляем каким координатам внутри тензора ядра соответствует данная точка
	int32_t kernel_in_tensor_x=kernel_offset%local_input_size_x;
	if (kernel_in_tensor_x>=0 && kernel_in_tensor_x<kernel_size_x && kernel_in_tensor_y>=0 && kernel_in_tensor_y<kernel_size_y) kernel=tensor_kernel.GetElement(z,kernel_in_tensor_y,kernel_in_tensor_x);
   }

   //координаты в вертикальном векторе изображения
   int32_t image_vector_y=m*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+in_block_y;
   //вычисляем координаты внутри тензора изображения с учётом дополнения
   int32_t image_padding_in_tensor_y=image_vector_y/local_input_size_x;
   int32_t image_padding_in_tensor_x=image_vector_y%local_input_size_x;
   //координаты внутри тензора изображения без дополнения
   int32_t image_in_tensor_y=image_padding_in_tensor_y-padding_y;
   int32_t image_in_tensor_x=image_padding_in_tensor_x-padding_x;
   if (image_in_tensor_y>=0 && image_in_tensor_y<input_size_y && image_in_tensor_x>=0 && image_in_tensor_x<input_size_x)
   {
	image=tensor_image.GetElement(z,image_in_tensor_y,image_in_tensor_x);
   }

   if (image_vector_y>=local_input_size_x*local_input_size_y || in_block_x!=0) image=0;

   //заполняем блок
   if (output_y>=output_tensor_size_y*output_tensor_size_x) kernel=0;

   Image[in_block_y][in_block_x]=image;
   Kernel[in_block_y][in_block_x]=kernel;

   //синхронизируем потоки
   __syncthreads();
   //умножаем
   for(size_t e=0;e<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;e++) summ+=Kernel[in_block_y][e]*Image[e][in_block_x];
   __syncthreads();
  }
  __syncthreads();
 }
 //сохраняем результат свёртки
 if (output_y<output_tensor_size_y*output_tensor_size_x && output_x==0) tensor_output.SetElement(output_z,output_in_tensor_y,output_in_tensor_x,summ);
}

//----------------------------------------------------------------------------------------------------
//!прямая свёртка
//
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

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);//выходной тензор
 STensorKernel<type_t> sTensorKernel_Image(cTensor_Image);//входной тензор

 cTensor_Image.CopyToDevice();

 //копируем на видеокарту указатели и смещения
 CCUDADeviceVector<type_t> bias_array(bias.size());
 bias_array.copy_host_to_device(&bias[0],bias.size());

 CCUDADeviceVector<type_t *> kernel_item_ptr_array(cTensor_Kernel.size());
 std::vector<type_t *> item_ptr(cTensor_Kernel.size());
 for(size_t n=0;n<cTensor_Kernel.size();n++)
 {
  cTensor_Kernel[n].CopyToDevice();
  item_ptr[n]=cTensor_Kernel[n].GetDeviceVector().get();
 }
 kernel_item_ptr_array.copy_host_to_device(&item_ptr[0],item_ptr.size());
 //выполняем свёртку
 STensorKernel<type_t> sTensorKernel_Kernel(cTensor_Kernel[0]);

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=1;
 size_t block_y=(output_y*output_x)/thread.y;
 if ((output_y*output_x)%thread.y) block_y++;
 size_t block_z=cTensor_Kernel.size();

 dim3 blocks(block_x,block_y,block_z);
 CUDAFastForwardConvolutionFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Image,sTensorKernel_Kernel,kernel_item_ptr_array.get(),bias_array.get(),padding_x,padding_y,step_x,step_y);
 //Test<type_t>(sTensorKernel_Output,sTensorKernel_Image,sTensorKernel_Kernel,kernel_item_ptr_array.get(),bias_array.get(),padding_x,padding_y,step_x,step_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
 cTensor_Output.SetDeviceOnChange();
*/



//Свёртка с умножением матрицы Теплица - матрица строится заранее. Требует
//схема с умножением матриц. Требует дофига памяти (для картинки 300x300 нужно 8 ГБ для матрицы ядра!)

/*


 //представим изображение вектором с учётом дополнения нулями
 int32_t local_input_x=input_x+2*padding_x;
 int32_t local_input_y=input_y+2*padding_y;
 CTensor<type_t> cTensor_LocalImage(input_z,local_input_x*local_input_y,1);
 cTensor_LocalImage.Zero();
 //копируем данные изображения
 for(int32_t z=0;z<input_z;z++)
 {
  for(int32_t y=0;y<local_input_y;y++)
  {
   int32_t yi=y-padding_y;
   if (yi<0 || yi>=input_y) continue;
   for(int32_t x=0;x<local_input_x;x++)
   {
	int32_t xi=x-padding_x;
	if (xi<0 || xi>=input_x) continue;
	type_t v=cTensor_Image.GetElement(z,yi,xi);
	cTensor_LocalImage.SetElement(z,local_input_x*y+x,0,v);
   }
  }
 }

// cTensor_LocalImage.Print("Local Image");

 //представим ядро свёртки матрицей с учётом шага
 size_t kernel_tensor_x=local_input_x*local_input_y;
 size_t kernel_tensor_y=((local_input_y-kernel_y)/step_y+1)*((local_input_x-kernel_x)/step_x+1);

 printf("Kernel Size:%i x %i\r\n",kernel_tensor_y,kernel_tensor_x);

 CTensor<type_t> KernelMatrix(kernel_z,kernel_tensor_y,kernel_tensor_x);
 CTensor<type_t> ConvResult(kernel_z,output_y,output_x);//результат свёртки с ядром
 KernelMatrix.Zero();

   throw("Стоп");

 //KernelMatrix.Print("Матрица ядра",false);

 //KernelMatrix.Print("Ядро");

 for(size_t k=0;k<output_z;k++)//для каждого ядра
 {
  //cTensor_Kernel[k].Print("Ядро");
  for(size_t z=0;z<kernel_z;z++)
  {
   size_t y=0;
   for(size_t oy=0;oy<output_y;oy++)
   {
    size_t offset=local_input_x*oy*step_y;
    for(size_t ox=0;ox<output_x;ox++,y++,offset+=step_x)
    {
     size_t local_offset=offset;
     for(size_t ky=0;ky<kernel_y;ky++)
     {
      for(size_t kx=0;kx<kernel_x;kx++)
      {
       type_t p=cTensor_Kernel[k].GetElement(z,ky,kx);
       KernelMatrix.SetElement(z,y,kx+local_offset,p);
      }
      local_offset+=local_input_x;
     }
    }
   }
  }
  //KernelMatrix.Print("Матрица ядра");

  //делаем умножение матриц
  ConvResult.ReinterpretSize(kernel_z,output_x*output_y,1);
  CTensorMath<type_t>::Mul(ConvResult,KernelMatrix,cTensor_LocalImage);
  ConvResult.RestoreSize();

  //ConvResult.Print("Результат свёртки");
  //суммируем результаты свёртки по глубине и помещаем в выходную матрицу
  for(size_t y=0;y<output_y;y++)
  {
   for(size_t x=0;x<output_x;x++)
   {
    type_t summ=bias[k];
	for(size_t z=0;z<kernel_z;z++)
	{
	 summ+=ConvResult.GetElement(z,y,x);
	}
	cTensor_Output.SetElement(k,y,x,summ);
   }
  }
 }
*/
 //throw("Стоп");


 //cTensor_Output.Print("Общий результат свёртки со всеми ядрами");
 //throw("Стоп");



/*
 //выполняем свёртку
 //для каждого фильтра
 for(size_t k=0;k<output_z;k++)
 {
  for(int32_t y=0;y<output_y;y++)
  {
   for(int32_t x=0;x<output_x;x++)
   {
    type_t sum=bias[k];//b[f];//сразу прибавляем смещение
    //применяем фильтр
    for(int32_t ky=0;ky<kernel_y;ky++)
    {
     for(int32_t kx=0;kx<kernel_x;kx++)
     {
      int32_t y0=step_y*y+ky-padding_y;
      int32_t x0=step_x*x+kx-padding_x;
      //игнорируем элементы вне границ входного тензора
      if (y0<0 || y0>=input_y || x0<0 || x0>=input_x) continue;
      //проходимся по всей глубине тензора и считаем сумму
      for(int32_t z=0;z<input_z;z++)
      {
       type_t kernel=cTensor_Kernel[k].GetElement(z,ky,kx);
       type_t image=cTensor_Image.GetElement(z,y0,x0);
       sum+=kernel*image;
      }
     }
    }
    cTensor_Output.SetElement(k,y,x,sum);
   }
  }
 }
 cTensor_Output.SetHostOnChange();
*/

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
     int32_t y0=static_cast<int32_t>(step_y*y+ky);
     y0-=static_cast<int32_t>(padding_y);
     if (y0<0 || y0>=input_y) continue;
     for(size_t kx=0;kx<kernel_x;kx++)
     {
      int32_t x0=static_cast<int32_t>(step_x*x+kx);
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
 }
 cTensor_Output.SetHostOnChange();
 */

