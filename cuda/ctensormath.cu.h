#ifndef C_TENSOR_MATH_CU_H
#define C_TENSOR_MATH_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensormath.h"

#endif


#ifndef USE_CPU

//****************************************************************************************************
//Операции над тензорами произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "ctensor.cu.h"

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

template<class type_t>
class CTensorMath
{
 template<class new_type_t>
 friend struct STensorKernel;
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  struct SPos
  {
   size_t X;
   size_t Y;
  };
  //-константы------------------------------------------------------------------------------------------
  static const size_t TENSOR_OPERATION_BLOCK_SIZE=16;///<размер блока операций с тензорами
 private:
  //-переменные-----------------------------------------------------------------------------------------
 public:
  //-конструктор----------------------------------------------------------------------------------------
  //-конструктор копирования----------------------------------------------------------------------------
  //-деструктор-----------------------------------------------------------------------------------------
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  static void Add(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<сложить тензоры
  static void Sub(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<вычесть тензоры
  static void AddBias(CTensor<type_t> &cTensor_Working,const CTensor<type_t> &cTensor_Bias);///<добавить смещения к элементам тензора (смещения одинаковы для x и y, но по z смещения разные)
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

  static void MaxPooling(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,size_t pooling_x,size_t pooling_y);///<уменьшение разрешения тензора выборкой большего элемента
  static void MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,size_t pooling_x,size_t pooling_y);///<обратный проход при увеличении разрешения тензора выборкой большего элемента
  static void Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value);///<выполнить отсечку значений тензора

  static void Adam(CTensor<type_t> &cTensor_Weight,CTensor<type_t> &cTensor_dWeight,CTensor<type_t> &cTensor_M,CTensor<type_t> &cTensor_V,double speed,double beta1,double beta2,double epsilon,double iteration);///<выполнить алгоритм Adam к весовому тензору


 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};

//****************************************************************************************************
///!структура ядра тензора
//****************************************************************************************************
template<class type_t>
struct STensorKernel
{
 size_t Size_X;///<размер по x
 size_t Size_Y;///<размер по y
 size_t Size_Z;///<размер по z
 size_t StrideX;///<строка по X
 size_t StrideZ;///<размер блока по Z
 type_t *TensorData_Ptr;///<указатель на данные тензора на стороне GPU

 STensorKernel(void)///<конструктор
 {
 }
 STensorKernel(const CTensor<type_t> &cTensor)///<конструктор
 {
  Set(cTensor);
 }

 __host__ __device__ type_t* GetTensorDataPtr(size_t z)///<получить указатель на элементы с глубиной z
 {
  return(&TensorData_Ptr[z*StrideZ]);
 }

 __host__ __device__ STensorKernel GetSubTensor(size_t z,size_t y,size_t x)///<получить подтензор с глубиной 1
 {
  STensorKernel sub_tensor;
  sub_tensor.Size_X=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Size_Y=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  sub_tensor.Size_Z=1;
  sub_tensor.StrideX=Size_X;
  sub_tensor.StrideZ=0;
  sub_tensor.TensorData_Ptr=&TensorData_Ptr[z*StrideZ+StrideX*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*y+CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*x];

  //условие не строгое, так как последний блок для матриц не кратных блоку гарантировано будет превышать размер матрицы.
  if ((x+1)*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE>Size_X) sub_tensor.Size_X=Size_X%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
  if ((y+1)*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE>Size_Y) sub_tensor.Size_Y=Size_Y%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;

  return(sub_tensor);
 }

 __host__ __device__ type_t GetElement(size_t z,size_t y,size_t x)
 {
  if (x>=Size_X || y>=Size_Y) return(0);
  if (z>=Size_Z) return(0);
  return TensorData_Ptr[z*StrideZ+y*StrideX+x];
 }
 __host__ __device__ void SetElement(size_t z,size_t y,size_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y) return;
  if (z>=Size_Z) return;
  TensorData_Ptr[z*StrideZ+y*StrideX+x]=value;
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
  Size_X=0;
  Size_Y=0;
  Size_Z=0;
  StrideX=0;
  StrideZ=0;
  TensorData_Ptr=NULL;
 }

 __host__ void Set(const CTensor<type_t> &cTensor)
 {
  Size_Z=cTensor.Size_Z;
  Size_Y=cTensor.Size_Y;
  Size_X=cTensor.Size_X;
  StrideX=cTensor.Size_X;
  StrideZ=cTensor.Size_X*cTensor.Size_Y;
  TensorData_Ptr=cTensor.DeviceItem.get();
 }
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
//функция CUDA для сложения тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right,type_t left_scale,type_t right_scale)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 size_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_left.GetTensorDataPtr(z)+offset;
 type_t *b_ptr=tensor_right.GetTensorDataPtr(z)+offset;
 type_t *c_ptr=tensor_output.GetTensorDataPtr(z)+offset;

 *c_ptr=(*a_ptr)*left_scale+(*b_ptr)*right_scale;
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
  throw "CTensor::Add: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Left.Size_X/thread.x;
 if (cTensor_Left.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Left.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorAddTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычитания тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSubTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right,type_t left_scale,type_t right_scale)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 size_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_left.GetTensorDataPtr(z)+offset;
 type_t *b_ptr=tensor_right.GetTensorDataPtr(z)+offset;
 type_t *c_ptr=tensor_output.GetTensorDataPtr(z)+offset;

 *c_ptr=(*a_ptr)*left_scale-(*b_ptr)*right_scale;

 __syncthreads();
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
  throw "CTensor::Sub: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 //запускаем процесс

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Left.Size_X/thread.x;
 if (cTensor_Left.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Left.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorSubTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для добавления смещения к элементам тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddBiasFunction(STensorKernel<type_t> tensor_working,STensorKernel<type_t> tensor_bias)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_working.GetSizeX() || yp>=tensor_working.GetSizeY()) return;

 size_t offset=xp+yp*tensor_working.GetSizeX();
 type_t *a_ptr=tensor_working.GetTensorDataPtr(z)+offset;
 type_t *b_ptr=tensor_bias.GetTensorDataPtr(z);

 *a_ptr=(*a_ptr)+(*b_ptr);

 __syncthreads();
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
 cTensor_Bias.CopyToDevice();
 cTensor_Working.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Bias(cTensor_Bias);
 STensorKernel<type_t> sTensorKernel_Working(cTensor_Working);
 //запускаем процесс

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Working.Size_X/thread.x;
 if (cTensor_Working.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Working.Size_Y/thread.y;
 if (cTensor_Working.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Working.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorAddBiasFunction<type_t><<<blocks,thread>>>(sTensorKernel_Working,sTensorKernel_Bias);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Working.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления суммы элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t,uint32_t blockSize>
__global__ void CUDASummXYTensorFunction(int32_t size,STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 size_t z=blockIdx.z;

 size_t tid=threadIdx.x;
 size_t i=blockDim.x*2*blockIdx.x+threadIdx.x;
 size_t gridSize=blockSize*2*gridDim.x;

 __shared__ type_t sdata[blockSize];
 sdata[tid]=0;

 type_t *d_xin=tensor_input.GetTensorDataPtr(z);

 while (i<size)
 {
  if (i+blockDim.x<size) sdata[tid]+=d_xin[i]+d_xin[i+blockDim.x];
  else
  {
   if (i<size) sdata[tid]+=d_xin[i];
  }
  i+=gridSize;
 }
 __syncthreads();

 if (blockSize>=512)
 {
  if (tid<256) sdata[tid]+=sdata[tid+256];
 }
 __syncthreads();
 if (blockSize>=256)
 {
  if (tid<128) sdata[tid]+=sdata[tid+128];
 }
 __syncthreads();
 if (blockSize>=128)
 {
  if (tid<64) sdata[tid]+=sdata[tid+64];
 }
 __syncthreads();

 if (tid<32)
 {
  if (blockSize>=64) sdata[tid]+=sdata[tid+32];
  __syncthreads();
  if (blockSize>=32) sdata[tid]+=sdata[tid+16];
  __syncthreads();
  if (blockSize>=16) sdata[tid]+=sdata[tid+8];
  __syncthreads();
  if (blockSize>=8) sdata[tid]+=sdata[tid+4];
  __syncthreads();
  if (blockSize>=4) sdata[tid]+=sdata[tid+2];
  __syncthreads();
  if (blockSize>=2) sdata[tid]+=sdata[tid+1];
  __syncthreads();
 }

 if (tid==0)
 {
  d_xin[blockIdx.x]=sdata[0];
  if (blockIdx.x==0) *(tensor_output.GetTensorDataPtr(z))=sdata[0];
 }
 __syncthreads();
}


//----------------------------------------------------------------------------------------------------
//вычислить сумму элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SummXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input)
{
 const int32_t B_SIZE=6;
 const int32_t BLOCK_SIZE=1<<B_SIZE;

 if (cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::SummXY: Размерности тензоров не совпадают!";
 }
 cTensor_Input.CopyToDevice();

 size_t input_x=cTensor_Input.GetSizeX();
 size_t input_y=cTensor_Input.GetSizeY();
 size_t input_z=cTensor_Input.GetSizeZ();

 cTensor_Input.ReinterpretSize(input_z,1,input_y*input_x);

 int32_t amount=input_y*input_x;

 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

 int32_t size=amount;
 while (size>1)
 {
  dim3 block((size+BLOCK_SIZE-1)/BLOCK_SIZE,1,input_z);
  dim3 thread(BLOCK_SIZE);

  CUDASummXYTensorFunction<type_t,BLOCK_SIZE><<<block,thread>>>(size,sTensorKernel_Output,sTensorKernel_Input);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
  size=(size+BLOCK_SIZE-1)>>B_SIZE;
 }
 cTensor_Output.SetDeviceOnChange();
 cTensor_Input.ReinterpretSize(input_z,input_y,input_x);
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для умножения тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t,class kernel_output_t,class kernel_left_t,class kernel_right_t>
__global__ void CUDATensorMulTensorFunction(kernel_output_t tensor_output,kernel_left_t tensor_left,kernel_right_t tensor_right)
{
 //блок TENSOR_OPERATION_BLOCK_SIZE x TENSOR_OPERATION_BLOCK_SIZE в выходном тензоре
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подматрицу выходной матрицы
 kernel_output_t Csub=tensor_output.GetSubTensor(z,blockRow,blockCol);

 type_t Cvalue=0;

 size_t m_max=tensor_left.Size_X/CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 if (tensor_left.Size_X%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE) m_max++;

 __shared__ type_t As[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];
 __shared__ type_t Bs[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];

 type_t *a_ptr_init=&As[y][0];
 type_t *b_ptr_init=&Bs[0][x];

 for(size_t m=0;m<m_max;m++)
 {
  // Get sub-tensor Asub of A
  kernel_left_t Asub=tensor_left.GetSubTensor(z,blockRow,m);
  // Get sub-tensor Bsub of B
  kernel_right_t Bsub=tensor_right.GetSubTensor(z,m,blockCol);
  // Shared memory used to store Asub and Bsub respectively
  // Load Asub and Bsub from device memory to shared memory
  // Each thread loads one element of each sub-tensor
  As[y][x]=Asub.GetElement(0,y,x);
  Bs[y][x]=Bsub.GetElement(0,y,x);
  // Synchronize to make sure the sub-matrices are loaded
  // before starting the computation
  __syncthreads();

  type_t *a_ptr=a_ptr_init;
  type_t *b_ptr=b_ptr_init;

  // Multiply Asub and Bsub together
  for(size_t e=0;e<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;e++,a_ptr++,b_ptr+=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE) Cvalue+=(*a_ptr)*(*b_ptr);
  //for(size_t e=0;e<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;e++) Cvalue+=As[y][e]*Bs[e][x];
  // Synchronize to make sure that the preceding
  // computation is done before loading two new
  // sub-matrices of A and B in the next iteration
  __syncthreads();
 }
 // Write Csub to device memory
 // Each thread writes one element
 Csub.SetElement(0,y,x,Cvalue);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//умножить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t> template<class kernel_output_t,class kernel_left_t,class kernel_right_t>
__host__ void CTensorMath<type_t>::MulAbstract(CTensor<type_t> &cTensor_Output,kernel_output_t &sTensorKernel_Output,const CTensor<type_t> &cTensor_Left,kernel_left_t &sTensorKernel_Left,const CTensor<type_t> &cTensor_Right,kernel_right_t &sTensorKernel_Right)
{
 if (sTensorKernel_Left.Size_X!=sTensorKernel_Right.Size_Y  || sTensorKernel_Left.Size_Z!=sTensorKernel_Right.Size_Z ||
     sTensorKernel_Output.Size_Y!=sTensorKernel_Left.Size_Y || sTensorKernel_Output.Size_X!=sTensorKernel_Right.Size_X ||
     sTensorKernel_Output.Size_Z!=sTensorKernel_Right.Size_Z)
 {
  throw "CTensor::MulAbstract: Размерности тензоров не совпадают!";
 }
 //копируем данные с устройство
 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 //разбиваем выходной тензор на блоки по TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE элементов
 //для каждого из этих элементов запускаем по нити (всего TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE нитей)

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=sTensorKernel_Right.Size_X/thread.x;
 if (sTensorKernel_Right.Size_X%thread.x) block_x++;
 size_t block_y=sTensorKernel_Left.Size_Y/thread.y;
 if (sTensorKernel_Left.Size_Y%thread.y) block_y++;
 size_t block_z=sTensorKernel_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorMulTensorFunction<type_t,kernel_output_t,kernel_left_t,kernel_right_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//умножить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);
 MulAbstract<STensorKernel<type_t>,STensorKernel<type_t>,STensorKernel<type_t>>(cTensor_Output,sTensorKernel_Output,cTensor_Left,sTensorKernel_Left,cTensor_Right,sTensorKernel_Right);
}


template<class type_t>
__global__ void CUDATransponseTensorMulTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right)
{
 //блок TENSOR_OPERATION_BLOCK_SIZE x TENSOR_OPERATION_BLOCK_SIZE в выходном тензоре
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;

 //получаем подматрицу выходной матрицы
 STensorKernel<type_t> Csub=tensor_output.GetSubTensor(z,blockRow,blockCol);

 type_t Cvalue=0;

 size_t m_max=tensor_left.Size_Y/CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;
 if (tensor_left.Size_Y%CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE) m_max++;

 __shared__ type_t As[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];
 __shared__ type_t Bs[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE];

 type_t *a_ptr_init=&As[0][y];
 type_t *b_ptr_init=&Bs[0][x];

 for(size_t m=0;m<m_max;m++)
 {

  // Get sub-tensor Asub of A
  STensorKernel<type_t> Asub=tensor_left.GetSubTensor(z,m,blockRow);
  // Get sub-tensor Bsub of B
  STensorKernel<type_t> Bsub=tensor_right.GetSubTensor(z,m,blockCol);
  // Shared memory used to store Asub and Bsub respectively
  // Load Asub and Bsub from device memory to shared memory
  // Each thread loads one element of each sub-tensor
  As[y][x]=Asub.GetElement(0,y,x);
  Bs[y][x]=Bsub.GetElement(0,y,x);
  // Synchronize to make sure the sub-matrices are loaded
  // before starting the computation
  __syncthreads();

  type_t *a_ptr=a_ptr_init;
  type_t *b_ptr=b_ptr_init;

  // Multiply Asub and Bsub together
//  for(size_t e=0;e<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;e++) Cvalue+=As[e][y]*Bs[e][x];
  for(size_t e=0;e<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE;e++,a_ptr+=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,b_ptr+=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE) Cvalue+=(*a_ptr)*(*b_ptr);
  // Synchronize to make sure that the preceding
  // computation is done before loading two new
  // sub-matrices of A and B in the next iteration
  __syncthreads();
 }
 // Write Csub to device memory
 // Each thread writes one element
 Csub.SetElement(0,y,x,Cvalue);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//умножить транспонированный левый тензор на правый
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::TransponseMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Left.Size_Y!=cTensor_Right.Size_Y  || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_Y!=cTensor_Left.Size_X || cTensor_Output.Size_X!=cTensor_Right.Size_X ||
     cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::TransponseMul: Размерности тензоров не совпадают!";
 }

 //копируем данные на устройство
 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 //разбиваем выходной тензор на блоки по TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE элементов
 //для каждого из этих элементов запускаем по нити (всего TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE нитей)

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Right.Size_X/thread.x;
 if (cTensor_Right.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Left.Size_X/thread.y;
 if (cTensor_Left.Size_X%thread.y) block_y++;
 size_t block_z=cTensor_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATransponseTensorMulTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для умножения тензора на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorMulValueFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t value)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 size_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(z)+offset;

 *b_ptr=(*a_ptr)*value;

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//умножить тензор на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const type_t &value_right)
{
 if (cTensor_Output.Size_X!=cTensor_Left.Size_X || cTensor_Output.Size_Y!=cTensor_Left.Size_Y || cTensor_Output.Size_Z!=cTensor_Left.Size_Z)
 {
  throw "CTensor::Mul: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Left);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Left.Size_X/thread.x;
 if (cTensor_Left.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Left.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorMulValueFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,value_right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
//----------------------------------------------------------------------------------------------------
//умножить тензор на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const type_t &value_left,const CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Mul: Размерности тензоров не совпадают!";
 }

 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Right);

 //запускаем процесс

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Right.Size_X/thread.x;
 if (cTensor_Right.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Right.Size_Y/thread.y;
 if (cTensor_Right.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Right.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorMulValueFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,value_left);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}
//----------------------------------------------------------------------------------------------------
//транспонировать тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Transponse(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 if (cTensor_Output.Size_Y!=cTensor_Input.Size_X || cTensor_Output.Size_X!=cTensor_Input.Size_Y || cTensor_Output.Size_Z!=cTensor_Input.Size_Z)
 {
  throw "void CTensor::Transponse: Размерности матриц не совпадают!";
 }
 cTensor_Input.CopyFromDevice(true);

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
 cTensor_Output.SetHostOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления скалярного произведения строк тензора между собой
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorItemProductionFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 size_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_left.GetTensorDataPtr(z)+offset;
 type_t *b_ptr=tensor_right.GetTensorDataPtr(z)+offset;
 type_t *c_ptr=tensor_output.GetTensorDataPtr(z)+offset;

 *c_ptr=(*a_ptr)*(*b_ptr);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//поэлементное произведение тензора на тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::TensorItemProduction(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Left,CTensor<type_t> &cTensor_Right)
{
 if (cTensor_Right.Size_X!=cTensor_Left.Size_X || cTensor_Right.Size_Y!=cTensor_Left.Size_Y || cTensor_Right.Size_Z!=cTensor_Left.Size_Z) throw("Ошибка поэлементного умножения тензора на тензор");
 if (cTensor_Right.Size_X!=cTensor_Output.Size_X || cTensor_Right.Size_Y!=cTensor_Output.Size_Y || cTensor_Right.Size_Z!=cTensor_Output.Size_Z) throw("Ошибка поэлементного умножения тензора на тензор");

 //копируем данные на устройство
 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 //разбиваем выходную матрицы тензора на блоки по TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE элементов
 //для каждого из этих элементов запускаем по нити (всего TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE нитей)

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Right.Size_X/thread.x;
 if (cTensor_Right.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorItemProductionFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
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
//функция CUDA для увеличения разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAUpSamplingTensor(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,size_t upsampling_x,size_t upsampling_y)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 size_t ixp=xp/upsampling_x;
 size_t iyp=yp/upsampling_y;

 size_t offset_output=xp+yp*tensor_output.GetSizeX();
 type_t *o_ptr=tensor_output.GetTensorDataPtr(z)+offset_output;

 if (ixp>=tensor_input.GetSizeX() || iyp>=tensor_input.GetSizeY())
 {
  *o_ptr=0;
  __syncthreads();
  return;
 }

 size_t offset_input=ixp+iyp*tensor_input.GetSizeX();

 type_t *i_ptr=tensor_input.GetTensorDataPtr(z)+offset_input;

 *o_ptr=*i_ptr;

 __syncthreads();
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

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDAUpSamplingTensor<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,upsampling_x,upsampling_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для уменьшение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADownSamplingTensor(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,size_t downsampling_x,size_t downsampling_y)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 size_t ixp=xp*downsampling_x;
 size_t iyp=yp*downsampling_y;

 size_t offset_output=xp+yp*tensor_output.GetSizeX();
 size_t offset_input=ixp+iyp*tensor_input.GetSizeX();

 type_t *i_ptr=tensor_input.GetTensorDataPtr(z)+offset_input;
 type_t *o_ptr=tensor_output.GetTensorDataPtr(z)+offset_output;

 type_t summ=0;
 for(size_t y=0;y<downsampling_y;y++,i_ptr+=tensor_input.StrideX)
 {
  type_t *i_ptr_local=i_ptr;
  for(size_t x=0;x<downsampling_x;x++,i_ptr_local++)
  {
   summ+=*i_ptr_local;
  }
 }

 *o_ptr=summ/static_cast<type_t>(downsampling_x*downsampling_y);

 __syncthreads();
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

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDADownSamplingTensor<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,downsampling_x,downsampling_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для уменьшение разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMaxPoolingTensor(STensorKernel<type_t> tensor_output,STensorKernel<typename CTensorMath<type_t>::SPos> tensor_position,STensorKernel<type_t> tensor_input,size_t pooling_x,size_t pooling_y)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 size_t ixp=xp*pooling_x;
 size_t iyp=yp*pooling_y;

 size_t offset_output=xp+yp*tensor_output.GetSizeX();
 size_t offset_input=ixp+iyp*tensor_input.GetSizeX();

 type_t *i_ptr=tensor_input.GetTensorDataPtr(z)+offset_input;
 type_t *o_ptr=tensor_output.GetTensorDataPtr(z)+offset_output;

 type_t max_e=*i_ptr;
 size_t max_x=ixp;
 size_t max_y=iyp;
 for(size_t y=0;y<pooling_y;y++,i_ptr+=tensor_input.StrideX)
 {
  type_t *i_ptr_local=i_ptr;
  for(size_t x=0;x<pooling_x;x++,i_ptr_local++)
  {
   type_t e=*i_ptr_local;
   if (e>max_e)
   {
    max_e=e;
    max_x=ixp+x;
    max_y=iyp+y;
   }
  }
 }

 *o_ptr=max_e;
 typename CTensorMath<type_t>::SPos sPos;
 sPos.X=max_x;
 sPos.Y=max_y;
 tensor_position.SetElement(z,yp,xp,sPos);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!увеличение разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPooling(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,size_t pooling_x,size_t pooling_y)
{
 if ((cTensor_Input.Size_X/pooling_x)!=cTensor_Output.Size_X || (cTensor_Input.Size_Y/pooling_y)!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::MaxPooling: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_Position.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<SPos> sTensorKernel_Position(cTensor_Position);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDAMaxPoolingTensor<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Position,sTensorKernel_Input,pooling_x,pooling_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
 cTensor_Position.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для обратного прохода при уменьшении разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMaxPoolingTensorBackward(STensorKernel<type_t> tensor_output,STensorKernel<typename CTensorMath<type_t>::SPos> tensor_position,STensorKernel<type_t> tensor_input,size_t pooling_x,size_t pooling_y)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 size_t ixp=xp/pooling_x;
 size_t iyp=yp/pooling_y;

 type_t value=tensor_input.GetElement(z,iyp,ixp);
 typename CTensorMath<type_t>::SPos sPos=tensor_position.TensorData_Ptr[z*tensor_position.StrideZ+iyp*tensor_position.StrideX+ixp];
 if (sPos.X!=xp || sPos.Y!=yp) value=0;
 tensor_output.SetElement(z,yp,xp,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!обратный проход при увеличении разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,size_t pooling_x,size_t pooling_y)
{
 if (cTensor_Input.Size_X!=(cTensor_Output.Size_X/pooling_x) || cTensor_Input.Size_Y!=(cTensor_Output.Size_Y/pooling_y) || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::MaxPooling: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_Position.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<SPos> sTensorKernel_Position(cTensor_Position);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDAMaxPoolingTensorBackward<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Position,sTensorKernel_Input,pooling_x,pooling_y);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
 cTensor_Position.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для выполнения отсечки значений тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAClipTensor(STensorKernel<type_t> tensor,type_t min_value,type_t max_value)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor.GetSizeX() || yp>=tensor.GetSizeY()) return;

 type_t value=tensor.GetElement(z,yp,xp);
 if (value>max_value) value=max_value;
 if (value<min_value) value=min_value;
 tensor.SetElement(z,yp,xp,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!выполнить отсечку значений тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value)
{
 cTensor.CopyToDevice();

 STensorKernel<type_t> sTensorKernel(cTensor);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor.Size_X/thread.x;
 if (cTensor.Size_X%thread.x) block_x++;
 size_t block_y=cTensor.Size_Y/thread.y;
 if (cTensor.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDAClipTensor<type_t><<<blocks,thread>>>(sTensorKernel,min_value,max_value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для выполнения алгоритма Adam
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAAdam(STensorKernel<type_t> tensor_weight,STensorKernel<type_t> tensor_dweight,STensorKernel<type_t> tensor_m,STensorKernel<type_t> tensor_v,double speed,double beta1,double beta2,double epsilon,double iteration)
{
 size_t blockCol=blockIdx.x;
 size_t blockRow=blockIdx.y;
 size_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 size_t x=threadIdx.x;
 size_t y=threadIdx.y;
 //получаем подтензоры
 size_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 size_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_weight.GetSizeX() || yp>=tensor_weight.GetSizeY()) return;

 type_t dw=tensor_dweight.GetElement(z,yp,xp);
 type_t m=tensor_m.GetElement(z,yp,xp);
 type_t v=tensor_v.GetElement(z,yp,xp);

 m=beta1*m+(1.0-beta1)*dw;
 type_t mc=m/(1.0-pow(beta1,iteration));

 v=beta2*v+(1.0-beta2)*dw*dw;
 type_t vc=v/(1.0-pow(beta2,iteration));

 dw=speed*mc/sqrt(vc+epsilon);

 tensor_m.SetElement(z,yp,xp,m);
 tensor_v.SetElement(z,yp,xp,v);

 //корректируем веса
 type_t w=tensor_weight.GetElement(z,yp,xp);
 tensor_weight.SetElement(z,yp,xp,w-dw);

 __syncthreads();
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

 cTensor_Weight.CopyToDevice();
 cTensor_dWeight.CopyToDevice();
 cTensor_M.CopyToDevice();
 cTensor_V.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Weight(cTensor_Weight);
 STensorKernel<type_t> sTensorKernel_dWeight(cTensor_dWeight);
 STensorKernel<type_t> sTensorKernel_M(cTensor_M);
 STensorKernel<type_t> sTensorKernel_V(cTensor_V);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 size_t block_x=cTensor_Weight.Size_X/thread.x;
 if (cTensor_Weight.Size_X%thread.x) block_x++;
 size_t block_y=cTensor_Weight.Size_Y/thread.y;
 if (cTensor_Weight.Size_Y%thread.y) block_y++;
 size_t block_z=cTensor_Weight.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 CUDAAdam<type_t><<<blocks,thread>>>(sTensorKernel_Weight,sTensorKernel_dWeight,sTensorKernel_M,sTensorKernel_V,speed,beta1,beta2,epsilon,iteration);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Weight.SetDeviceOnChange();
 cTensor_M.SetDeviceOnChange();
 cTensor_V.SetDeviceOnChange();
}


#endif

#endif
