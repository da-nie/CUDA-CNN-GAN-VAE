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

 template<class new_type_t>
 friend struct STensorTransponseKernel;
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  struct SPos
  {
   uint32_t X;
   uint32_t Y;
  };
  //-константы------------------------------------------------------------------------------------------
  static const uint32_t TENSOR_OPERATION_BLOCK_SIZE_SCALE=2;///<размер множителя блока операций с тензорами
  static const uint32_t TENSOR_OPERATION_BLOCK_SIZE=16;///<размер блока операций с тензорами
 private:
  //-переменные-----------------------------------------------------------------------------------------
 public:
  //-конструктор----------------------------------------------------------------------------------------
  //-конструктор копирования----------------------------------------------------------------------------
  //-деструктор-----------------------------------------------------------------------------------------
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  static void Fill(CTensor<type_t> &cTensor_Output,type_t value=0);///<записать в тензор число
  static void Inv(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<вычислить обратный тензор
  static void Div(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<поделить тензоры
  static void Add(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<сложить тензоры
  static void AddSumW(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<сложить тензоры по координате W
  static void AddValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value);///<прибавить к каждому элементу тензора число
  static void Sub(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<вычесть тензоры
  static void SubValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value);///<отнять от каждого элемента тензора число
  static void Set(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale=1,type_t add_value=0);///<скопировать тензор с масштабированием
  static void Pow2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale=1);///<возведение элементов тензора в квадрат
  static void SQRT(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_sqrt_value);///<вычисление квадратного корня из элементов тензора
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

  static void UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y);///<увеличение разрешения тензора
  static void DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y);///<уменьшение разрешения тензора

  static void MaxPooling(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y);///<уменьшение разрешения тензора выборкой большего элемента
  static void MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y);///<обратный проход при увеличении разрешения тензора выборкой большего элемента
  static void Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value);///<выполнить отсечку значений тензора

  static void Adam(CTensor<type_t> &cTensor_Weight,CTensor<type_t> &cTensor_dWeight,CTensor<type_t> &cTensor_M,CTensor<type_t> &cTensor_V,uint32_t batch_size,double speed,double beta1,double beta2,double epsilon,double iteration);///<выполнить алгоритм Adam к весовому тензору


 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};


__forceinline__ __host__ __device__ uint32_t Mod(uint32_t param,uint32_t divider)
{
 uint32_t div=param/divider;
 return(param-div*divider);
}

//****************************************************************************************************
///!структура ядра тензора
//****************************************************************************************************
template<class type_t>
struct STensorKernel
{
 uint32_t Size_X;///<размер по x
 uint32_t Size_Y;///<размер по y
 uint32_t Size_Z;///<размер по z
 uint32_t Size_W;///<размер по w
 uint32_t StrideX;///<строка по X
 uint32_t StrideZ;///<размер блока по Z
 uint32_t StrideW;///<размер блока по W
 type_t *TensorData_Ptr;///<указатель на данные тензора на стороне GPU

 uint32_t SelectedZ;///<выбранный слой Z
 uint32_t SelectedW;///<выбранный слой W
 type_t *TensorData_WZ_Ptr;///<указатель на данные тензора на стороне GPU выбранного слоя Z и W
 type_t *TensorData_W_Ptr;///<указатель на данные тензора на стороне GPU выбранного слоя W

 __host__ __device__ STensorKernel(void)///<конструктор
 {
 }
 __host__ __device__ STensorKernel(const CTensor<type_t> &cTensor)///<конструктор
 {
  Set(cTensor);
 }

 __forceinline__ __host__ __device__ type_t* GetTensorDataPtr(uint32_t w,uint32_t z)///<получить указатель на элементы с глубиной z
 {
  return(&TensorData_Ptr[z*StrideZ+w*StrideW]);
 }

 __forceinline__ __host__ __device__ type_t GetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x)
 {
  if (x>=Size_X || y>=Size_Y || z>=Size_Z || w>=Size_W) return(0);
  return(TensorData_Ptr[w*StrideW+z*StrideZ+y*StrideX+x]);
 }
 __forceinline__ __host__ __device__ void SetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y || z>=Size_Z || w>=Size_W) return;
  TensorData_Ptr[w*StrideW+z*StrideZ+y*StrideX+x]=value;
 }

 __forceinline__ __host__ __device__ uint32_t GetSizeX(void) const
 {
  return(Size_X);
 }

 __forceinline__ __host__ __device__ uint32_t GetSizeY(void) const
 {
  return(Size_Y);
 }

 __forceinline__ __host__ __device__ uint32_t GetSizeZ(void) const
 {
  return(Size_Z);
 }

 __forceinline__ __host__ __device__ uint32_t GetSizeW(void) const
 {
  return(Size_W);
 }

 __forceinline__ __host__ __device__ void SelectW(uint32_t w)///<выбрать слой W
 {
  SelectedW=w;
  TensorData_WZ_Ptr=TensorData_Ptr+SelectedZ*StrideZ+w*StrideW;
  TensorData_W_Ptr=TensorData_Ptr+w*StrideW;
 }

 __forceinline__ __host__ __device__ void SelectZ(uint32_t z)///<выбрать слой Z
 {
  SelectedZ=z;
  TensorData_WZ_Ptr=TensorData_W_Ptr+z*StrideZ;
 }

 __forceinline__ __host__ __device__ type_t GetElement(uint32_t y,uint32_t x)
 {
  if (x>=Size_X || y>=Size_Y) return(0);
  return(TensorData_WZ_Ptr[y*StrideX+x]);
 }
 __forceinline__ __host__ __device__ void SetElement(uint32_t y,uint32_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y) return;
  TensorData_WZ_Ptr[y*StrideX+x]=value;
 }

 __forceinline__ __host__ __device__ type_t GetElement(uint32_t z,uint32_t y,uint32_t x)
 {
  if (x>=Size_X || y>=Size_Y || z>=Size_Z) return(0);
  return(TensorData_W_Ptr[z*StrideZ+y*StrideX+x]);
 }
 __forceinline__ __host__ __device__ void SetElement(uint32_t z,uint32_t y,uint32_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y || z>=Size_Z) return;
  TensorData_W_Ptr[z*StrideZ+y*StrideX+x]=value;
 }


 __host__ __device__ void Reset(void)
 {
  Size_X=0;
  Size_Y=0;
  Size_Z=0;
  Size_W=0;
  StrideX=0;
  StrideZ=0;
  StrideW=0;
  SelectedZ=0;
  SelectedW=0;
  TensorData_Ptr=NULL;
  TensorData_W_Ptr=NULL;
  TensorData_WZ_Ptr=NULL;
 }

 __host__ __device__ void Set(const CTensor<type_t> &cTensor)
 {
  Size_W=cTensor.Size_W;
  Size_Z=cTensor.Size_Z;
  Size_Y=cTensor.Size_Y;
  Size_X=cTensor.Size_X;
  StrideX=cTensor.Size_X;
  StrideZ=cTensor.Size_X*cTensor.Size_Y;
  StrideW=cTensor.Size_X*cTensor.Size_Y*cTensor.Size_Z;
  TensorData_Ptr=cTensor.DeviceItem.get();
  TensorData_W_Ptr=TensorData_Ptr;
  TensorData_WZ_Ptr=TensorData_Ptr;
  SelectedZ=0;
  SelectedW=0;
 }
};

//****************************************************************************************************
///!структура ядра транспонированного тензора
//****************************************************************************************************
template<class type_t>
struct STensorTransponseKernel
{
 uint32_t Size_X;///<размер по x
 uint32_t Size_Y;///<размер по y
 uint32_t Size_Z;///<размер по z
 uint32_t Size_W;///<размер по w
 uint32_t StrideX;///<строка по X
 uint32_t StrideZ;///<размер блока по Z
 uint32_t StrideW;///<размер блока по W
 type_t *TensorData_Ptr;///<указатель на данные тензора на стороне GPU

 uint32_t SelectedZ;///<выбранный слой Z
 uint32_t SelectedW;///<выбранный слой W
 type_t *TensorData_WZ_Ptr;///<указатель на данные тензора на стороне GPU выбранного слоя Z и W
 type_t *TensorData_W_Ptr;///<указатель на данные тензора на стороне GPU выбранного слоя W

 __host__ __device__ STensorTransponseKernel(void)///<конструктор
 {
 }
 __host__ __device__ STensorTransponseKernel(const CTensor<type_t> &cTensor)///<конструктор
 {
  Set(cTensor);
 }

 __host__ __device__ type_t* GetTensorDataPtr(uint32_t w,uint32_t z)///<получить указатель на элементы с глубиной z
 {
  return(&TensorData_Ptr[w*StrideW+z*StrideZ]);
 }

 __host__ __device__ STensorTransponseKernel<type_t> GetSubTensor(uint32_t w,uint32_t z,uint32_t y,uint32_t x)///<получить подтензор с глубиной 1
 {
  STensorTransponseKernel<type_t> sub_tensor;

  sub_tensor.Size_X=(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE);
  sub_tensor.Size_Y=(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE);
  sub_tensor.Size_Z=1;
  sub_tensor.Size_W=Size_W;
  sub_tensor.StrideX=StrideX;
  sub_tensor.StrideZ=0;
  sub_tensor.StrideW=0;

  sub_tensor.TensorData_Ptr=&TensorData_Ptr[w*StrideW+z*StrideZ+StrideX*(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE)*x+(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE)*y];

  //условие не строгое, так как последний блок для матриц не кратных блоку гарантировано будет превышать размер матрицы.
  if ((x+1)*(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE)>Size_X) sub_tensor.Size_X=Size_X%(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE);
  if ((y+1)*(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE)>Size_Y) sub_tensor.Size_Y=Size_Y%(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE);

  return(sub_tensor);
 }

 __host__ __device__ type_t GetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x)
 {
  if (x>=Size_Y || y>=Size_X || z>=Size_Z || z>=Size_W) return(0);
  return TensorData_Ptr[w*StrideW+z*StrideZ+x*StrideX+y];
 }
 __host__ __device__ void SetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x,type_t value)
 {
  if (x>=Size_Y || y>=Size_X || z>=Size_Z || w>=Size_W) return;
  TensorData_Ptr[w*StrideW+z*StrideZ+x*StrideX+y]=value;
 }

 __host__ __device__ uint32_t GeTransponsetSizeX(void)
 {
  return(Size_X);
 }

 __host__ __device__ uint32_t GetSizeY(void)
 {
  return(Size_Y);
 }

 __host__ __device__ uint32_t GetSizeZ(void)
 {
  return(Size_Z);
 }

 __host__ __device__ uint32_t GetSizeW(void)
 {
  return(Size_W);
 }

 __forceinline__ __host__ __device__ void SelectW(uint32_t w)///<выбрать слой W
 {
  SelectedW=w;
  TensorData_WZ_Ptr=TensorData_Ptr+SelectedZ*StrideZ+w*StrideW;
  TensorData_W_Ptr=TensorData_Ptr+w*StrideW;
 }

 __forceinline__ __host__ __device__ void SelectZ(uint32_t z)///<выбрать слой Z
 {
  SelectedZ=z;
  TensorData_WZ_Ptr=TensorData_W_Ptr+z*StrideZ;
 }

__forceinline__ __host__ __device__ type_t GetElement(uint32_t y,uint32_t x)
 {
  if (x>=Size_Y || y>=Size_X) return(0);
  return(TensorData_WZ_Ptr[x*StrideX+y]);
 }
 __forceinline__ __host__ __device__ void SetElement(uint32_t y,uint32_t x,type_t value)
 {
  if (x>=Size_Y || y>=Size_X) return;
  TensorData_WZ_Ptr[x*StrideX+y]=value;
 }

 __forceinline__ __host__ __device__ type_t GetElement(uint32_t z,uint32_t y,uint32_t x)
 {
  if (x>=Size_Y || y>=Size_X || z>=Size_Z) return(0);
  return(TensorData_W_Ptr[z*StrideZ+x*StrideX+y]);
 }
 __forceinline__ __host__ __device__ void SetElement(uint32_t z,uint32_t y,uint32_t x,type_t value)
 {
  if (x>=Size_Y || y>=Size_X || z>=Size_Z) return;
  TensorData_W_Ptr[z*StrideZ+x*StrideX+y]=value;
 }

 __host__ __device__ void Reset(void)
 {
  Size_X=0;
  Size_Y=0;
  Size_Z=0;
  Size_W=0;
  StrideX=0;
  StrideZ=0;
  StrideW=0;
  TensorData_Ptr=NULL;
  TensorData_W_Ptr=NULL;
  TensorData_WZ_Ptr=NULL;
 }

 __host__ __device__ void Set(const CTensor<type_t> &cTensor)
 {
  Size_W=cTensor.Size_W;
  Size_Z=cTensor.Size_Z;
  Size_Y=cTensor.Size_X;
  Size_X=cTensor.Size_Y;
  StrideX=cTensor.Size_X;
  StrideZ=cTensor.Size_X*cTensor.Size_Y;
  StrideW=cTensor.Size_X*cTensor.Size_Y*cTensor.Size_Z;
  TensorData_Ptr=cTensor.DeviceItem.get();
  TensorData_W_Ptr=TensorData_Ptr;
  TensorData_WZ_Ptr=TensorData_Ptr;
  SelectedZ=0;
  SelectedW=0;
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
 CTensor<type_t> cTensor(cTensor_Right.Size_W,cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Left.Size_X);
 CTensorMath<type_t>::Add(cTensor,cTensor_Left,cTensor_Right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "-"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator-(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Right.Size_W,cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Left.Size_X);
 CTensorMath<type_t>::Sub(cTensor,cTensor_Left,cTensor_Right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator*(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Right.Size_W,cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Right.Size_X);
 CTensorMath<type_t>::Mul(cTensor,cTensor_Left,cTensor_Right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator*(const CTensor<type_t> &cTensor_Left,const type_t &value_right)
{
 CTensor<type_t> cTensor(cTensor_Left.Size_W,cTensor_Left.Size_Z,cTensor_Left.Size_Y,cTensor_Left.Size_X);
 CTensorMath<type_t>::Mul(cTensor,cTensor_Left,value_right);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t> operator*(const type_t &value_left,const CTensor<type_t> &cTensor_Right)
{
 CTensor<type_t> cTensor(cTensor_Right.Size_W,cTensor_Right.Size_Z,cTensor_Right.Size_Y,cTensor_Right.Size_X);
 CTensorMath<type_t>::Mul(cTensor,value_left,cTensor_Right);
 return(cTensor);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//функция CUDA для записи числа в тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorFillFunction(STensorKernel<type_t> tensor_output,type_t value)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w,z)+offset;
 *out_ptr=value;
}

//----------------------------------------------------------------------------------------------------
//записать в тензор число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Fill(CTensor<type_t> &cTensor_Output,type_t value)
{
 //return;//TODO: тест

 if (cTensor_Output.GetSizeX()*cTensor_Output.GetSizeY()*cTensor_Output.GetSizeZ()<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE)
 {
  cTensor_Output.Fill(value);
  return;
 }

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorFillFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления обратного тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorInvTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;
 type_t e=(*in_ptr);

 *out_ptr=1.0/e;
}

//----------------------------------------------------------------------------------------------------
//вычисление обратного тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Inv(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 //return;//TODO: тест
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::Inv: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorInvTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для деления тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorDivTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right,type_t left_scale,type_t right_scale)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_left=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_left.GetSizeW());
 uint32_t w_right=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_right.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_left.GetTensorDataPtr(w_left,z)+offset;
 type_t *b_ptr=tensor_right.GetTensorDataPtr(w_right,z)+offset;
 type_t *c_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *c_ptr=(*a_ptr)*left_scale/((*b_ptr)*right_scale);
}

//----------------------------------------------------------------------------------------------------
//поделить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Div(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale,type_t right_scale)
{
 //return;//TODO: тест
 if (cTensor_Left.Size_X!=cTensor_Right.Size_X || cTensor_Left.Size_Y!=cTensor_Right.Size_Y || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Div: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Left.Size_X/thread.x;
 if (cTensor_Left.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorDivTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для сложения тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right,type_t left_scale,type_t right_scale)
{
 //блок TENSOR_OPERATION_BLOCK_SIZE x TENSOR_OPERATION_BLOCK_SIZE в выходном тензоре
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_left=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_left.GetSizeW());
 uint32_t w_right=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_right.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t ox=threadIdx.x*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE+blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;
 uint32_t oy=threadIdx.y*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE+blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;

 tensor_left.SelectW(w_left);
 tensor_left.SelectZ(z);
 tensor_right.SelectW(w_right);
 tensor_right.SelectZ(z);
 tensor_output.SelectW(w_out);
 tensor_output.SelectZ(z);

 for(uint32_t kx=0;kx<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;kx++)
 {
  uint32_t x=kx+ox;
  for(uint32_t ky=0;ky<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;ky++)
  {
   uint32_t y=ky+oy;
   type_t left=tensor_left.GetElement(y,x);
   type_t right=tensor_right.GetElement(y,x);
   type_t v=left*left_scale+right*right_scale;
   tensor_output.SetElement(y,x,v);
  }
 }
 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//сложить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Add(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale,type_t right_scale)
{
 //return;//TODO: тест
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

 //разбиваем выходной тензор на блоки по TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE элементов
 //для каждого из этих элементов запускаем по нити (всего TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE нитей)

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE);

 uint32_t block_x=sTensorKernel_Output.Size_X/thread.x;
 if (sTensorKernel_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=sTensorKernel_Output.Size_Y/thread.y;
 if (sTensorKernel_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=sTensorKernel_Output.Size_Z*sTensorKernel_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 dim3 thread_basic(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 CUDATensorAddTensorFunction<type_t><<<blocks,thread_basic>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для сложения тензоров по координате W
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddSumWTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right,type_t left_scale,type_t right_scale)
{
 //блок TENSOR_OPERATION_BLOCK_SIZE x TENSOR_OPERATION_BLOCK_SIZE в выходном тензоре
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 uint32_t ox=threadIdx.x*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE+blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;
 uint32_t oy=threadIdx.y*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE+blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;

 for(uint32_t kx=0;kx<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;kx++)
 {
  uint32_t x=kx+ox;
  for(uint32_t ky=0;ky<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;ky++)
  {
   uint32_t y=ky+oy;
   type_t summ_left=0;
   type_t summ_right=0;
   for(uint32_t w=0;w<tensor_left.GetSizeW();w++)
   {
    type_t v=tensor_left.GetElement(w,z,y,x);
    summ_left+=v;
   }
   summ_left*=left_scale;
   for(uint32_t w=0;w<tensor_right.GetSizeW();w++)
   {
    type_t v=tensor_right.GetElement(w,z,y,x);
    summ_right+=v;
   }
   summ_right*=right_scale;
   type_t summ_output=summ_left+summ_right;
   tensor_output.SetElement(0,z,y,x,summ_output);//помещаем сумму в нулевой слой w
   for(uint32_t w=1;w<tensor_output.GetSizeW();w++) tensor_output.SetElement(w,z,y,x,0);//все остальные слои w обнулены
  }
 }
 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//сложить тензоры по координате W
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::AddSumW(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale,type_t right_scale)
{
 //return;//TODO: тест
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

 //разбиваем выходной тензор на блоки по TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE элементов
 //для каждого из этих элементов запускаем по нити (всего TENSOR_OPERATION_BLOCK_SIZExTENSOR_OPERATION_BLOCK_SIZE нитей)

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE);

 uint32_t block_x=sTensorKernel_Output.Size_X/thread.x;
 if (sTensorKernel_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=sTensorKernel_Output.Size_Y/thread.y;
 if (sTensorKernel_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=sTensorKernel_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 dim3 thread_basic(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 CUDATensorAddSumWTensorFunction<type_t><<<blocks,thread_basic>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}





//----------------------------------------------------------------------------------------------------
//функция CUDA для прибавления числа к элементам тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddValueTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t tensor_scale,type_t value)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *out_ptr=(*in_ptr)*tensor_scale+value;
}

//----------------------------------------------------------------------------------------------------
//прибавить к каждому элементу тензора число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::AddValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value)
{
 //return;//TODO: тест
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::AddValue: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorAddValueTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,tensor_scale,value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//вычесть тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Sub(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale,type_t right_scale)
{
 //return;//TODO: тест
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

 uint32_t block_x=cTensor_Left.Size_X/thread.x;
 if (cTensor_Left.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorAddTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,-right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//отнять от каждого элемента тензора число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SubValue(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t tensor_scale,type_t value)
{
 //return;//TODO: тест
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::SubValue: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorAddValueTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,tensor_scale,-value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для копирования с масштабированием тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSetTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t scale,type_t add_value)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *out_ptr=(*in_ptr)*scale+add_value;

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//скопировать тензор с масштабированием
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Set(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_value)
{
 //return;//TODO: тест
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::Set: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorSetTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,scale,add_value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}



//----------------------------------------------------------------------------------------------------
//функция CUDA для возведение элементов тензора в квадрат
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorPow2TensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t scale)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;
 type_t e=(*in_ptr);

 *out_ptr=e*e*scale;
}

//----------------------------------------------------------------------------------------------------
//возведение элементов тензора в квадрат
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Pow2(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale)
{
 //return;//TODO: тест
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::Pow2: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorPow2TensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисление квадратного корня из элементов тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSQRTTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t scale,type_t add_sqrt_value)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *in_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *out_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;
 type_t e=(*in_ptr);

 *out_ptr=(sqrt(e+add_sqrt_value))*scale;
}

//----------------------------------------------------------------------------------------------------
//вычисление квадратного корня из элементов тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SQRT(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,type_t scale,type_t add_sqrt_value)
{
 //return;//TODO: тест
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::SQRT: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorSQRTTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,scale,add_sqrt_value);
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
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_working.GetSizeZ());
 uint32_t w_bias=Mod((blockIdx.z/tensor_working.GetSizeZ()),tensor_bias.GetSizeW());
 uint32_t w_working=Mod((blockIdx.z/tensor_working.GetSizeZ()),tensor_working.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_working.GetSizeX() || yp>=tensor_working.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_working.GetSizeX();
 type_t *a_ptr=tensor_working.GetTensorDataPtr(w_working,z)+offset;
 type_t *b_ptr=tensor_bias.GetTensorDataPtr(w_bias,z);

 *a_ptr=(*a_ptr)+(*b_ptr);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//добавить смещения к элементам тензора (смещения одинаковы для x и y, но по z смещения разные)
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::AddBias(CTensor<type_t> &cTensor_Working,const CTensor<type_t> &cTensor_Bias)
{
 //return;//TODO: тест
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

 uint32_t block_x=cTensor_Working.Size_X/thread.x;
 if (cTensor_Working.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Working.Size_Y/thread.y;
 if (cTensor_Working.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Working.Size_Z*cTensor_Working.Size_W;

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
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());

 uint32_t tid=threadIdx.x;
 uint32_t i=blockDim.x*2*blockIdx.x+threadIdx.x;
 uint32_t gridSize=blockSize*2*gridDim.x;

 __shared__ type_t sdata[blockSize];
 sdata[tid]=0;

 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z);

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
  if (blockIdx.x==0) *(tensor_output.GetTensorDataPtr(w_out,z))=sdata[0];
 }

 __syncthreads();
}


//----------------------------------------------------------------------------------------------------
//вычислить сумму элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::SummXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input)
{
 //return;//TODO: тест

 const int32_t B_SIZE=6;
 const int32_t BLOCK_SIZE=1<<B_SIZE;

 if (cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::SummXY: Размерности тензоров не совпадают!";
 }
 cTensor_Input.CopyToDevice();

 uint32_t input_x=cTensor_Input.GetSizeX();
 uint32_t input_y=cTensor_Input.GetSizeY();
 uint32_t input_z=cTensor_Input.GetSizeZ();
 uint32_t input_w=cTensor_Input.GetSizeW();

 cTensor_Input.ReinterpretSize(input_w,input_z,1,input_y*input_x);

 int32_t amount=input_y*input_x;

 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

 int32_t size=amount;
 while (size>1)
 {
  dim3 block((size+BLOCK_SIZE-1)/BLOCK_SIZE,1,cTensor_Output.GetSizeZ()*cTensor_Output.GetSizeW());
  dim3 thread(BLOCK_SIZE);

  CUDASummXYTensorFunction<type_t,BLOCK_SIZE><<<block,thread>>>(size,sTensorKernel_Output,sTensorKernel_Input);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
  size=(size+BLOCK_SIZE-1)>>B_SIZE;
 }
 cTensor_Output.SetDeviceOnChange();
 cTensor_Input.ReinterpretSize(input_w,input_z,input_y,input_x);
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для умножения тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t,class kernel_output_t,class kernel_left_t,class kernel_right_t>
__global__ void CUDATensorMulTensorFunction(kernel_output_t tensor_output,kernel_left_t tensor_left,kernel_right_t tensor_right)
{
 //блок CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE x CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE в выходном тензоре
 uint32_t block_x=blockIdx.x;
 uint32_t block_y=blockIdx.y;
 //координаты элементов блока в выходном тензоре
 uint32_t out_in_block_x=threadIdx.x*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;
 uint32_t out_in_block_y=threadIdx.y*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;
 //координаты блока в выходном тензоре
 uint32_t out_block_x=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE*block_x;
 uint32_t out_block_y=CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE*block_y;
 //глобальные координаты в выходном тензоре
 uint32_t out_x=out_in_block_x+out_block_x;
 uint32_t out_y=out_in_block_y+out_block_y;
 uint32_t out_z=Mod(blockIdx.z,tensor_output.GetSizeZ());

 uint32_t w_left=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_left.GetSizeW());
 uint32_t w_right=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_right.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());

 //получаем подматрицу выходной матрицы
 type_t Cvalue[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE];
 for(uint32_t ky=0;ky<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;ky++)
 {
  for(uint32_t kx=0;kx<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;kx++)
  {
   Cvalue[ky][kx]=0;
  }
 }

 tensor_left.SelectW(w_left);
 tensor_right.SelectW(w_right);
 tensor_output.SelectW(w_out);

 tensor_left.SelectZ(out_z);
 tensor_right.SelectZ(out_z);
 tensor_output.SelectZ(out_z);

 //считаем, сколькно нужно проходов блоком по X
 uint32_t m_max=tensor_left.Size_X/(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE);
 if (tensor_left.Size_X%(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE)) m_max++;

 __shared__ type_t As[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE];
 __shared__ type_t Bs[CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE][CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE];

 for(uint32_t m=0;m<m_max;m++)
 {
  uint32_t offset=m*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;
  uint32_t py_left=out_in_block_y+out_block_y;
  uint32_t py_right=out_in_block_y+offset;
  uint32_t py=out_in_block_y;
  for(uint32_t ky=0;ky<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;ky++,py_left++,py_right++,py++)
  {
   uint32_t px_left=out_in_block_x+offset;
   uint32_t px_right=out_in_block_x+out_block_x;
   uint32_t px=out_in_block_x;
   for(uint32_t kx=0;kx<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;kx++,px_left++,px_right++,px++)
   {
    As[py][px]=tensor_left.GetElement(py_left,px_left);
    Bs[py][px]=tensor_right.GetElement(py_right,px_right);
   }
  }
  __syncthreads();

  //выполняем умножение только для тех элементов, которые не выходят за размер выходного тензора
  if ((out_y<tensor_output.Size_Y) &&
      (out_x<tensor_output.Size_X))
  {

   for(uint32_t ky=0;ky<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;ky++)
   {
    for(uint32_t kx=0;kx<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;kx++)
    {
     type_t &cv=Cvalue[ky][kx];
     uint32_t ay=out_in_block_y+ky;
     uint32_t bx=out_in_block_x+kx;
     for(uint32_t e=0;e<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;e++) cv+=As[ay][e]*Bs[e][bx];
    }
   }
  }
  __syncthreads();
 }
 uint32_t py=out_y;
 for(uint32_t ky=0;ky<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;ky++,py++)
 {
  uint32_t px=out_x;
  for(uint32_t kx=0;kx<CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE;kx++,px++)
  {
   tensor_output.SetElement(py,px,Cvalue[ky][kx]);
  }
 }
 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//умножить тензоры
//----------------------------------------------------------------------------------------------------
template<class type_t> template<class kernel_output_t,class kernel_left_t,class kernel_right_t>
__host__ void CTensorMath<type_t>::MulAbstract(CTensor<type_t> &cTensor_Output,kernel_output_t &sTensorKernel_Output,const CTensor<type_t> &cTensor_Left,kernel_left_t &sTensorKernel_Left,const CTensor<type_t> &cTensor_Right,kernel_right_t &sTensorKernel_Right)
{
 //return;//TODO: тест

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

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE_SCALE);

 uint32_t block_x=sTensorKernel_Output.Size_X/thread.x;
 if (sTensorKernel_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=sTensorKernel_Output.Size_Y/thread.y;
 if (sTensorKernel_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=sTensorKernel_Output.Size_Z*sTensorKernel_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 dim3 thread_basic(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 CUDATensorMulTensorFunction<type_t,kernel_output_t,kernel_left_t,kernel_right_t><<<blocks,thread_basic>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right);
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

//----------------------------------------------------------------------------------------------------
//умножить транспонированный левый тензор на правый
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::TransponseMul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right)
{
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorTransponseKernel<type_t> sTensorTransponseKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);
 MulAbstract<STensorKernel<type_t>,STensorTransponseKernel<type_t>,STensorKernel<type_t>>(cTensor_Output,sTensorKernel_Output,cTensor_Left,sTensorTransponseKernel_Left,cTensor_Right,sTensorKernel_Right);
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для умножения тензора на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorMulValueFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t value)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset;
 type_t *b_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *b_ptr=(*a_ptr)*value;

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//умножить тензор на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Mul(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const type_t &value_right)
{
 //return;//TODO: тест
 if (cTensor_Output.Size_X!=cTensor_Left.Size_X || cTensor_Output.Size_Y!=cTensor_Left.Size_Y || cTensor_Output.Size_Z!=cTensor_Left.Size_Z)
 {
  throw "CTensor::Mul: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Left);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Left.Size_X/thread.x;
 if (cTensor_Left.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

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
 //return;//TODO: тест
 if (cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::Mul: Размерности тензоров не совпадают!";
 }

 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Right);

 //запускаем процесс

 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Right.Size_X/thread.x;
 if (cTensor_Right.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Right.Size_Y/thread.y;
 if (cTensor_Right.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

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
 //return;//TODO: тест
 if (cTensor_Output.Size_Y!=cTensor_Input.Size_X || cTensor_Output.Size_X!=cTensor_Input.Size_Y || cTensor_Output.Size_Z!=cTensor_Input.Size_Z || cTensor_Output.Size_W!=cTensor_Input.Size_W)
 {
  throw "void CTensor::Transponse: Размерности матриц не совпадают!";
 }
 cTensor_Input.CopyFromDevice(true);

 for(uint32_t w=0;w<cTensor_Output.Size_W;w++)
 {
  for(uint32_t z=0;z<cTensor_Input.Size_Z;z++)
  {
   const type_t *i_ptr=cTensor_Input.GetColumnPtr(w,z,0);
   type_t *o_ptr=cTensor_Output.GetColumnPtr(w,z,0);
   for(uint32_t y=0;y<cTensor_Input.Size_Y;y++,o_ptr++)
   {
    type_t *o_ptr_local=o_ptr;
    for(uint32_t x=0;x<cTensor_Input.Size_X;x++,o_ptr_local+=cTensor_Input.Size_Y,i_ptr++)
    {
     *o_ptr_local=*i_ptr;
    }
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
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_left=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_left.GetSizeW());
 uint32_t w_right=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_right.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензорыuint32_t w_left=(blockIdx.z/tensor_output.GetSizeZ())%tensor_right.GetSizeW();
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t offset=xp+yp*tensor_output.GetSizeX();
 type_t *a_ptr=tensor_left.GetTensorDataPtr(w_left,z)+offset;
 type_t *b_ptr=tensor_right.GetTensorDataPtr(w_right,z)+offset;
 type_t *c_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset;

 *c_ptr=(*a_ptr)*(*b_ptr);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//поэлементное произведение тензора на тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::TensorItemProduction(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Left,CTensor<type_t> &cTensor_Right)
{
 //return;//TODO: тест
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

 uint32_t block_x=cTensor_Right.Size_X/thread.x;
 if (cTensor_Right.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Left.Size_Y/thread.y;
 if (cTensor_Left.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

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
 CTensor<type_t> cTensor(cTensor_Input.Size_W,cTensor_Input.Size_Z,cTensor_Input.Size_X,cTensor_Input.Size_Y);
 Transponse(cTensor,cTensor_Input);
 return(cTensor);
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для увеличения разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAUpSamplingTensor(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,uint32_t upsampling_x,uint32_t upsampling_y)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t ixp=xp/upsampling_x;
 uint32_t iyp=yp/upsampling_y;

 uint32_t offset_output=xp+yp*tensor_output.GetSizeX();
 type_t *o_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset_output;

 if (ixp>=tensor_input.GetSizeX() || iyp>=tensor_input.GetSizeY())
 {
  *o_ptr=0;
  __syncthreads();
  return;
 }

 uint32_t offset_input=ixp+iyp*tensor_input.GetSizeX();

 type_t *i_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset_input;

 *o_ptr=*i_ptr;

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!увеличение разрешения тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::UpSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t upsampling_x,uint32_t upsampling_y)
{
 //return;//TODO: тест
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X/upsampling_x || cTensor_Input.Size_Y!=cTensor_Output.Size_Y/upsampling_y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::UpSampling: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

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
__global__ void CUDADownSamplingTensor(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,uint32_t downsampling_x,uint32_t downsampling_y)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t ixp=xp*downsampling_x;
 uint32_t iyp=yp*downsampling_y;

 uint32_t offset_output=xp+yp*tensor_output.GetSizeX();
 uint32_t offset_input=ixp+iyp*tensor_input.GetSizeX();

 type_t *i_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset_input;
 type_t *o_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset_output;

 type_t summ=0;
 for(uint32_t y=0;y<downsampling_y;y++,i_ptr+=tensor_input.StrideX)
 {
  type_t *i_ptr_local=i_ptr;
  for(uint32_t x=0;x<downsampling_x;x++,i_ptr_local++)
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
void CTensorMath<type_t>::DownSampling(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,uint32_t downsampling_x,uint32_t downsampling_y)
{
 //return;//TODO: тест
 if (cTensor_Input.Size_X/downsampling_x!=cTensor_Output.Size_X || cTensor_Input.Size_Y/downsampling_y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z)
 {
  throw "CTensor::DownSampling: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

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
__global__ void CUDAMaxPoolingTensor(STensorKernel<type_t> tensor_output,STensorKernel<typename CTensorMath<type_t>::SPos> tensor_position,STensorKernel<type_t> tensor_input,uint32_t pooling_x,uint32_t pooling_y)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_output.GetSizeZ());
 uint32_t w_in=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_input.GetSizeW());
 uint32_t w_out=Mod((blockIdx.z/tensor_output.GetSizeZ()),tensor_output.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t ixp=xp*pooling_x;
 uint32_t iyp=yp*pooling_y;

 uint32_t offset_output=xp+yp*tensor_output.GetSizeX();
 uint32_t offset_input=ixp+iyp*tensor_input.GetSizeX();

 type_t *i_ptr=tensor_input.GetTensorDataPtr(w_in,z)+offset_input;
 type_t *o_ptr=tensor_output.GetTensorDataPtr(w_out,z)+offset_output;

 type_t max_e=*i_ptr;
 uint32_t max_x=ixp;
 uint32_t max_y=iyp;
 for(uint32_t y=0;y<pooling_y;y++,i_ptr+=tensor_input.StrideX)
 {
  type_t *i_ptr_local=i_ptr;
  for(uint32_t x=0;x<pooling_x;x++,i_ptr_local++)
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
 tensor_position.SetElement(w_out,z,yp,xp,sPos);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!увеличение разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPooling(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y)
{
 //return;//TODO: тест
 if ((cTensor_Input.Size_X/pooling_x)!=cTensor_Output.Size_X || (cTensor_Input.Size_Y/pooling_y)!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Position.Size_W!=cTensor_Output.Size_W)
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

 uint32_t block_x=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z*cTensor_Output.Size_W;

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
__global__ void CUDAMaxPoolingTensorBackward(STensorKernel<type_t> tensor_output,STensorKernel<typename CTensorMath<type_t>::SPos> tensor_position,STensorKernel<type_t> tensor_input,uint32_t pooling_x,uint32_t pooling_y)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z;
 uint32_t w=0;
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_output.GetSizeX() || yp>=tensor_output.GetSizeY()) return;

 uint32_t ixp=xp/pooling_x;
 uint32_t iyp=yp/pooling_y;

 type_t value=tensor_input.GetElement(w,z,iyp,ixp);
 typename CTensorMath<type_t>::SPos sPos=tensor_position.TensorData_Ptr[w*tensor_position.StrideW+z*tensor_position.StrideZ+iyp*tensor_position.StrideX+ixp];
 if (sPos.X!=xp || sPos.Y!=yp) value=0;
 tensor_output.SetElement(w,z,yp,xp,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!обратный проход при увеличении разрешения тензора выборкой большего элемента
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::MaxPoolingBackward(CTensor<type_t> &cTensor_Output,const CTensor<CTensorMath<type_t>::SPos> &cTensor_Position,const CTensor<type_t> &cTensor_Input,uint32_t pooling_x,uint32_t pooling_y)
{
 //return;//TODO: тест
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

 uint32_t block_x=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Output.Size_Z;

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
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor.GetSizeZ());
 uint32_t w=Mod((blockIdx.z/tensor.GetSizeZ()),tensor.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor.GetSizeX() || yp>=tensor.GetSizeY()) return;

 type_t value=tensor.GetElement(w,z,yp,xp);
 if (value>max_value) value=max_value;
 if (value<min_value) value=min_value;
 tensor.SetElement(w,z,yp,xp,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!выполнить отсечку значений тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Clip(CTensor<type_t> &cTensor,type_t min_value,type_t max_value)
{
 //return;//TODO: тест

 cTensor.CopyToDevice();

 STensorKernel<type_t> sTensorKernel(cTensor);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE,CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE);

 uint32_t block_x=cTensor.Size_X/thread.x;
 if (cTensor.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor.Size_Y/thread.y;
 if (cTensor.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor.Size_Z*cTensor.Size_W;

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
__global__ void CUDAAdam(STensorKernel<type_t> tensor_weight,STensorKernel<type_t> tensor_dweight,STensorKernel<type_t> tensor_m,STensorKernel<type_t> tensor_v,uint32_t batch_size,double speed,double beta1,double beta2,double epsilon,double iteration)
{
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.z,tensor_weight.GetSizeZ());
 uint32_t w=Mod((blockIdx.z/tensor_weight.GetSizeZ()),tensor_weight.GetSizeW());
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TENSOR_OPERATION_BLOCK_SIZE+y;

 if (xp>=tensor_weight.GetSizeX() || yp>=tensor_weight.GetSizeY()) return;

 tensor_dweight.SelectW(w);
 tensor_dweight.SelectZ(z);

 tensor_weight.SelectW(w);
 tensor_weight.SelectZ(z);

 tensor_m.SelectW(w);
 tensor_m.SelectZ(z);

 tensor_v.SelectW(w);
 tensor_v.SelectZ(z);

 type_t dweight=tensor_dweight.GetElement(yp,xp);
 type_t m=tensor_m.GetElement(yp,xp);
 type_t v=tensor_v.GetElement(yp,xp);

 dweight/=static_cast<type_t>(batch_size);

 m=beta1*m+(1.0-beta1)*dweight;
 v=beta2*v+(1.0-beta2)*dweight*dweight;

 type_t mc=m/(1.0-pow(beta1,iteration));
 type_t vc=v/(1.0-pow(beta2,iteration));

 dweight=speed*mc/(sqrt(vc)+epsilon);

 tensor_m.SetElement(yp,xp,m);
 tensor_v.SetElement(yp,xp,v);

 //корректируем веса
 type_t weight=tensor_weight.GetElement(yp,xp);
 tensor_weight.SetElement(yp,xp,weight-dweight);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//!выполнить алгоритм Adam к весовому тензору
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorMath<type_t>::Adam(CTensor<type_t> &cTensor_Weight,CTensor<type_t> &cTensor_dWeight,CTensor<type_t> &cTensor_M,CTensor<type_t> &cTensor_V,uint32_t batch_size,double speed,double beta1,double beta2,double epsilon,double iteration)
{
 //return;//TODO: тест
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

 uint32_t block_x=cTensor_Weight.Size_X/thread.x;
 if (cTensor_Weight.Size_X%thread.x) block_x++;
 uint32_t block_y=cTensor_Weight.Size_Y/thread.y;
 if (cTensor_Weight.Size_Y%thread.y) block_y++;
 uint32_t block_z=cTensor_Weight.Size_Z*cTensor_Weight.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 CUDAAdam<type_t><<<blocks,thread>>>(sTensorKernel_Weight,sTensorKernel_dWeight,sTensorKernel_M,sTensorKernel_V,batch_size,speed,beta1,beta2,epsilon,iteration);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Weight.SetDeviceOnChange();
 cTensor_M.SetDeviceOnChange();
 cTensor_V.SetDeviceOnChange();
}

#endif

#endif
