#ifndef TENSOR_KERNEL_CU_H
#define TENSOR_KERNEL_CU_H

#include "../../settings.h"


#ifndef USE_CPU

//****************************************************************************************************
//Операции над тензорами произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "../ctensor.cu.h"

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
struct STensorTransposeKernel
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

 __host__ __device__ STensorTransposeKernel(void)///<конструктор
 {
 }
 __host__ __device__ STensorTransposeKernel(const CTensor<type_t> &cTensor)///<конструктор
 {
  Set(cTensor);
 }

 __host__ __device__ type_t* GetTensorDataPtr(uint32_t w,uint32_t z)///<получить указатель на элементы с глубиной z
 {
  return(&TensorData_Ptr[w*StrideW+z*StrideZ]);
 }

 __host__ __device__ type_t GetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x)
 {
  if (x>=Size_X || y>=Size_Y || z>=Size_Z || w>=Size_W) return(0);
  return TensorData_Ptr[w*StrideW+z*StrideZ+x*StrideX+y];
 }
 __host__ __device__ void SetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y || z>=Size_Z || w>=Size_W) return;
  TensorData_Ptr[w*StrideW+z*StrideZ+x*StrideX+y]=value;
 }

 __host__ __device__ uint32_t GetSizeX(void)
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
  if (x>=Size_X || y>=Size_Y) return(0);
  return(TensorData_WZ_Ptr[x*StrideX+y]);
 }
 __forceinline__ __host__ __device__ void SetElement(uint32_t y,uint32_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y) return;
  TensorData_WZ_Ptr[x*StrideX+y]=value;
 }

 __forceinline__ __host__ __device__ type_t GetElement(uint32_t z,uint32_t y,uint32_t x)
 {
  if (x>=Size_X || y>=Size_Y || z>=Size_Z) return(0);
  return(TensorData_W_Ptr[z*StrideZ+x*StrideX+y]);
 }
 __forceinline__ __host__ __device__ void SetElement(uint32_t z,uint32_t y,uint32_t x,type_t value)
 {
  if (x>=Size_X || y>=Size_Y || z>=Size_Z) return;
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
///!Функции
//****************************************************************************************************
static __forceinline__ __host__ __device__ uint32_t Mod(uint32_t param,uint32_t divider)
{
 uint32_t div=param/divider;
 return(param-div*divider);
}

#endif

#endif
