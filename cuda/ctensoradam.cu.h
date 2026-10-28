#ifndef C_TENSOR_ADAM_CU_H
#define C_TENSOR_ADAM_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensoadam.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции алгоритма Adam над тензорами произвольной размерности
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
///!Операции алгоритма Adam над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorAdam
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
  static void Adam(CTensor<type_t> &cTensor_Weight,CTensor<type_t> &cTensor_dWeight,CTensor<type_t> &cTensor_M,CTensor<type_t> &cTensor_V,uint32_t batch_size,double speed,double beta1,double beta2,double epsilon,double iteration);///<выполнить алгоритм Adam к весовому тензору
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
//функция CUDA для выполнения алгоритма Adam
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAAdam(STensorKernel<type_t> tensor_weight,STensorKernel<type_t> tensor_dweight,STensorKernel<type_t> tensor_m,STensorKernel<type_t> tensor_v,uint32_t batch_size,double speed,double beta1,double beta2,double epsilon,double db1,double db2)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_weight.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_weight.Size_Z),tensor_weight.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_weight.Size_X || yp>=tensor_weight.Size_Y) return;

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

 type_t mc=m/db1;
 type_t vc=v/db2;

 dweight=speed*mc/(sqrt(vc)+epsilon);

 tensor_m.SetElement(yp,xp,m);
 tensor_v.SetElement(yp,xp,v);

 //корректируем веса
 type_t weight=tensor_weight.GetElement(yp,xp);
 tensor_weight.SetElement(yp,xp,weight-dweight);
}

//----------------------------------------------------------------------------------------------------
//!выполнить алгоритм Adam к весовому тензору
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorAdam<type_t>::Adam(CTensor<type_t> &cTensor_Weight,CTensor<type_t> &cTensor_dWeight,CTensor<type_t> &cTensor_M,CTensor<type_t> &cTensor_V,uint32_t batch_size,double speed,double beta1,double beta2,double epsilon,double iteration)
{
 if (cTensor_Weight.Size_X!=cTensor_dWeight.Size_X || cTensor_Weight.Size_Y!=cTensor_dWeight.Size_Y || cTensor_Weight.Size_Z!=cTensor_dWeight.Size_Z || cTensor_Weight.Size_W!=cTensor_dWeight.Size_W ||
     cTensor_Weight.Size_X!=cTensor_M.Size_X || cTensor_Weight.Size_Y!=cTensor_M.Size_Y || cTensor_Weight.Size_Z!=cTensor_M.Size_Z || cTensor_Weight.Size_W!=cTensor_M.Size_W ||
     cTensor_Weight.Size_X!=cTensor_V.Size_X || cTensor_Weight.Size_Y!=cTensor_V.Size_Y || cTensor_Weight.Size_Z!=cTensor_V.Size_Z || cTensor_Weight.Size_W!=cTensor_V.Size_W)
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
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Weight.Size_X/thread.x;
 if (cTensor_Weight.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Weight.Size_Y/thread.y;
 if (cTensor_Weight.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Weight.Size_Z*cTensor_Weight.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 iteration+=1;//чтобы избежать деления на 0 при 0 итерации
 double db1=1.0-pow(beta1,iteration);
 double db2=1.0-pow(beta2,iteration);

 CUDAAdam<type_t><<<blocks,thread>>>(sTensorKernel_Weight,sTensorKernel_dWeight,sTensorKernel_M,sTensorKernel_V,batch_size,speed,beta1,beta2,epsilon,db1,db2);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Weight.SetDeviceOnChange();
 cTensor_M.SetDeviceOnChange();
 cTensor_V.SetDeviceOnChange();
}

#endif

#endif
