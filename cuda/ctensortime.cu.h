#ifndef C_TENSOR_TIME_CU_H
#define C_TENSOR_TIME_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensortime.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции наложения временного шага над тензорами произвольной размерности
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
///!Операции наложения временного шага над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorTime
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
  static void SetTimeStep(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,const CTensor<uint32_t> &cTensor_TimeStep,type_t scale);///<добавить к тензору временной шаг
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};

//****************************************************************************************************
//реализация функций
//****************************************************************************************************

//****************************************************************************************************
//открытые функции
//****************************************************************************************************


/*
//----------------------------------------------------------------------------------------------------
//функция CUDA для добавления временного шага
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASetTimeStep(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<uint32_t> tensor_time_step,type_t scale)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_input.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_input.Size_Z),tensor_input.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_input.Size_X || yp>=tensor_input.Size_Y) return;

 uint32_t size=tensor_input.Size_X*tensor_input.Size_Y*tensor_input.Size_Z;
 uint32_t pos=xp+yp*tensor_input.Size_X+z*tensor_input.Size_X*tensor_input.Size_Y;

 type_t time_step=tensor_time_step.GetElement(w,0,0,0);

 type_t angle=time_step/pow(10000.0f,static_cast<type_t>(pos)/static_cast<type_t>(size));
 type_t value=tensor_input.GetElement(w,z,yp,xp);
 if ((pos&0x01)==0) value+=sin(angle)*scale;//чётное
               else value+=cos(angle)*scale;//нечётное
 tensor_output.SetElement(w,z,yp,xp,value);
 __syncthreads();
}*/


//----------------------------------------------------------------------------------------------------
// Функция CUDA для добавления временного шага
//
// ВАЖНО: временное кодирование зависит ТОЛЬКО от канала z и шага t.
// Оно одинаково для всех пространственных позиций (xp, yp).
// Формула (стандарт "Attention is All You Need"):
//   emb[2i]  =sin( t/10000^(2i/C) )
//   emb[2i+1]=cos( t/10000^(2i/C) )
// где C — число каналов тензора, i=z/2.
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASetTimeStep(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,STensorKernel<uint32_t> tensor_time_step,type_t scale)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_input.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_input.Size_Z),tensor_input.Size_W);

 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;

 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_input.Size_X || yp>=tensor_input.Size_Y) return;

 //шаг времени для данного элемента пакета (w — индекс картинки в батче)
 type_t time_step=static_cast<type_t>(tensor_time_step.GetElement(w,0,0,0));

 // --- Временное кодирование: зависит только от z и t ---
 const uint32_t C=tensor_input.Size_Z;//число каналов
 const uint32_t i=z/2;//индекс "полупары" sin/cos
 const type_t exponent=static_cast<type_t>(2*i)/static_cast<type_t>(C);
 const type_t freq=pow(10000.0f, exponent);
 const type_t angle=time_step/freq;

 type_t emb_z;
 if ((z & 0x01)==0) emb_z=sin(angle)*scale;//чётный канал
               else emb_z=cos(angle)*scale;//нечётный канал

 //добавление одинакового значения ко всем пикселям канала z
 type_t value=tensor_input.GetElement(w, z, yp, xp);
 value+=emb_z;
 tensor_output.SetElement(w, z, yp, xp, value);
}

//----------------------------------------------------------------------------------------------------
//!добавить к тензору временной шаг
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorTime<type_t>::SetTimeStep(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input,const CTensor<uint32_t> &cTensor_TimeStep,type_t scale)
{
 if (cTensor_Input.Size_X!=cTensor_Output.Size_X || cTensor_Input.Size_Y!=cTensor_Output.Size_Y || cTensor_Input.Size_Z!=cTensor_Output.Size_Z ||
     cTensor_Input.Size_W!=cTensor_Output.Size_W || cTensor_TimeStep.Size_W!=cTensor_Input.Size_W || cTensor_TimeStep.Size_X!=1 ||
     cTensor_TimeStep.Size_Y!=1 || cTensor_TimeStep.Size_Z!=1)
 {
  throw "CTensorTime::SetTimeStep: Размерности тензоров не совпадают!";
 }

 cTensor_Input.CopyToDevice();
 cTensor_Output.CopyToDevice();
 cTensor_TimeStep.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<uint32_t> sTensorKernel_TimeStep(cTensor_TimeStep);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Input.Size_Z*cTensor_Input.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDASetTimeStep<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,sTensorKernel_TimeStep,scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

#endif

#endif
