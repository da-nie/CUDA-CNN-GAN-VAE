#ifndef C_TENSOR_SUM_CU_H
#define C_TENSOR_SUM_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensorsum.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции суммирования над тензорами произвольной размерности
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
///!Операции суммирования над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorSum
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
  static void AddSumW(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale=1,type_t right_scale=1);///<сложить тензоры по координате W
  static void SumXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,type_t scale=1);///<вычислить сумму элементов по X и Y для каждого Z
  static void AddSumX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input_A,CTensor<type_t> &cTensor_Input_B);///<сложить в слой X слои Y,Z,W
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
//функция CUDA для сложения тензоров по координате W
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddSumWTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_left,STensorKernel<type_t> tensor_right,type_t left_scale,type_t right_scale)
{
 //блок TILE_BLOCK_SIZE x TILE_BLOCK_SIZE в выходном тензоре
 uint32_t blockCol=blockIdx.x;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=blockIdx.z;
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x+blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE;
 uint32_t y=threadIdx.y+blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE;

 type_t summ_left=0;
 type_t summ_right=0;
 for(uint32_t w=0;w<tensor_left.Size_W;w++)
 {
  type_t v=tensor_left.GetElement(w,z,y,x);
  summ_left+=v;
 }
 summ_left*=left_scale;
 for(uint32_t w=0;w<tensor_right.Size_W;w++)
 {
  type_t v=tensor_right.GetElement(w,z,y,x);
  summ_right+=v;
 }
 summ_right*=right_scale;
 type_t summ_output=summ_left+summ_right;
 tensor_output.SetElement(0,z,y,x,summ_output);//помещаем сумму в нулевой слой w
 for(uint32_t w=1;w<tensor_output.Size_W;w++) tensor_output.SetElement(w,z,y,x,0);//все остальные слои w обнулены

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
//сложить тензоры по координате W
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorSum<type_t>::AddSumW(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right,type_t left_scale,type_t right_scale)
{
 if (cTensor_Left.Size_X!=cTensor_Right.Size_X || cTensor_Left.Size_Y!=cTensor_Right.Size_Y || cTensor_Left.Size_Z!=cTensor_Right.Size_Z ||
     cTensor_Output.Size_X!=cTensor_Right.Size_X || cTensor_Output.Size_Y!=cTensor_Right.Size_Y || cTensor_Output.Size_Z!=cTensor_Right.Size_Z)
 {
  throw "CTensor::AddSumW: Размерности тензоров не совпадают!";
 }

 cTensor_Left.CopyToDevice();
 cTensor_Right.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Left(cTensor_Left);
 STensorKernel<type_t> sTensorKernel_Right(cTensor_Right);

 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_x=sTensorKernel_Output.Size_X/thread.x;
 if (sTensorKernel_Output.Size_X%thread.x) block_x++;
 uint32_t block_y=sTensorKernel_Output.Size_Y/thread.y;
 if (sTensorKernel_Output.Size_Y%thread.y) block_y++;
 uint32_t block_z=sTensorKernel_Output.Size_Z;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 CUDATensorAddSumWTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Left,sTensorKernel_Right,left_scale,right_scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления суммы элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSumXYTensorFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input,type_t scale)
{
 uint32_t w_in=Mod(blockIdx.x,tensor_input.Size_W);
 uint32_t w_out=Mod(blockIdx.x,tensor_output.Size_W);
 uint32_t z=Mod(blockIdx.y,tensor_output.Size_Z);

 //суммируем по X и Y
 type_t summ=0;

 type_t *d_xin=tensor_input.GetTensorDataPtr(w_in,z);
 type_t *d_xout=tensor_output.GetTensorDataPtr(w_out,z);

 for(uint32_t y=0;y<tensor_input.Size_Y;y++)
 {
  for(uint32_t x=0;x<tensor_input.Size_X;x++,d_xin++) summ+=*d_xin;
 }
 *d_xout=summ*scale;
}

//----------------------------------------------------------------------------------------------------
//вычислить сумму элементов по X и Y для каждого Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorSum<type_t>::SumXY(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input,type_t scale)
{
 if (cTensor_Input.Size_Z!=cTensor_Output.Size_Z || cTensor_Input.Size_W!=cTensor_Output.Size_W)
 {
  throw "CTensor::SumXY: Размерности тензоров не совпадают!";
 }
 if (cTensor_Output.Size_X!=1 || cTensor_Output.Size_Y!=1)
 {
  throw "CTensor::SumXY: Размерность выходного тензора по x и y должна быть 1!";
 }

 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);

 //запускаем процесс
 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input.Size_W,cTensor_Input.Size_Z);

/*
 dim3 thread(CTensorSum<type_t>::TILE_BLOCK_SIZE,CTensorSum<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Input.Size_X/thread.x;
 if (cTensor_Input.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Input.Size_Y/thread.y;
 if (cTensor_Input.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorSumXYTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
*/
 CUDATensorSumXYTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input,scale);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
///функция CUDA для сложения в слой X значений по X, Y и Z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorAddSumXFunction(STensorKernel<type_t> tensor_output,STensorKernel<type_t> tensor_input_a,STensorKernel<type_t> tensor_input_b)
{
 uint32_t x=blockIdx.x;
 type_t summ=tensor_input_a.GetElement(0,0,0,x);
 for(uint32_t w=0;w<tensor_input_b.Size_W;w++)
 {
  for(uint32_t y=0;y<tensor_input_b.Size_Y;y++)
  {
   for(uint32_t z=0;z<tensor_input_b.Size_Z;z++)
   {
    summ+=tensor_input_b.GetElement(w,z,y,x);
   }
  }
 }
 tensor_output.SetElement(0,0,0,x,summ);
}
//----------------------------------------------------------------------------------------------------
///!сложить в слой X слои Y,Z,W
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorSum<type_t>::AddSumX(CTensor<type_t> &cTensor_Output,CTensor<type_t> &cTensor_Input_A,CTensor<type_t> &cTensor_Input_B)
{
 if (cTensor_Output.Size_Z!=1) throw "CTensorSum::AddSumX: Размерность выходного тензора по Z должна быть равна единице!";
 if (cTensor_Output.Size_W!=1) throw "CTensorSum::AddSumX: Размерность выходного тензора по W должна быть равна единице!";
 if (cTensor_Output.Size_Y!=1) throw "CTensorSum::AddSumX: Размерность выходного тензора по Y должна быть равна единице!";

 if (cTensor_Input_A.Size_Z!=1) throw "CTensorSum::AddSumX: Размерность первого sвходного тензора по Z должна быть равна единице!";
 if (cTensor_Input_A.Size_W!=1) throw "CTensorSum::AddSumX: Размерность первого sвходного тензора по W должна быть равна единице!";
 if (cTensor_Input_A.Size_Y!=1) throw "CTensorSum::AddSumX: Размерность первого sвходного тензора по Y должна быть равна единице!";

 if (cTensor_Input_A.Size_X!=cTensor_Output.Size_X || cTensor_Input_A.Size_X!=cTensor_Input_B.Size_X)
 {
  throw "CTensorSum::AddSumX: Размерности тензоров не совпадают!";
 }

 cTensor_Input_A.CopyToDevice();
 cTensor_Input_B.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);
 STensorKernel<type_t> sTensorKernel_Input_A(cTensor_Input_A);
 STensorKernel<type_t> sTensorKernel_Input_B(cTensor_Input_B);

 //запускаем процесс
 dim3 thread(1,1,1);

 dim3 blocks(cTensor_Input_B.Size_X,1,1);

 CUDATensorAddSumXFunction<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input_A,sTensorKernel_Input_B);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


#endif

#endif
