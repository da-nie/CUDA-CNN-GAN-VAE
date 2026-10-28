#ifndef C_TENSOR_ATTENTION_CU_H
#define C_TENSOR_ATTENTION_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensorattention.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции внимания над тензорами произвольной размерности
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
///!Операции внимания над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorAttention
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
  static void SplitKQVTensor(CTensor<type_t> &cTensor_Q,CTensor<type_t> &cTensor_K,CTensor<type_t> &cTensor_V,const CTensor<type_t> &cTensor_QKV,int32_t num_total_tokens,int32_t head_dim,int32_t arch_dim,int32_t q_split_index,int32_t k_split_index,int32_t v_split_index);//разделить тензор на Q,K,V
  static void SetCausalMask(CTensor<type_t> &cTensor);///<нанести каузальную маску на тензор
  static void AttentionUnion(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input);///<объединение для слоя внимания
  static void MulAttentionTranspose(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_X,const CTensor<type_t> &cTensor_W,const CTensor<type_t> &cTensor_B);///<умножить тензоры для слоя внимания
  static void AttentionBackward_dV_dP(CTensor<type_t> &cTensor_dVh,CTensor<type_t> &cTensor_dP,const CTensor<type_t> &cTensor_Vh,const CTensor<type_t> &cTensor_SoftMax,const CTensor<type_t> &cTensor_Delta,int32_t head_dimension);///<получение dV и dP для обратного проход слоя внимания
  static void AttentionBackward_dQ_dK_dV(CTensor<type_t> &cTensor_dQ,CTensor<type_t> &cTensor_dK,CTensor<type_t> &cTensor_dV,const CTensor<type_t> &cTensor_dQh,const CTensor<type_t> &cTensor_dKh,const CTensor<type_t> &cTensor_dVh);///<перестановка осей для обратного прохода слоя внимания
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
//функция CUDA для разделения тензора на Q,K,V
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATensorSplitQKVTensorFunction(STensorKernel<type_t> tensor_q,STensorKernel<type_t> tensor_k,STensorKernel<type_t> tensor_v,STensorKernel<type_t> tensor_qkv,int32_t num_total_tokens,int32_t head_dim,int32_t arch_dim,int32_t q_split_index,int32_t k_split_index,int32_t v_split_index)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t zp=Mod(blockIdx.x,tensor_k.Size_Z);
 uint32_t wp=blockIdx.x/tensor_k.Size_Z;
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 type_t q=0;
 type_t k=0;
 type_t v=0;
 if (yp<num_total_tokens)
 {
  q=tensor_qkv.GetElement(wp,0,yp,q_split_index*arch_dim+zp*head_dim+xp);
  k=tensor_qkv.GetElement(wp,0,yp,k_split_index*arch_dim+zp*head_dim+xp);
  v=tensor_qkv.GetElement(wp,0,yp,v_split_index*arch_dim+zp*head_dim+xp);
 }

 tensor_q.SetElement(wp,zp,yp,xp,q);
 tensor_k.SetElement(wp,zp,yp,xp,k);
 tensor_v.SetElement(wp,zp,yp,xp,v);
}

//----------------------------------------------------------------------------------------------------
//разделить тензор на Q,K,V
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorAttention<type_t>::SplitKQVTensor(CTensor<type_t> &cTensor_Q,CTensor<type_t> &cTensor_K,CTensor<type_t> &cTensor_V,const CTensor<type_t> &cTensor_QKV,int32_t num_total_tokens,int32_t head_dim,int32_t arch_dim,int32_t q_split_index,int32_t k_split_index,int32_t v_split_index)
{
 cTensor_QKV.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_QKV(cTensor_QKV);
 STensorKernel<type_t> sTensorKernel_Q(cTensor_Q);
 STensorKernel<type_t> sTensorKernel_K(cTensor_K);
 STensorKernel<type_t> sTensorKernel_V(cTensor_V);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_K.Size_X/thread.x;
 if (cTensor_K.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_K.Size_Y/thread.y;
 if (cTensor_K.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_K.Size_Z*cTensor_K.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDATensorSplitQKVTensorFunction<type_t><<<blocks,thread>>>(sTensorKernel_Q,sTensorKernel_K,sTensorKernel_V,sTensorKernel_QKV,num_total_tokens,head_dim,arch_dim,q_split_index,k_split_index,v_split_index);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_K.SetDeviceOnChange();
 cTensor_Q.SetDeviceOnChange();
 cTensor_V.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для нанесения каузальной маски на тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASetCausalMask(STensorKernel<type_t> tensor)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor.Size_Z),tensor.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor.Size_X || yp>=tensor.Size_Y) return;

 if (xp>=yp+1) tensor.SetElement(w,z,yp,xp,static_cast<type_t>(-1e9));

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!нанести каузальную маску на тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorAttention<type_t>::SetCausalMask(CTensor<type_t> &cTensor)
{
 cTensor.CopyToDevice();

 STensorKernel<type_t> sTensorKernel(cTensor);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor.Size_X/thread.x;
 if (cTensor.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor.Size_Y/thread.y;
 if (cTensor.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor.Size_Z*cTensor.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDASetCausalMask<type_t><<<blocks,thread>>>(sTensorKernel);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для объединения для слоя внимания
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAAttentionUnion(STensorKernel<type_t> tensor_output,const STensorKernel<type_t> tensor_input)
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

 type_t value=tensor_input.GetElement(w,z,yp,xp);
 tensor_output.SetElement(w,0,yp,xp+z*tensor_input.Size_X,value);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!объединение для слоя внимания
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorAttention<type_t>::AttentionUnion(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_Input)
{
 /*
 if (cTensor_Output_Q.Size_W!=cTensor_Output_K.Size_W || cTensor_Output_Q.Size_Z!=cTensor_Output_K.Size_Z ||
     cTensor_Output_Q.Size_Y!=cTensor_Output_K.Size_Y || cTensor_Output_Q.Size_X!=cTensor_Output_K.Size_X)
 {
  throw "CTensorAttention::SoftMaxX: Размерности выходных тензоров Q и K не совпадают!";
 }
 */
 cTensor_Input.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_Input(cTensor_Input);
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

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
 CUDAAttentionUnion<type_t><<<blocks,thread>>>(sTensorKernel_Output,sTensorKernel_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//!умножить тензоры для слоя внимания
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorAttention<type_t>::MulAttentionTranspose(CTensor<type_t> &cTensor_Output,const CTensor<type_t> &cTensor_X,const CTensor<type_t> &cTensor_W,const CTensor<type_t> &cTensor_B)
{
 cTensor_B.CopyToDevice();

 STensorKernel_AttentionTranspose<type_t> sTensorKernel_AttentionTranspose_Output(cTensor_Output,cTensor_B);
 STensorKernel<type_t> sTensorKernel_X(cTensor_X);
 STensorKernel<type_t> sTensorKernel_W(cTensor_W);
 CTensorMath<type_t>::MulAbstract(cTensor_Output,sTensorKernel_AttentionTranspose_Output,cTensor_X,sTensorKernel_X,cTensor_W,sTensorKernel_W);
}

//----------------------------------------------------------------------------------------------------
//!получение dV и dP для обратного проход слоя внимания
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorAttention<type_t>::AttentionBackward_dV_dP(CTensor<type_t> &cTensor_dVh,CTensor<type_t> &cTensor_dP,const CTensor<type_t> &cTensor_Vh,const CTensor<type_t> &cTensor_SoftMax,const CTensor<type_t> &cTensor_Delta,int32_t head_dimension)
{
 cTensor_Vh.CopyToDevice();
 cTensor_SoftMax.CopyToDevice();
 cTensor_Delta.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_dVh(cTensor_dVh);
 STensorKernel<type_t> sTensorKernel_dP(cTensor_dP);
 STensorKernel_Transpose<type_t> sTensorKernel_Transpose_Vh(cTensor_Vh);
 STensorKernel_Transpose<type_t> sTensorKernel_Transpose_SoftMax(cTensor_SoftMax);
 STensorKernel_AttentionBackwardTransposeDelta<type_t> sTensorKernel_AttentionBackwardTransposeDelta(cTensor_Delta,head_dimension);

 //dVh = P.T @ dOh;
 CTensorMath<type_t>::MulAbstract(cTensor_dVh,sTensorKernel_dVh,cTensor_SoftMax,sTensorKernel_Transpose_SoftMax,cTensor_Delta,sTensorKernel_AttentionBackwardTransposeDelta);
 //dP = dOh @ Vh.T
 CTensorMath<type_t>::MulAbstract(cTensor_dP,sTensorKernel_dP,cTensor_Delta,sTensorKernel_AttentionBackwardTransposeDelta,cTensor_Vh,sTensorKernel_Transpose_Vh);
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для перестановки осей для обратного прохода слоя внимания
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAAttentionBackward_dQ_dK_dV(STensorKernel<type_t> tensor_dQ,const STensorKernel<type_t> tensor_dQh,STensorKernel<type_t> tensor_dK,const STensorKernel<type_t> tensor_dKh,STensorKernel<type_t> tensor_dV,const STensorKernel<type_t> tensor_dVh)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_dQh.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_dQh.Size_Z),tensor_dQh.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_dQh.Size_X || yp>=tensor_dQh.Size_Y) return;
 if (xp>=tensor_dKh.Size_X || yp>=tensor_dKh.Size_Y) return;
 if (xp>=tensor_dVh.Size_X || yp>=tensor_dVh.Size_Y) return;

 uint32_t head_dimension=tensor_dQ.Size_X;

 type_t dQh=tensor_dQh.GetElement(w,z,yp,xp);
 tensor_dQ.SetElement(w,0,yp,xp+z*head_dimension,dQh);

 type_t dKh=tensor_dKh.GetElement(w,z,yp,xp);
 tensor_dK.SetElement(w,0,yp,xp+z*head_dimension,dKh);

 type_t dVh=tensor_dVh.GetElement(w,z,yp,xp);
 tensor_dV.SetElement(w,0,yp,xp+z*head_dimension,dVh);

 __syncthreads();
}

//----------------------------------------------------------------------------------------------------
///!перестановка осей для обратного прохода слоя внимания
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorAttention<type_t>::AttentionBackward_dQ_dK_dV(CTensor<type_t> &cTensor_dQ,CTensor<type_t> &cTensor_dK,CTensor<type_t> &cTensor_dV,const CTensor<type_t> &cTensor_dQh,const CTensor<type_t> &cTensor_dKh,const CTensor<type_t> &cTensor_dVh)
{
 if (cTensor_dQ.Size_W!=cTensor_dK.Size_W || cTensor_dQ.Size_Z!=cTensor_dK.Size_Z ||
     cTensor_dQ.Size_Y!=cTensor_dK.Size_Y || cTensor_dQ.Size_X!=cTensor_dK.Size_X)
 {
  throw "CTensorAttention::SoftMaxX: Размерности выходных тензоров dQ и dK не совпадают!";
 }
 if (cTensor_dQ.Size_W!=cTensor_dV.Size_W || cTensor_dQ.Size_Z!=cTensor_dV.Size_Z ||
     cTensor_dQ.Size_Y!=cTensor_dV.Size_Y || cTensor_dQ.Size_X!=cTensor_dV.Size_X)
 {
  throw "CTensorAttention::SoftMaxX: Размерности выходных тензоров dQh и dVh не совпадают!";
 }

 if (cTensor_dQh.Size_W!=cTensor_dKh.Size_W || cTensor_dQh.Size_Z!=cTensor_dKh.Size_Z ||
     cTensor_dQh.Size_Y!=cTensor_dKh.Size_Y || cTensor_dQh.Size_X!=cTensor_dKh.Size_X)
 {
  throw "CTensorAttention::SoftMaxX: Размерности входных тензоров dQh и dKh не совпадают!";
 }
 if (cTensor_dQh.Size_W!=cTensor_dVh.Size_W || cTensor_dQh.Size_Z!=cTensor_dVh.Size_Z ||
     cTensor_dQh.Size_Y!=cTensor_dVh.Size_Y || cTensor_dQh.Size_X!=cTensor_dVh.Size_X)
 {
  throw "CTensorAttention::SoftMaxX: Размерности входных тензоров dQ и dV не совпадают!";
 }

 if (cTensor_dQ.Size_W!=cTensor_dQh.Size_W) throw("Размерность w входных и выходных тензоров не совпадает!");
 if (cTensor_dQ.Size_Z!=1) throw("Размерность z выходных тензоров должна быть единичной!");
 if (cTensor_dQ.Size_Y!=cTensor_dQh.Size_Y) throw("Размерность y входных и выходных тензоров не совпадает!");
 if (cTensor_dQ.Size_X!=cTensor_dQh.Size_X*cTensor_dQh.Size_Z) throw("Размерность x выходных тензоров должна быть равна размерности z*x входных тензоров!");

 cTensor_dQh.CopyToDevice();
 cTensor_dKh.CopyToDevice();
 cTensor_dVh.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_dQh(cTensor_dQh);
 STensorKernel<type_t> sTensorKernel_dQ(cTensor_dQ);

 STensorKernel<type_t> sTensorKernel_dKh(cTensor_dKh);
 STensorKernel<type_t> sTensorKernel_dK(cTensor_dK);

 STensorKernel<type_t> sTensorKernel_dVh(cTensor_dVh);
 STensorKernel<type_t> sTensorKernel_dV(cTensor_dV);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_dQh.Size_X/thread.x;
 if (cTensor_dQh.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_dQh.Size_Y/thread.y;
 if (cTensor_dQh.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_dQh.Size_Z*cTensor_dQh.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;
 CUDAAttentionBackward_dQ_dK_dV<type_t><<<blocks,thread>>>(sTensorKernel_dQ,sTensorKernel_dQ,sTensorKernel_dK,sTensorKernel_dKh,sTensorKernel_dV,sTensorKernel_dVh);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_dQ.SetDeviceOnChange();
 cTensor_dK.SetDeviceOnChange();
 cTensor_dV.SetDeviceOnChange();
}

#endif

#endif
