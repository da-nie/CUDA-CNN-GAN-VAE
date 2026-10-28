#ifndef C_TENSOR_RANDOM_CU_H
#define C_TENSOR_RANDOM_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensorandom.h"

#endif

#ifndef USE_CPU

//****************************************************************************************************
//Операции со случайными числами над тензорами произвольной размерности
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
///!Операции со случайными числамия над тензорами произвольной размерности
//****************************************************************************************************

template<class type_t>
class CTensorRandom
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
  static void CreateDropOutMatrix(CTensor<type_t> &cTensor_Output,type_t drop_out);///<создать матрицу исключения
  static void SetNormalNoise(CTensor<type_t> &cTensor_Output);///<задать тензор случайными значениями с нормальным распределением
  static void GetNoiseImageAndNoise(CTensor<type_t> &cTensor_NoisyImage,CTensor<type_t> &cTensor_Noise,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_SqrtAlphaBar,const CTensor<type_t> &cTensor_SqrtOneMinusAlphaBar);///<заполнить тензоры зашумлённым изображением и шумом
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
//функция CUDA для заполнения матрицы исключения
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADropOut(STensorKernel<type_t> tensor_output,unsigned long long seed,float drop_out)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_output.Size_Z),tensor_output.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 uint32_t pos=(xp+yp*tensor_output.Size_X+z*tensor_output.Size_X*tensor_output.Size_Y)+w*tensor_output.Size_X*tensor_output.Size_Y*tensor_output.Size_Z;

 curandStatePhilox4_32_10_t state;
 curand_init(seed,pos,0,&state);
 type_t random=curand_uniform(&state);
 /*
 curandState state;
 curand_init(seed,pos,0,&state);
 type_t random=curand_uniform(&state);
 */

 type_t mult=static_cast<type_t>(1.0/(1.0-drop_out));
 if (random>=drop_out) tensor_output.SetElement(w,z,yp,xp,mult);
                  else tensor_output.SetElement(w,z,yp,xp,0);
}
//----------------------------------------------------------------------------------------------------
//!создать матрицу исключения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorRandom<type_t>::CreateDropOutMatrix(CTensor<type_t> &cTensor_Output,type_t drop_out)
{
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 CUDADropOut<type_t><<<blocks,thread>>>(sTensorKernel_Output,rand(),drop_out);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для заполнения тензора случайными числами с нормальным распределением
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASetNormalNoise(STensorKernel<type_t> tensor_output,unsigned long long seed)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_output.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_output.Size_Z),tensor_output.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_output.Size_X || yp>=tensor_output.Size_Y) return;

 uint32_t pos=xp+yp*tensor_output.Size_X+z*tensor_output.Size_X*tensor_output.Size_Y+w*tensor_output.Size_X*tensor_output.Size_Y*tensor_output.Size_Z;
 curandState state;
 curand_init(seed,pos,0,&state);
 type_t value=curand_normal(&state);
 tensor_output.SetElement(w,z,yp,xp,value);
}

//----------------------------------------------------------------------------------------------------
//!задать тензор случайными значениями с нормальным распределением
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorRandom<type_t>::SetNormalNoise(CTensor<type_t> &cTensor_Output)
{
 STensorKernel<type_t> sTensorKernel_Output(cTensor_Output);

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Output.Size_X/thread.x;
 if (cTensor_Output.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Output.Size_Y/thread.y;
 if (cTensor_Output.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Output.Size_Z*cTensor_Output.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 CUDASetNormalNoise<type_t><<<blocks,thread>>>(sTensorKernel_Output,rand());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Output.SetDeviceOnChange();
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для заполнения тензора случайными числами с нормальным распределением и создания изображения с шумом
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAGetNoiseImageAndNoise(STensorKernel<type_t> tensor_noisy_image,STensorKernel<type_t> tensor_noise,STensorKernel<type_t> tensor_image,STensorKernel<type_t> tensor_sqrt_alpha_bar,STensorKernel<type_t> tensor_sqrt_one_minus_alpha_bar,unsigned long long seed)
{
 uint32_t blockCol=blockIdx.z;
 uint32_t blockRow=blockIdx.y;
 uint32_t z=Mod(blockIdx.x,tensor_noisy_image.Size_Z);
 uint32_t w=Mod((blockIdx.x/tensor_noisy_image.Size_Z),tensor_noisy_image.Size_W);
 //координаты элементов блока в выходном тензоре
 uint32_t x=threadIdx.x;
 uint32_t y=threadIdx.y;
 //получаем подтензоры
 uint32_t xp=blockCol*CTensorMath<type_t>::TILE_BLOCK_SIZE+x;
 uint32_t yp=blockRow*CTensorMath<type_t>::TILE_BLOCK_SIZE+y;

 if (xp>=tensor_noisy_image.Size_X || yp>=tensor_noisy_image.Size_Y) return;

 uint32_t pos=(xp+yp*tensor_noisy_image.Size_X+z*tensor_noisy_image.Size_X*tensor_noisy_image.Size_Y)+w*tensor_noisy_image.Size_X*tensor_noisy_image.Size_Y*tensor_noisy_image.Size_Z;


 curandStatePhilox4_32_10_t state;
 curand_init(seed,pos,0,&state);
 type_t noise=curand_normal(&state);

 /*curandState state;
 curand_init(seed,pos,0,&state);
 type_t noise=curand_normal(&state);*/

 tensor_noise.SetElement(w,z,yp,xp,noise);

 type_t image=tensor_image.GetElement(w,z,yp,xp);
 type_t sqrt_alpha_bar=tensor_sqrt_alpha_bar.GetElement(w,0,0,0);
 type_t sqrt_one_minus_alpha_bar=tensor_sqrt_one_minus_alpha_bar.GetElement(w,0,0,0);

 type_t value=sqrt_alpha_bar*image+sqrt_one_minus_alpha_bar*noise;

 tensor_noisy_image.SetElement(w,z,yp,xp,value);
}

//----------------------------------------------------------------------------------------------------
//!заполнить тензоры зашумлённым изображением и шумом
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensorRandom<type_t>::GetNoiseImageAndNoise(CTensor<type_t> &cTensor_NoisyImage,CTensor<type_t> &cTensor_Noise,const CTensor<type_t> &cTensor_Image,const CTensor<type_t> &cTensor_SqrtAlphaBar,const CTensor<type_t> &cTensor_SqrtOneMinusAlphaBar)
{
 cTensor_Image.CopyToDevice();
 cTensor_SqrtAlphaBar.CopyToDevice();
 cTensor_SqrtOneMinusAlphaBar.CopyToDevice();

 STensorKernel<type_t> sTensorKernel_NoisyImage(cTensor_NoisyImage);
 STensorKernel<type_t> sTensorKernel_Noise(cTensor_Noise);
 STensorKernel<type_t> sTensorKernel_Image(cTensor_Image);
 STensorKernel<type_t> sTensorKernel_SqrtAlphaBar(cTensor_SqrtAlphaBar);
 STensorKernel<type_t> sTensorKernel_SqrtOneMinusAlphaBar(cTensor_SqrtOneMinusAlphaBar);

 if (cTensor_NoisyImage.Size_W!=cTensor_Noise.Size_W || cTensor_NoisyImage.Size_X!=cTensor_Noise.Size_X || cTensor_NoisyImage.Size_Y!=cTensor_Noise.Size_Y || cTensor_NoisyImage.Size_Z!=cTensor_Noise.Size_Z)
 {
  throw "CTensor::GetNoiseImageAndNoise: Размерности тензоров NoisyImage и Noise не совпадают!";
 }
 if (cTensor_NoisyImage.Size_W!=cTensor_Image.Size_W || cTensor_NoisyImage.Size_X!=cTensor_Image.Size_X || cTensor_NoisyImage.Size_Y!=cTensor_Image.Size_Y || cTensor_NoisyImage.Size_Z!=cTensor_Image.Size_Z)
 {
  throw "CTensor::GetNoiseImageAndNoise: Размерности тензоров NoisyImage и Image не совпадают!";
 }
 if (cTensor_SqrtAlphaBar.Size_W!=cTensor_Image.Size_W || cTensor_SqrtOneMinusAlphaBar.Size_W!=cTensor_Image.Size_W)
 {
  throw "CTensor::GetNoiseImageAndNoise: Размерности тензоров SqrtAlphaBar/SqrtOneMinusAlphaBar и Image не совпадают!";
 }

 //запускаем процесс
 dim3 thread(CTensorMath<type_t>::TILE_BLOCK_SIZE,CTensorMath<type_t>::TILE_BLOCK_SIZE);

 uint32_t block_z=cTensor_Noise.Size_X/thread.x;
 if (cTensor_Noise.Size_X%thread.x) block_z++;
 uint32_t block_y=cTensor_Noise.Size_Y/thread.y;
 if (cTensor_Noise.Size_Y%thread.y) block_y++;
 uint32_t block_x=cTensor_Noise.Size_Z*cTensor_Noise.Size_W;

 dim3 blocks(block_x,block_y,block_z);
 if (blocks.x==0) blocks.x=1;
 if (blocks.y==0) blocks.y=1;
 if (blocks.z==0) blocks.z=1;

 CUDAGetNoiseImageAndNoise<type_t><<<blocks,thread>>>(sTensorKernel_NoisyImage,sTensorKernel_Noise,sTensorKernel_Image,sTensorKernel_SqrtAlphaBar,sTensorKernel_SqrtOneMinusAlphaBar,rand());
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 cTensor_Noise.SetDeviceOnChange();
 cTensor_NoisyImage.SetDeviceOnChange();
}

#endif

#endif
