#ifndef C_TENSOR_TEST_CU_H
#define C_TENSOR_TEST_CU_H

//****************************************************************************************************
//Тестирование тензоров
//****************************************************************************************************


//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "ctensor.cu.h"
#include "../system/system.h"
#include "ctensormath.cu.h"
#include "ctensorapplyfunc.cu.h"
#include "ctensorconv.cu.h"

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
///!Тестирование тензоров
//****************************************************************************************************

template<class type_t>
class CTensorTest
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
  static bool Test(void);///<протестировать класс тензоров
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
  static bool TestForwardConvolution(void);///<протестировать прямую свёртку тензоров
  static bool TestForwardConvolutionWithStepAndPadding(void);///<протестировать прямую свёртку тензоров c шагом и дополнением
  static bool TestCreateDeltaWeightAndBiasWithStepAndPadding(void);///<протестировать создание поправок c шагом и дополнением
  static bool TestCreateDeltaWeightAndBias(void);///<протестировать создание поправок
  static bool TestBackwardConvolutionWithStepAndPadding(void);///<протестировать обратную свёртку с шагом и дополнением
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//протестировать прямую свёртку тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensorTest<type_t>::TestForwardConvolution(void)
{
 SYSTEM::PutMessageToConsole("Тест функции ForwardConvolution.");
 const size_t kernel_x=2;
 const size_t kernel_y=2;
 const size_t kernel_z=3;
 const size_t batch_size=10;
 const size_t kernel_amount=2;

 type_t kernel_a[kernel_x*kernel_y*kernel_z]={1,2,3,4,5,6,7,8,9,10,11,12};
 type_t kernel_b[kernel_x*kernel_y*kernel_z]={12,11,10,9,8,7,6,5,4,3,2,1};

 //создаём тензор ядер
 CTensor<type_t> cTensor_Kernel=CTensor<type_t>(1,1,kernel_amount,kernel_x*kernel_y*kernel_z);
 for(size_t w=0;w<1;w++)
 {
  for(size_t n=0;n<kernel_x*kernel_y*kernel_z;n++)
  {
   cTensor_Kernel.SetElement(w,0,0,n,kernel_a[n]);
   cTensor_Kernel.SetElement(w,0,1,n,kernel_b[n]);
  }
 }

 //создаём тензор смещений
 CTensor<type_t> cTensor_Bias=CTensor<type_t>(1,kernel_amount,1,1);
 cTensor_Bias.Zero();

 //входное изображение
 CTensor<type_t> cTensor_Image(batch_size,3,3,3);

 for(size_t w=0;w<batch_size;w++)
 {
  cTensor_Image.SetElement(w,0,0,0,1);
  cTensor_Image.SetElement(w,0,0,1,2);
  cTensor_Image.SetElement(w,0,0,2,3);
  cTensor_Image.SetElement(w,0,1,0,4);
  cTensor_Image.SetElement(w,0,1,1,5);
  cTensor_Image.SetElement(w,0,1,2,6);
  cTensor_Image.SetElement(w,0,2,0,7);
  cTensor_Image.SetElement(w,0,2,1,8);
  cTensor_Image.SetElement(w,0,2,2,9);

  cTensor_Image.SetElement(w,1,0,0,10);
  cTensor_Image.SetElement(w,1,0,1,11);
  cTensor_Image.SetElement(w,1,0,2,12);
  cTensor_Image.SetElement(w,1,1,0,13);
  cTensor_Image.SetElement(w,1,1,1,14);
  cTensor_Image.SetElement(w,1,1,2,15);
  cTensor_Image.SetElement(w,1,2,0,16);
  cTensor_Image.SetElement(w,1,2,1,17);
  cTensor_Image.SetElement(w,1,2,2,18);

  cTensor_Image.SetElement(w,2,0,0,19);
  cTensor_Image.SetElement(w,2,0,1,20);
  cTensor_Image.SetElement(w,2,0,2,21);
  cTensor_Image.SetElement(w,2,1,0,22);
  cTensor_Image.SetElement(w,2,1,1,23);
  cTensor_Image.SetElement(w,2,1,2,24);
  cTensor_Image.SetElement(w,2,2,0,25);
  cTensor_Image.SetElement(w,2,2,1,26);
  cTensor_Image.SetElement(w,2,2,2,27);
 }

 //выходной тензор свёртки
 CTensor<type_t> cTensor_Output(batch_size,2,2,2);
 //проверочный тензор свёртки
 CTensor<type_t> cTensor_Control(batch_size,2,2,2);
 for(size_t w=0;w<batch_size;w++)
 {
  cTensor_Control.SetElement(w,0,0,0,1245);
  cTensor_Control.SetElement(w,0,0,1,1323);

  cTensor_Control.SetElement(w,0,1,0,1479);
  cTensor_Control.SetElement(w,0,1,1,1557);

  cTensor_Control.SetElement(w,1,0,0,627);
  cTensor_Control.SetElement(w,1,0,1,705);

  cTensor_Control.SetElement(w,1,1,0,861);
  cTensor_Control.SetElement(w,1,1,1,939);
 }

 //выполняем прямую свёртку
 CTensorConv<type_t>::ForwardConvolution(cTensor_Output,cTensor_Image,cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_Bias,1,1,0,0);
 //сравниваем полученный тензор
 if (cTensor_Output.Compare(cTensor_Control,"")==false) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");
 return(true);
}

//----------------------------------------------------------------------------------------------------
//протестировать прямую свёртку тензоров c шагом и дополнением
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensorTest<type_t>::TestForwardConvolutionWithStepAndPadding(void)
{
 SYSTEM::PutMessageToConsole("Тест функции ForwardConvolution с шагом и дополнением.");
 const size_t kernel_x=3;
 const size_t kernel_y=3;
 const size_t kernel_z=3;
 const size_t batch_size=10;
 const size_t kernel_amount=2;

 type_t kernel_a[kernel_x*kernel_y*kernel_z]={-1,1,1, -1,1,1, 0,0,1, 0,-1,1, -1,0,-1, 1,0,0, 1,-1,0, -1,0,-1, 0,1,-1};
 type_t kernel_b[kernel_x*kernel_y*kernel_z]={0,0,-1, 1,0,0, 0,-1,0, 1,-1,0, -1,1,0, -1,1,1, 1,0,1, 1,-1,1, -1,0,0};

 //создаём тензор ядер
 CTensor<type_t> cTensor_Kernel=CTensor<type_t>(1,1,kernel_amount,kernel_x*kernel_y*kernel_z);
 for(size_t w=0;w<1;w++)
 {
  for(size_t n=0;n<kernel_x*kernel_y*kernel_z;n++)
  {
   cTensor_Kernel.SetElement(w,0,0,n,kernel_a[n]);
   cTensor_Kernel.SetElement(w,0,1,n,kernel_b[n]);
  }
 }

 //создаём тензор смещений
 CTensor<type_t> cTensor_Bias=CTensor<type_t>(1,kernel_amount,1,1);
 cTensor_Bias.Zero();

 //создаём изображение
 CTensor<type_t> cTensor_Image(batch_size,3,5,5);
 for(size_t w=0;w<batch_size;w++)
 {
  cTensor_Image.SetElement(w,0,0,0,1);
  cTensor_Image.SetElement(w,0,0,1,2);
  cTensor_Image.SetElement(w,0,0,2,0);
  cTensor_Image.SetElement(w,0,0,3,1);
  cTensor_Image.SetElement(w,0,0,4,0);

  cTensor_Image.SetElement(w,0,1,0,2);
  cTensor_Image.SetElement(w,0,1,1,0);
  cTensor_Image.SetElement(w,0,1,2,0);
  cTensor_Image.SetElement(w,0,1,3,0);
  cTensor_Image.SetElement(w,0,1,4,1);

  cTensor_Image.SetElement(w,0,2,0,1);
  cTensor_Image.SetElement(w,0,2,1,2);
  cTensor_Image.SetElement(w,0,2,2,2);
  cTensor_Image.SetElement(w,0,2,3,0);
  cTensor_Image.SetElement(w,0,2,4,2);

  cTensor_Image.SetElement(w,0,3,0,2);
  cTensor_Image.SetElement(w,0,3,1,2);
  cTensor_Image.SetElement(w,0,3,2,2);
  cTensor_Image.SetElement(w,0,3,3,0);
  cTensor_Image.SetElement(w,0,3,4,1);

  cTensor_Image.SetElement(w,0,4,0,2);
  cTensor_Image.SetElement(w,0,4,1,0);
  cTensor_Image.SetElement(w,0,4,2,1);
  cTensor_Image.SetElement(w,0,4,3,0);
  cTensor_Image.SetElement(w,0,4,4,1);

  cTensor_Image.SetElement(w,1,0,0,1);
  cTensor_Image.SetElement(w,1,0,1,2);
  cTensor_Image.SetElement(w,1,0,2,2);
  cTensor_Image.SetElement(w,1,0,3,1);
  cTensor_Image.SetElement(w,1,0,4,2);

  cTensor_Image.SetElement(w,1,1,0,0);
  cTensor_Image.SetElement(w,1,1,1,2);
  cTensor_Image.SetElement(w,1,1,2,2);
  cTensor_Image.SetElement(w,1,1,3,0);
  cTensor_Image.SetElement(w,1,1,4,2);

  cTensor_Image.SetElement(w,1,2,0,1);
  cTensor_Image.SetElement(w,1,2,1,2);
  cTensor_Image.SetElement(w,1,2,2,2);
  cTensor_Image.SetElement(w,1,2,3,1);
  cTensor_Image.SetElement(w,1,2,4,1);

  cTensor_Image.SetElement(w,1,3,0,2);
  cTensor_Image.SetElement(w,1,3,1,2);
  cTensor_Image.SetElement(w,1,3,2,0);
  cTensor_Image.SetElement(w,1,3,3,1);
  cTensor_Image.SetElement(w,1,3,4,0);

  cTensor_Image.SetElement(w,1,4,0,2);
  cTensor_Image.SetElement(w,1,4,1,2);
  cTensor_Image.SetElement(w,1,4,2,1);
  cTensor_Image.SetElement(w,1,4,3,0);
  cTensor_Image.SetElement(w,1,4,4,0);

  cTensor_Image.SetElement(w,2,0,0,0);
  cTensor_Image.SetElement(w,2,0,1,2);
  cTensor_Image.SetElement(w,2,0,2,0);
  cTensor_Image.SetElement(w,2,0,3,1);
  cTensor_Image.SetElement(w,2,0,4,1);

  cTensor_Image.SetElement(w,2,1,0,2);
  cTensor_Image.SetElement(w,2,1,1,0);
  cTensor_Image.SetElement(w,2,1,2,2);
  cTensor_Image.SetElement(w,2,1,3,1);
  cTensor_Image.SetElement(w,2,1,4,1);

  cTensor_Image.SetElement(w,2,2,0,2);
  cTensor_Image.SetElement(w,2,2,1,0);
  cTensor_Image.SetElement(w,2,2,2,1);
  cTensor_Image.SetElement(w,2,2,3,2);
  cTensor_Image.SetElement(w,2,2,4,1);

  cTensor_Image.SetElement(w,2,3,0,0);
  cTensor_Image.SetElement(w,2,3,1,1);
  cTensor_Image.SetElement(w,2,3,2,0);
  cTensor_Image.SetElement(w,2,3,3,1);
  cTensor_Image.SetElement(w,2,3,4,2);

  cTensor_Image.SetElement(w,2,4,0,1);
  cTensor_Image.SetElement(w,2,4,1,0);
  cTensor_Image.SetElement(w,2,4,2,1);
  cTensor_Image.SetElement(w,2,4,3,2);
  cTensor_Image.SetElement(w,2,4,4,1);
 }
 //делаем свёртку
 CTensor<type_t> cTensor_Convolution(batch_size,2,3,3);
 CTensorConv<type_t>::ForwardConvolution(cTensor_Convolution,cTensor_Image,cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_Bias,2,2,1,1);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //cTensor_Convolution.Print("Результат свёртки 1");

 //TODO: добавить проверку

 return(true);
}

//----------------------------------------------------------------------------------------------------
//протестировать создание поправок c шагом и дополнением
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensorTest<type_t>::TestCreateDeltaWeightAndBiasWithStepAndPadding(void)
{
 SYSTEM::PutMessageToConsole("Тест функции CreateDeltaWeightAndBias с шагом и дополнением.");

 const size_t batch_size=10;

 CTensor<type_t> cTensor_Image(batch_size,1,7,7);
 //входное изображение
 for(size_t w=0;w<batch_size;w++)
 {
  for(size_t y=0;y<7;y++)
  {
   for(size_t x=0;x<7;x++)
   {
    cTensor_Image.SetElement(w,0,y,x,x+y*7+1);
   }
  }
 }
 //дельта
 CTensor<type_t> cTensor_Delta(batch_size,1,4,4);
 for(size_t w=0;w<batch_size;w++)
 {
  for(size_t y=0;y<4;y++)
  {
   for(size_t x=0;x<4;x++)
   {
    cTensor_Delta.SetElement(w,0,y,x,x+y*4+1);
   }
  }
 }
 //создаём тензор поправок ядер
 const size_t kernel_x=5;
 const size_t kernel_y=5;
 const size_t kernel_z=1;
 const size_t kernel_amount=1;
 CTensor<type_t> cTensor_dKernel(batch_size,1,kernel_amount,kernel_x*kernel_y*kernel_z);
 //создаём тензор поправок смещений
 CTensor<type_t> cTensor_dBias=CTensor<type_t>(batch_size,kernel_amount,1,1);
 cTensor_dBias.Zero();
 CTensorConv<type_t>::CreateDeltaWeightAndBias(cTensor_dKernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_dBias,cTensor_Image,cTensor_Delta,2,2,2,2);

 //проверочный тензор
 CTensor<type_t> cTensor_Control(batch_size,1,5,5);

 for(size_t w=0;w<batch_size;w++)
 {
  cTensor_Control.SetElement(w,0,0,0,2031);
  cTensor_Control.SetElement(w,0,0,1,2130);
  cTensor_Control.SetElement(w,0,0,2,2746);
  cTensor_Control.SetElement(w,0,0,3,1968);
  cTensor_Control.SetElement(w,0,0,4,2058);

  cTensor_Control.SetElement(w,0,1,0,2724);
  cTensor_Control.SetElement(w,0,1,1,2823);
  cTensor_Control.SetElement(w,0,1,2,3628);
  cTensor_Control.SetElement(w,0,1,3,2598);
  cTensor_Control.SetElement(w,0,1,4,2688);

  cTensor_Control.SetElement(w,0,2,0,3448);
  cTensor_Control.SetElement(w,0,2,1,3556);
  cTensor_Control.SetElement(w,0,2,2,4560);
  cTensor_Control.SetElement(w,0,2,3,3256);
  cTensor_Control.SetElement(w,0,2,4,3352);

  cTensor_Control.SetElement(w,0,3,0,1860);
  cTensor_Control.SetElement(w,0,3,1,1923);
  cTensor_Control.SetElement(w,0,3,2,2428);
  cTensor_Control.SetElement(w,0,3,3,1698);
  cTensor_Control.SetElement(w,0,3,4,1752);

  cTensor_Control.SetElement(w,0,4,0,2301);
  cTensor_Control.SetElement(w,0,4,1,2364);
  cTensor_Control.SetElement(w,0,4,2,2974);
  cTensor_Control.SetElement(w,0,4,3,2076);
  cTensor_Control.SetElement(w,0,4,4,2130);
 }

 CTensor<type_t> cTensor_dKernelBased=CTensor<type_t>(batch_size,kernel_z,kernel_y,kernel_x);

 for(size_t w=0;w<batch_size;w++)
 {
  size_t pos=0;
  for(size_t z=0;z<kernel_z;z++)
  {
   for(size_t y=0;y<kernel_y;y++)
   {
    for(size_t x=0;x<kernel_x;x++,pos++)
    {
     cTensor_dKernelBased.SetElement(w,z,y,x,cTensor_dKernel.GetElement(w,0,0,pos));
    }
   }
  }
 }

 if (cTensor_dKernelBased.Compare(cTensor_Control,"")==false) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 return(true);
}
//----------------------------------------------------------------------------------------------------
//протестировать создание поправок
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensorTest<type_t>::TestCreateDeltaWeightAndBias(void)
{
 SYSTEM::PutMessageToConsole("Тест функции CreateDeltaWeightAndBias.");
 const size_t kernel_x=3;
 const size_t kernel_y=3;
 const size_t kernel_z=1;
 const size_t batch_size=10;
 const size_t kernel_amount=1;

 type_t kernel_a[kernel_x*kernel_y*kernel_z]={1,4,1, 1,4,3, 3,3,1};

 //создаём тензор ядер
 CTensor<type_t> cTensor_Kernel=CTensor<type_t>(batch_size,1,kernel_amount,kernel_x*kernel_y*kernel_z);
 for(size_t w=0;w<batch_size;w++)
 {
  for(size_t n=0;n<kernel_x*kernel_y*kernel_z;n++)
  {
   cTensor_Kernel.SetElement(w,0,0,n,kernel_a[n]);
  }
 }
 //создаём тензор смещений
 CTensor<type_t> cTensor_Bias=CTensor<type_t>(batch_size,kernel_amount,1,1);
 cTensor_Bias.Zero();

 //создаём изображение
 CTensor<type_t> cTensor_Image(batch_size,1,4,4);

 for(size_t w=0;w<batch_size;w++)
 {
  cTensor_Image.SetElement(w,0,0,0,4);
  cTensor_Image.SetElement(w,0,0,1,5);
  cTensor_Image.SetElement(w,0,0,2,8);
  cTensor_Image.SetElement(w,0,0,3,7);

  cTensor_Image.SetElement(w,0,1,0,1);
  cTensor_Image.SetElement(w,0,1,1,8);
  cTensor_Image.SetElement(w,0,1,2,8);
  cTensor_Image.SetElement(w,0,1,3,8);

  cTensor_Image.SetElement(w,0,2,0,3);
  cTensor_Image.SetElement(w,0,2,1,6);
  cTensor_Image.SetElement(w,0,2,2,6);
  cTensor_Image.SetElement(w,0,2,3,4);

  cTensor_Image.SetElement(w,0,3,0,6);
  cTensor_Image.SetElement(w,0,3,1,5);
  cTensor_Image.SetElement(w,0,3,2,7);
  cTensor_Image.SetElement(w,0,3,3,8);
 }
 //делаем прямую свёртку
 CTensor<type_t> cTensor_Convolution(batch_size,1,2,2);
 CTensorConv<type_t>::ForwardConvolution(cTensor_Convolution,cTensor_Image,cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_Bias,1,1,0,0);

 CTensor<type_t> cTensor_dOut(batch_size,1,2,2);
 for(size_t w=0;w<batch_size;w++)
 {
  cTensor_dOut.SetElement(w,0,0,0,2);
  cTensor_dOut.SetElement(w,0,0,1,1);

  cTensor_dOut.SetElement(w,0,1,0,4);
  cTensor_dOut.SetElement(w,0,1,1,4);
 }

 //делаем обратное распространение
 CTensor<type_t> cTensor_Delta(batch_size,1,4,4);

 CTensor<type_t> cTensor_dBias=CTensor<type_t>(batch_size,kernel_amount,1,1);
 cTensor_Bias.Zero();

 CTensorConv<type_t>::BackwardConvolution(cTensor_Delta,cTensor_dOut,cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_dBias,1,1,0,0);
 cTensor_Delta.Print("Тензор поправок дельта");

 //создаём вектор тензоров поправок к ядрам
 CTensor<type_t> cTensor_dKernel=CTensor<type_t>(batch_size,1,kernel_amount,kernel_x*kernel_y*kernel_z);

 cTensor_dKernel.Zero();
 //создаём вектор поправок смещений
 cTensor_dBias.Zero();

 CTensorConv<type_t>::CreateDeltaWeightAndBias(cTensor_dKernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_dBias,cTensor_Image,cTensor_dOut,1,1,0,0);

 //проверяем
 CTensor<type_t> cTensor_Control(batch_size,1,3,3);

 for(size_t w=0;w<batch_size;w++)
 {
  cTensor_Control.SetElement(w,0,0,0,49);
  cTensor_Control.SetElement(w,0,0,1,82);
  cTensor_Control.SetElement(w,0,0,2,87);

  cTensor_Control.SetElement(w,0,1,0,46);
  cTensor_Control.SetElement(w,0,1,1,72);
  cTensor_Control.SetElement(w,0,1,2,64);

  cTensor_Control.SetElement(w,0,2,0,56);
  cTensor_Control.SetElement(w,0,2,1,66);
  cTensor_Control.SetElement(w,0,2,2,76);
 }

 CTensor<type_t> cTensor_dKernelBased=CTensor<type_t>(batch_size,kernel_z,kernel_y,kernel_x);

 for(size_t w=0;w<batch_size;w++)
 {
  size_t pos=0;
  for(size_t z=0;z<kernel_z;z++)
  {
   for(size_t y=0;y<kernel_y;y++)
   {
    for(size_t x=0;x<kernel_x;x++,pos++)
    {
     cTensor_dKernelBased.SetElement(w,z,y,x,cTensor_dKernel.GetElement(w,0,0,pos));
    }
   }
  }
 }

 if (cTensor_dKernelBased.Compare(cTensor_Control,"")==false) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");
 return(true);
}

//----------------------------------------------------------------------------------------------------
//протестировать обратную свёртку с шагом и дополнением
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensorTest<type_t>::TestBackwardConvolutionWithStepAndPadding(void)
 {
  SYSTEM::PutMessageToConsole("Тест функции BackwardConvolution с шагом и дополнением.");

 const size_t kernel_x=5;
 const size_t kernel_y=5;
 const size_t kernel_z=1;
 const size_t batch_size=10;
 const size_t kernel_amount=1;

 //создаём тензор ядер
 CTensor<type_t> cTensor_Kernel=CTensor<type_t>(1,1,kernel_amount,kernel_x*kernel_y*kernel_z);
 for(size_t w=0;w<1;w++)
 {
  size_t pos=0;
  for(size_t z=0;z<1;z++)
  {
   for(size_t y=0;y<5;y++)
   {
    for(size_t x=0;x<5;x++,pos++)
    {
     cTensor_Kernel.SetElement(w,0,0,pos,x+y*5+1);
    }
   }
  }
 }
 //создаём тензор смещений
 CTensor<type_t> cTensor_Bias=CTensor<type_t>(1,kernel_amount,1,1);
 cTensor_Bias.Zero();
 //входное изображение
 CTensor<type_t> cTensor_Delta(batch_size,1,4,4);
 for(size_t w=0;w<batch_size;w++)
 {
  for(size_t y=0;y<4;y++)
  {
   for(size_t x=0;x<4;x++)
   {
    cTensor_Delta.SetElement(w,0,y,x,x+y*4+1);
   }
  }
 }
 //выходной тензор обратной свёртки
 CTensor<type_t> cTensor_Output(batch_size,1,7,7);
 //проверочный тензор обратной свёртки
 CTensor<type_t> cTensor_Control(batch_size,1,7,7);
 //выполняем обратную свёртку
 CTensorConv<type_t>::BackwardConvolution(cTensor_Output,cTensor_Delta,cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_Bias,2,2,2,2);

 for(size_t w=0;w<batch_size;w++)
 {
  cTensor_Control.SetElement(w,0,0,0,56);
  cTensor_Control.SetElement(w,0,0,1,70);
  cTensor_Control.SetElement(w,0,0,2,124);
  cTensor_Control.SetElement(w,0,0,3,102);
  cTensor_Control.SetElement(w,0,0,4,172);
  cTensor_Control.SetElement(w,0,0,5,134);
  cTensor_Control.SetElement(w,0,0,6,156);

  cTensor_Control.SetElement(w,0,1,0,126);
  cTensor_Control.SetElement(w,0,1,1,140);
  cTensor_Control.SetElement(w,0,1,2,244);
  cTensor_Control.SetElement(w,0,1,3,192);
  cTensor_Control.SetElement(w,0,1,4,322);
  cTensor_Control.SetElement(w,0,1,5,244);
  cTensor_Control.SetElement(w,0,1,6,266);

  cTensor_Control.SetElement(w,0,2,0,233);
  cTensor_Control.SetElement(w,0,2,1,266);
  cTensor_Control.SetElement(w,0,2,2,450);
  cTensor_Control.SetElement(w,0,2,3,344);
  cTensor_Control.SetElement(w,0,2,4,567);
  cTensor_Control.SetElement(w,0,2,5,422);
  cTensor_Control.SetElement(w,0,2,6,467);

  cTensor_Control.SetElement(w,0,3,0,318);
  cTensor_Control.SetElement(w,0,3,1,348);
  cTensor_Control.SetElement(w,0,3,2,556);
  cTensor_Control.SetElement(w,0,3,3,400);
  cTensor_Control.SetElement(w,0,3,4,634);
  cTensor_Control.SetElement(w,0,3,5,452);
  cTensor_Control.SetElement(w,0,3,6,490);

  cTensor_Control.SetElement(w,0,4,0,521);
  cTensor_Control.SetElement(w,0,4,1,578);
  cTensor_Control.SetElement(w,0,4,2,918);
  cTensor_Control.SetElement(w,0,4,3,656);
  cTensor_Control.SetElement(w,0,4,4,1035);
  cTensor_Control.SetElement(w,0,4,5,734);
  cTensor_Control.SetElement(w,0,4,6,803);

  cTensor_Control.SetElement(w,0,5,0,510);
  cTensor_Control.SetElement(w,0,5,1,556);
  cTensor_Control.SetElement(w,0,5,2,868);
  cTensor_Control.SetElement(w,0,5,3,608);
  cTensor_Control.SetElement(w,0,5,4,946);
  cTensor_Control.SetElement(w,0,5,5,660);
  cTensor_Control.SetElement(w,0,5,6,714);

  cTensor_Control.SetElement(w,0,6,0,740);
  cTensor_Control.SetElement(w,0,6,1,786);
  cTensor_Control.SetElement(w,0,6,2,1228);
  cTensor_Control.SetElement(w,0,6,3,858);
  cTensor_Control.SetElement(w,0,6,4,1336);
  cTensor_Control.SetElement(w,0,6,5,930);
  cTensor_Control.SetElement(w,0,6,6,984);
 }
 //сравниваем полученный тензор
 if (cTensor_Output.Compare(cTensor_Control,"")==false) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");
 return(true);
}



//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//****************************************************************************************************
//статические функции
//****************************************************************************************************



//----------------------------------------------------------------------------------------------------
//протестировать класс тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensorTest<type_t>::Test(void)
{
 const size_t batch_size=10;

 CTensor<type_t> cTensorA(batch_size,1,2,2);
 CTensor<type_t> cTensorB(batch_size,1,2,2);
 CTensor<type_t> cTensorC(batch_size,1,2,2);

 CTensor<type_t> cTensorAone(1,1,2,2);
 CTensor<type_t> cTensorBone(1,1,2,2);

 for(size_t w=0;w<batch_size;w++)
 {
  cTensorA.SetElement(w,0,0,0,1);
  cTensorA.SetElement(w,0,0,1,2);
  cTensorA.SetElement(w,0,1,0,3);
  cTensorA.SetElement(w,0,1,1,4);

  cTensorB.SetElement(w,0,0,0,1);
  cTensorB.SetElement(w,0,0,1,2);
  cTensorB.SetElement(w,0,1,0,3);
  cTensorB.SetElement(w,0,1,1,4);

  if (w==0)
  {
   cTensorAone.SetElement(w,0,0,0,1);
   cTensorAone.SetElement(w,0,0,1,2);
   cTensorAone.SetElement(w,0,1,0,3);
   cTensorAone.SetElement(w,0,1,1,4);

   cTensorBone.SetElement(w,0,0,0,1);
   cTensorBone.SetElement(w,0,0,1,2);
   cTensorBone.SetElement(w,0,1,0,3);
   cTensorBone.SetElement(w,0,1,1,4);
  }
 }

 SYSTEM::PutMessageToConsole("Тест заполнения тензора по элементам.");
 //проверка на заполнение тензора
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorA.GetElement(w,0,0,0)!=1) return(false);
  if (cTensorA.GetElement(w,0,0,1)!=2) return(false);
  if (cTensorA.GetElement(w,0,1,0)!=3) return(false);
  if (cTensorA.GetElement(w,0,1,1)!=4) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //умножение на число справа
 SYSTEM::PutMessageToConsole("Тест умножения на число справа.");
 cTensorC=cTensorA*static_cast<type_t>(2.0);
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorC.GetElement(w,0,0,0)!=2) return(false);
  if (cTensorC.GetElement(w,0,0,1)!=4) return(false);
  if (cTensorC.GetElement(w,0,1,0)!=6) return(false);
  if (cTensorC.GetElement(w,0,1,1)!=8) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //умножение на число слева
 SYSTEM::PutMessageToConsole("Тест умножения на число слева.");
 cTensorC=static_cast<type_t>(2.0)*cTensorA;
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorC.GetElement(w,0,0,0)!=2) return(false);
  if (cTensorC.GetElement(w,0,0,1)!=4) return(false);
  if (cTensorC.GetElement(w,0,1,0)!=6) return(false);
  if (cTensorC.GetElement(w,0,1,1)!=8) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //умножение тензоров
 SYSTEM::PutMessageToConsole("Тест умножения тензоров.");
 cTensorC=cTensorA*cTensorB;
 cTensorC.Print("Mul");
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorC.GetElement(w,0,0,0)!=7) return(false);
  if (cTensorC.GetElement(w,0,0,1)!=10) return(false);
  if (cTensorC.GetElement(w,0,1,0)!=15) return(false);
  if (cTensorC.GetElement(w,0,1,1)!=22) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //умножение тензоров
 SYSTEM::PutMessageToConsole("Тест умножения тензоров разной глубины w C=1xB.");
 cTensorC=cTensorAone*cTensorB;
 cTensorC.Print("Mul");
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorC.GetElement(w,0,0,0)!=7) return(false);
  if (cTensorC.GetElement(w,0,0,1)!=10) return(false);
  if (cTensorC.GetElement(w,0,1,0)!=15) return(false);
  if (cTensorC.GetElement(w,0,1,1)!=22) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //умножение тензоров
 SYSTEM::PutMessageToConsole("Тест умножения тензоров разной глубины w C=Bx1.");
 CTensorMath<type_t>::Mul(cTensorC,cTensorA,cTensorBone);
 cTensorC.Print("Mul");
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorC.GetElement(w,0,0,0)!=7) return(false);
  if (cTensorC.GetElement(w,0,0,1)!=10) return(false);
  if (cTensorC.GetElement(w,0,1,0)!=15) return(false);
  if (cTensorC.GetElement(w,0,1,1)!=22) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //вычитание тензоров
 SYSTEM::PutMessageToConsole("Тест вычитания тензоров.");
 cTensorC=cTensorA-cTensorB;
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorC.GetElement(w,0,0,0)!=0) return(false);
  if (cTensorC.GetElement(w,0,0,1)!=0) return(false);
  if (cTensorC.GetElement(w,0,1,0)!=0) return(false);
  if (cTensorC.GetElement(w,0,1,1)!=0) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //сложение тензоров
 SYSTEM::PutMessageToConsole("Тест сложения тензоров.");
 cTensorC=cTensorA+cTensorB;
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorC.GetElement(w,0,0,0)!=2) return(false);
  if (cTensorC.GetElement(w,0,0,1)!=4) return(false);
  if (cTensorC.GetElement(w,0,1,0)!=6) return(false);
  if (cTensorC.GetElement(w,0,1,1)!=8) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //транспонирование тензора
 SYSTEM::PutMessageToConsole("Тест транспонирования тензоров.");
 cTensorC=CTensorMath<type_t>::Transpose(cTensorA);
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorC.GetElement(w,0,0,0)!=1) return(false);
  if (cTensorC.GetElement(w,0,0,1)!=3) return(false);
  if (cTensorC.GetElement(w,0,1,0)!=2) return(false);
  if (cTensorC.GetElement(w,0,1,1)!=4) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //поэлементное умножение тензоров
 SYSTEM::PutMessageToConsole("Тест поэлементного умножения тензоров.");
 CTensor<type_t> cTensorE(batch_size,1,2,2);
 CTensorMath<type_t>::TensorItemProduction(cTensorE,cTensorA,cTensorB);
 for(size_t w=0;w<batch_size;w++)
 {
  if (cTensorE.GetElement(w,0,0,0)!=1) return(false);
  if (cTensorE.GetElement(w,0,0,1)!=4) return(false);
  if (cTensorE.GetElement(w,0,1,0)!=9) return(false);
  if (cTensorE.GetElement(w,0,1,1)!=16) return(false);
 }
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 if (TestForwardConvolution()==false) return(false);
 if (TestForwardConvolutionWithStepAndPadding()==false) return(false);
 if (TestCreateDeltaWeightAndBiasWithStepAndPadding()==false) return(false);
 if (TestCreateDeltaWeightAndBias()==false) return(false);
 if (TestBackwardConvolutionWithStepAndPadding()==false) return(false);
 return(true);
}

#endif
