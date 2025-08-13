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
 const uint32_t kernel_x=2;
 const uint32_t kernel_y=2;
 const uint32_t kernel_z=3;
 const uint32_t kernel_amount=2;

 type_t kernel_a[kernel_x*kernel_y*kernel_z]={1,2,3,4,5,6,7,8,9,10,11,12};
 type_t kernel_b[kernel_x*kernel_y*kernel_z]={12,11,10,9,8,7,6,5,4,3,2,1};

 //создаём тензор ядер
 CTensor<type_t> cTensor_Kernel=CTensor<type_t>(1,kernel_amount,kernel_x*kernel_y*kernel_z);
 for(uint32_t n=0;n<kernel_x*kernel_y*kernel_z;n++)
 {
  cTensor_Kernel.SetElement(0,0,n,kernel_a[n]);
  cTensor_Kernel.SetElement(0,1,n,kernel_b[n]);
 }

 //создаём тензор смещений
 CTensor<type_t> cTensor_Bias=CTensor<type_t>(kernel_amount,1,1);
 cTensor_Bias.Zero();

 //входное изображение
 CTensor<type_t> cTensor_Image(3,3,3);

 cTensor_Image.SetElement(0,0,0,1);
 cTensor_Image.SetElement(0,0,1,2);
 cTensor_Image.SetElement(0,0,2,3);
 cTensor_Image.SetElement(0,1,0,4);
 cTensor_Image.SetElement(0,1,1,5);
 cTensor_Image.SetElement(0,1,2,6);
 cTensor_Image.SetElement(0,2,0,7);
 cTensor_Image.SetElement(0,2,1,8);
 cTensor_Image.SetElement(0,2,2,9);

 cTensor_Image.SetElement(1,0,0,10);
 cTensor_Image.SetElement(1,0,1,11);
 cTensor_Image.SetElement(1,0,2,12);
 cTensor_Image.SetElement(1,1,0,13);
 cTensor_Image.SetElement(1,1,1,14);
 cTensor_Image.SetElement(1,1,2,15);
 cTensor_Image.SetElement(1,2,0,16);
 cTensor_Image.SetElement(1,2,1,17);
 cTensor_Image.SetElement(1,2,2,18);

 cTensor_Image.SetElement(2,0,0,19);
 cTensor_Image.SetElement(2,0,1,20);
 cTensor_Image.SetElement(2,0,2,21);
 cTensor_Image.SetElement(2,1,0,22);
 cTensor_Image.SetElement(2,1,1,23);
 cTensor_Image.SetElement(2,1,2,24);
 cTensor_Image.SetElement(2,2,0,25);
 cTensor_Image.SetElement(2,2,1,26);
 cTensor_Image.SetElement(2,2,2,27);
 //выходной тензор свёртки
 CTensor<type_t> cTensor_Output(2,2,2);
 //проверочный тензор свёртки
 CTensor<type_t> cTensor_Control(2,2,2);

 cTensor_Control.SetElement(0,0,0,1245);
 cTensor_Control.SetElement(0,0,1,1323);

 cTensor_Control.SetElement(0,1,0,1479);
 cTensor_Control.SetElement(0,1,1,1557);

 cTensor_Control.SetElement(1,0,0,627);
 cTensor_Control.SetElement(1,0,1,705);

 cTensor_Control.SetElement(1,1,0,861);
 cTensor_Control.SetElement(1,1,1,939);

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
 const uint32_t kernel_x=3;
 const uint32_t kernel_y=3;
 const uint32_t kernel_z=3;
 const uint32_t kernel_amount=2;

 type_t kernel_a[kernel_x*kernel_y*kernel_z]={-1,1,1, -1,1,1, 0,0,1, 0,-1,1, -1,0,-1, 1,0,0, 1,-1,0, -1,0,-1, 0,1,-1};
 type_t kernel_b[kernel_x*kernel_y*kernel_z]={0,0,-1, 1,0,0, 0,-1,0, 1,-1,0, -1,1,0, -1,1,1, 1,0,1, 1,-1,1, -1,0,0};

 //создаём тензор ядер
 CTensor<type_t> cTensor_Kernel=CTensor<type_t>(1,kernel_amount,kernel_x*kernel_y*kernel_z);
 for(uint32_t n=0;n<kernel_x*kernel_y*kernel_z;n++)
 {
  cTensor_Kernel.SetElement(0,0,n,kernel_a[n]);
  cTensor_Kernel.SetElement(0,1,n,kernel_b[n]);
 }

 //создаём тензор смещений
 CTensor<type_t> cTensor_Bias=CTensor<type_t>(kernel_amount,1,1);
 cTensor_Bias.Zero();

 //создаём изображение
 CTensor<type_t> cTensor_Image(3,5,5);

 cTensor_Image.SetElement(0,0,0,1);
 cTensor_Image.SetElement(0,0,1,2);
 cTensor_Image.SetElement(0,0,2,0);
 cTensor_Image.SetElement(0,0,3,1);
 cTensor_Image.SetElement(0,0,4,0);

 cTensor_Image.SetElement(0,1,0,2);
 cTensor_Image.SetElement(0,1,1,0);
 cTensor_Image.SetElement(0,1,2,0);
 cTensor_Image.SetElement(0,1,3,0);
 cTensor_Image.SetElement(0,1,4,1);

 cTensor_Image.SetElement(0,2,0,1);
 cTensor_Image.SetElement(0,2,1,2);
 cTensor_Image.SetElement(0,2,2,2);
 cTensor_Image.SetElement(0,2,3,0);
 cTensor_Image.SetElement(0,2,4,2);

 cTensor_Image.SetElement(0,3,0,2);
 cTensor_Image.SetElement(0,3,1,2);
 cTensor_Image.SetElement(0,3,2,2);
 cTensor_Image.SetElement(0,3,3,0);
 cTensor_Image.SetElement(0,3,4,1);

 cTensor_Image.SetElement(0,4,0,2);
 cTensor_Image.SetElement(0,4,1,0);
 cTensor_Image.SetElement(0,4,2,1);
 cTensor_Image.SetElement(0,4,3,0);
 cTensor_Image.SetElement(0,4,4,1);

 cTensor_Image.SetElement(1,0,0,1);
 cTensor_Image.SetElement(1,0,1,2);
 cTensor_Image.SetElement(1,0,2,2);
 cTensor_Image.SetElement(1,0,3,1);
 cTensor_Image.SetElement(1,0,4,2);

 cTensor_Image.SetElement(1,1,0,0);
 cTensor_Image.SetElement(1,1,1,2);
 cTensor_Image.SetElement(1,1,2,2);
 cTensor_Image.SetElement(1,1,3,0);
 cTensor_Image.SetElement(1,1,4,2);

 cTensor_Image.SetElement(1,2,0,1);
 cTensor_Image.SetElement(1,2,1,2);
 cTensor_Image.SetElement(1,2,2,2);
 cTensor_Image.SetElement(1,2,3,1);
 cTensor_Image.SetElement(1,2,4,1);

 cTensor_Image.SetElement(1,3,0,2);
 cTensor_Image.SetElement(1,3,1,2);
 cTensor_Image.SetElement(1,3,2,0);
 cTensor_Image.SetElement(1,3,3,1);
 cTensor_Image.SetElement(1,3,4,0);

 cTensor_Image.SetElement(1,4,0,2);
 cTensor_Image.SetElement(1,4,1,2);
 cTensor_Image.SetElement(1,4,2,1);
 cTensor_Image.SetElement(1,4,3,0);
 cTensor_Image.SetElement(1,4,4,0);

 cTensor_Image.SetElement(2,0,0,0);
 cTensor_Image.SetElement(2,0,1,2);
 cTensor_Image.SetElement(2,0,2,0);
 cTensor_Image.SetElement(2,0,3,1);
 cTensor_Image.SetElement(2,0,4,1);

 cTensor_Image.SetElement(2,1,0,2);
 cTensor_Image.SetElement(2,1,1,0);
 cTensor_Image.SetElement(2,1,2,2);
 cTensor_Image.SetElement(2,1,3,1);
 cTensor_Image.SetElement(2,1,4,1);

 cTensor_Image.SetElement(2,2,0,2);
 cTensor_Image.SetElement(2,2,1,0);
 cTensor_Image.SetElement(2,2,2,1);
 cTensor_Image.SetElement(2,2,3,2);
 cTensor_Image.SetElement(2,2,4,1);

 cTensor_Image.SetElement(2,3,0,0);
 cTensor_Image.SetElement(2,3,1,1);
 cTensor_Image.SetElement(2,3,2,0);
 cTensor_Image.SetElement(2,3,3,1);
 cTensor_Image.SetElement(2,3,4,2);

 cTensor_Image.SetElement(2,4,0,1);
 cTensor_Image.SetElement(2,4,1,0);
 cTensor_Image.SetElement(2,4,2,1);
 cTensor_Image.SetElement(2,4,3,2);
 cTensor_Image.SetElement(2,4,4,1);

 //делаем свёртку
 CTensor<type_t> cTensor_Convolution(2,3,3);
 CTensorConv<type_t>::ForwardConvolution(cTensor_Convolution,cTensor_Image,cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_Bias,2,2,1,1);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //cTensor_Convolution.Print("Результат свёртки 1");

 //TODO: добавить проверку

 return(true);
}

//----------------------------------------------------------------------------------------------------
//протестировать создание поправокк c шагом и дополнением
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensorTest<type_t>::TestCreateDeltaWeightAndBiasWithStepAndPadding(void)
{
 SYSTEM::PutMessageToConsole("Тест функции CreateDeltaWeightAndBias с шагом и дополнением.");
 CTensor<type_t> cTensor_Image(1,7,7);
 //входное изображение
 for(uint32_t y=0;y<7;y++)
 {
  for(uint32_t x=0;x<7;x++)
  {
   cTensor_Image.SetElement(0,y,x,x+y*7+1);
  }
 }
 //дельта
 CTensor<type_t> cTensor_Delta(1,4,4);
 for(uint32_t y=0;y<4;y++)
 {
  for(uint32_t x=0;x<4;x++)
  {
   cTensor_Delta.SetElement(0,y,x,x+y*4+1);
  }
 }
 //создаём тензор поправок ядер
 const uint32_t kernel_x=5;
 const uint32_t kernel_y=5;
 const uint32_t kernel_z=1;
 const uint32_t kernel_amount=1;
 CTensor<type_t> cTensor_dKernel(1,kernel_amount,kernel_x*kernel_y*kernel_z);
 //создаём тензор поправок смещений
 CTensor<type_t> cTensor_dBias=CTensor<type_t>(kernel_amount,1,1);
 cTensor_dBias.Zero();
 CTensor<type_t> cTensor_Tmp;
 CTensorConv<type_t>::CreateDeltaWeightAndBias(cTensor_dKernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_dBias,cTensor_Image,cTensor_Delta,2,2,2,2,cTensor_Tmp);

 //проверочный тензор
 CTensor<type_t> cTensor_Control(1,5,5);

 cTensor_Control.SetElement(0,0,0,2031);
 cTensor_Control.SetElement(0,0,1,2130);
 cTensor_Control.SetElement(0,0,2,2746);
 cTensor_Control.SetElement(0,0,3,1968);
 cTensor_Control.SetElement(0,0,4,2058);

 cTensor_Control.SetElement(0,1,0,2724);
 cTensor_Control.SetElement(0,1,1,2823);
 cTensor_Control.SetElement(0,1,2,3628);
 cTensor_Control.SetElement(0,1,3,2598);
 cTensor_Control.SetElement(0,1,4,2688);

 cTensor_Control.SetElement(0,2,0,3448);
 cTensor_Control.SetElement(0,2,1,3556);
 cTensor_Control.SetElement(0,2,2,4560);
 cTensor_Control.SetElement(0,2,3,3256);
 cTensor_Control.SetElement(0,2,4,3352);

 cTensor_Control.SetElement(0,3,0,1860);
 cTensor_Control.SetElement(0,3,1,1923);
 cTensor_Control.SetElement(0,3,2,2428);
 cTensor_Control.SetElement(0,3,3,1698);
 cTensor_Control.SetElement(0,3,4,1752);

 cTensor_Control.SetElement(0,4,0,2301);
 cTensor_Control.SetElement(0,4,1,2364);
 cTensor_Control.SetElement(0,4,2,2974);
 cTensor_Control.SetElement(0,4,3,2076);
 cTensor_Control.SetElement(0,4,4,2130);

 CTensor<type_t> cTensor_dKernelBased=CTensor<type_t>(kernel_z,kernel_y,kernel_x);

 uint32_t pos=0;
 for(uint32_t z=0;z<kernel_z;z++)
 {
  for(uint32_t y=0;y<kernel_y;y++)
  {
   for(uint32_t x=0;x<kernel_x;x++,pos++)
   {
    cTensor_dKernelBased.SetElement(z,y,x,cTensor_dKernel.GetElement(0,0,pos));
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
 const uint32_t kernel_x=3;
 const uint32_t kernel_y=3;
 const uint32_t kernel_z=1;
 const uint32_t kernel_amount=1;

 type_t kernel_a[kernel_x*kernel_y*kernel_z]={1,4,1, 1,4,3, 3,3,1};

 //создаём тензор ядер
 CTensor<type_t> cTensor_Kernel=CTensor<type_t>(1,kernel_amount,kernel_x*kernel_y*kernel_z);
 for(uint32_t n=0;n<kernel_x*kernel_y*kernel_z;n++)
 {
  cTensor_Kernel.SetElement(0,0,n,kernel_a[n]);
 }

 //создаём тензор смещений
 CTensor<type_t> cTensor_Bias=CTensor<type_t>(kernel_amount,1,1);
 cTensor_Bias.Zero();

 //создаём изображение
 CTensor<type_t> cTensor_Image(1,4,4);

 cTensor_Image.SetElement(0,0,0,4);
 cTensor_Image.SetElement(0,0,1,5);
 cTensor_Image.SetElement(0,0,2,8);
 cTensor_Image.SetElement(0,0,3,7);

 cTensor_Image.SetElement(0,1,0,1);
 cTensor_Image.SetElement(0,1,1,8);
 cTensor_Image.SetElement(0,1,2,8);
 cTensor_Image.SetElement(0,1,3,8);

 cTensor_Image.SetElement(0,2,0,3);
 cTensor_Image.SetElement(0,2,1,6);
 cTensor_Image.SetElement(0,2,2,6);
 cTensor_Image.SetElement(0,2,3,4);

 cTensor_Image.SetElement(0,3,0,6);
 cTensor_Image.SetElement(0,3,1,5);
 cTensor_Image.SetElement(0,3,2,7);
 cTensor_Image.SetElement(0,3,3,8);

 //делаем прямую свёртку
 CTensor<type_t> cTensor_Convolution(1,2,2);
 CTensorConv<type_t>::ForwardConvolution(cTensor_Convolution,cTensor_Image,cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_Bias,1,1,0,0);

 CTensor<type_t> cTensor_dOut(1,2,2);
 cTensor_dOut.SetElement(0,0,0,2);
 cTensor_dOut.SetElement(0,0,1,1);

 cTensor_dOut.SetElement(0,1,0,4);
 cTensor_dOut.SetElement(0,1,1,4);

 //делаем обратное распространение
 CTensor<type_t> cTensor_Delta(1,4,4);

 CTensor<type_t> cTensor_dBias=CTensor<type_t>(kernel_amount,1,1);
 cTensor_Bias.Zero();

 CTensorConv<type_t>::BackwardConvolution(cTensor_Delta,cTensor_dOut,cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_dBias,1,1,0,0);
 cTensor_Delta.Print("Тензор поправок дельта");

 //создаём вектор тензоров поправок к ядрам
 CTensor<type_t> cTensor_dKernel=CTensor<type_t>(1,kernel_amount,kernel_x*kernel_y*kernel_z);
 cTensor_dKernel.Zero();
 //создаём вектор поправок смещений
 cTensor_dBias.Zero();
 CTensor<type_t> cTensor_Tmp;
 CTensorConv<type_t>::CreateDeltaWeightAndBias(cTensor_dKernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_dBias,cTensor_Image,cTensor_dOut,1,1,0,0,cTensor_Tmp);

 //проверяем
 CTensor<type_t> cTensor_Control(1,3,3);

 cTensor_Control.SetElement(0,0,0,49);
 cTensor_Control.SetElement(0,0,1,82);
 cTensor_Control.SetElement(0,0,2,87);

 cTensor_Control.SetElement(0,1,0,46);
 cTensor_Control.SetElement(0,1,1,72);
 cTensor_Control.SetElement(0,1,2,64);

 cTensor_Control.SetElement(0,2,0,56);
 cTensor_Control.SetElement(0,2,1,66);
 cTensor_Control.SetElement(0,2,2,76);

 CTensor<type_t> cTensor_dKernelBased=CTensor<type_t>(kernel_z,kernel_y,kernel_x);

 uint32_t pos=0;
 for(uint32_t z=0;z<kernel_z;z++)
 {
  for(uint32_t y=0;y<kernel_y;y++)
  {
   for(uint32_t x=0;x<kernel_x;x++,pos++)
   {
    cTensor_dKernelBased.SetElement(z,y,x,cTensor_dKernel.GetElement(0,0,pos));
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

 const uint32_t kernel_x=5;
 const uint32_t kernel_y=5;
 const uint32_t kernel_z=1;
 const uint32_t kernel_amount=1;

 //создаём тензор ядер
 CTensor<type_t> cTensor_Kernel=CTensor<type_t>(1,kernel_amount,kernel_x*kernel_y*kernel_z);
 uint32_t pos=0;
 for(uint32_t z=0;z<1;z++)
 {
  for(uint32_t y=0;y<5;y++)
  {
   for(uint32_t x=0;x<5;x++,pos++)
   {
    cTensor_Kernel.SetElement(0,0,pos,x+y*5+1);
   }
  }
 }
 //создаём тензор смещений
 CTensor<type_t> cTensor_Bias=CTensor<type_t>(kernel_amount,1,1);
 cTensor_Bias.Zero();
 //входное изображение
 CTensor<type_t> cTensor_Delta(1,4,4);
 for(uint32_t y=0;y<4;y++)
 {
  for(uint32_t x=0;x<4;x++)
  {
   cTensor_Delta.SetElement(0,y,x,x+y*4+1);
  }
 }
 //выходной тензор обратной свёртки
 CTensor<type_t> cTensor_Output(1,7,7);
 //проверочный тензор обратной свёртки
 CTensor<type_t> cTensor_Control(1,7,7);
 //выполняем обратную свёртку
 CTensorConv<type_t>::BackwardConvolution(cTensor_Output,cTensor_Delta,cTensor_Kernel,kernel_x,kernel_y,kernel_z,kernel_amount,cTensor_Bias,2,2,2,2);

 cTensor_Control.SetElement(0,0,0,56);
 cTensor_Control.SetElement(0,0,1,70);
 cTensor_Control.SetElement(0,0,2,124);
 cTensor_Control.SetElement(0,0,3,102);
 cTensor_Control.SetElement(0,0,4,172);
 cTensor_Control.SetElement(0,0,5,134);
 cTensor_Control.SetElement(0,0,6,156);

 cTensor_Control.SetElement(0,1,0,126);
 cTensor_Control.SetElement(0,1,1,140);
 cTensor_Control.SetElement(0,1,2,244);
 cTensor_Control.SetElement(0,1,3,192);
 cTensor_Control.SetElement(0,1,4,322);
 cTensor_Control.SetElement(0,1,5,244);
 cTensor_Control.SetElement(0,1,6,266);

 cTensor_Control.SetElement(0,2,0,233);
 cTensor_Control.SetElement(0,2,1,266);
 cTensor_Control.SetElement(0,2,2,450);
 cTensor_Control.SetElement(0,2,3,344);
 cTensor_Control.SetElement(0,2,4,567);
 cTensor_Control.SetElement(0,2,5,422);
 cTensor_Control.SetElement(0,2,6,467);

 cTensor_Control.SetElement(0,3,0,318);
 cTensor_Control.SetElement(0,3,1,348);
 cTensor_Control.SetElement(0,3,2,556);
 cTensor_Control.SetElement(0,3,3,400);
 cTensor_Control.SetElement(0,3,4,634);
 cTensor_Control.SetElement(0,3,5,452);
 cTensor_Control.SetElement(0,3,6,490);

 cTensor_Control.SetElement(0,4,0,521);
 cTensor_Control.SetElement(0,4,1,578);
 cTensor_Control.SetElement(0,4,2,918);
 cTensor_Control.SetElement(0,4,3,656);
 cTensor_Control.SetElement(0,4,4,1035);
 cTensor_Control.SetElement(0,4,5,734);
 cTensor_Control.SetElement(0,4,6,803);

 cTensor_Control.SetElement(0,5,0,510);
 cTensor_Control.SetElement(0,5,1,556);
 cTensor_Control.SetElement(0,5,2,868);
 cTensor_Control.SetElement(0,5,3,608);
 cTensor_Control.SetElement(0,5,4,946);
 cTensor_Control.SetElement(0,5,5,660);
 cTensor_Control.SetElement(0,5,6,714);

 cTensor_Control.SetElement(0,6,0,740);
 cTensor_Control.SetElement(0,6,1,786);
 cTensor_Control.SetElement(0,6,2,1228);
 cTensor_Control.SetElement(0,6,3,858);
 cTensor_Control.SetElement(0,6,4,1336);
 cTensor_Control.SetElement(0,6,5,930);
 cTensor_Control.SetElement(0,6,6,984);

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
 CTensor<type_t> cTensorA(1,2,2);
 CTensor<type_t> cTensorB(1,2,2);
 CTensor<type_t> cTensorC(1,2,2);

 cTensorA.SetElement(0,0,0,1);
 cTensorA.SetElement(0,0,1,2);
 cTensorA.SetElement(0,1,0,3);
 cTensorA.SetElement(0,1,1,4);

 cTensorB.SetElement(0,0,0,1);
 cTensorB.SetElement(0,0,1,2);
 cTensorB.SetElement(0,1,0,3);
 cTensorB.SetElement(0,1,1,4);

 SYSTEM::PutMessageToConsole("Тест заполнения тензора по элементам.");
 //проверка на заполнение тензора
 if (cTensorA.GetElement(0,0,0)!=1) return(false);
 if (cTensorA.GetElement(0,0,1)!=2) return(false);
 if (cTensorA.GetElement(0,1,0)!=3) return(false);
 if (cTensorA.GetElement(0,1,1)!=4) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //умножение на число справа
 SYSTEM::PutMessageToConsole("Тест умножения на число справа.");
 cTensorC=cTensorA*static_cast<type_t>(2.0);
 if (cTensorC.GetElement(0,0,0)!=2) return(false);
 if (cTensorC.GetElement(0,0,1)!=4) return(false);
 if (cTensorC.GetElement(0,1,0)!=6) return(false);
 if (cTensorC.GetElement(0,1,1)!=8) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //умножение на число слева
 SYSTEM::PutMessageToConsole("Тест умножения на число слева.");
 cTensorC=static_cast<type_t>(2.0)*cTensorA;
 if (cTensorC.GetElement(0,0,0)!=2) return(false);
 if (cTensorC.GetElement(0,0,1)!=4) return(false);
 if (cTensorC.GetElement(0,1,0)!=6) return(false);
 if (cTensorC.GetElement(0,1,1)!=8) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //умножение тензоров
 SYSTEM::PutMessageToConsole("Тест умножения тензоров.");
 cTensorC=cTensorA*cTensorB;
 cTensorC.Print("Mul");
 if (cTensorC.GetElement(0,0,0)!=7) return(false);
 if (cTensorC.GetElement(0,0,1)!=10) return(false);
 if (cTensorC.GetElement(0,1,0)!=15) return(false);
 if (cTensorC.GetElement(0,1,1)!=22) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //вычитание тензоров
 SYSTEM::PutMessageToConsole("Тест вычитания тензоров.");
 cTensorC=cTensorA-cTensorB;
 if (cTensorC.GetElement(0,0,0)!=0) return(false);
 if (cTensorC.GetElement(0,0,1)!=0) return(false);
 if (cTensorC.GetElement(0,1,0)!=0) return(false);
 if (cTensorC.GetElement(0,1,1)!=0) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //сложение тензоров
 SYSTEM::PutMessageToConsole("Тест сложения тензоров.");
 cTensorC=cTensorA+cTensorB;
 if (cTensorC.GetElement(0,0,0)!=2) return(false);
 if (cTensorC.GetElement(0,0,1)!=4) return(false);
 if (cTensorC.GetElement(0,1,0)!=6) return(false);
 if (cTensorC.GetElement(0,1,1)!=8) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 //транспонирование тензора
 SYSTEM::PutMessageToConsole("Тест транспонирования тензоров.");
 cTensorC=CTensorMath<type_t>::Transpose(cTensorA);
 if (cTensorC.GetElement(0,0,0)!=1) return(false);
 if (cTensorC.GetElement(0,0,1)!=3) return(false);
 if (cTensorC.GetElement(0,1,0)!=2) return(false);
 if (cTensorC.GetElement(0,1,1)!=4) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

/* //построчное скалярное произведение тензоров
 CTensor<type_t> cTensorD(1,2,1);
 TensorColumnScalarProduction(cTensorD,cTensorA,cTensorB);
 if (cTensorD.GetElement(0,0,0)!=5) return(false);
 if (cTensorD.GetElement(0,1,0)!=25) return(false);
 */

 //поэлементное умножение тензоров
 SYSTEM::PutMessageToConsole("Тест поэлементного умножения тензоров.");
 CTensor<type_t> cTensorE(1,2,2);
 CTensorMath<type_t>::TensorItemProduction(cTensorE,cTensorA,cTensorB);
 if (cTensorE.GetElement(0,0,0)!=1) return(false);
 if (cTensorE.GetElement(0,0,1)!=4) return(false);
 if (cTensorE.GetElement(0,1,0)!=9) return(false);
 if (cTensorE.GetElement(0,1,1)!=16) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");
 SYSTEM::PutMessageToConsole("");

 if (TestForwardConvolution()==false) return(false);
 if (TestForwardConvolutionWithStepAndPadding()==false) return(false);
 if (TestCreateDeltaWeightAndBiasWithStepAndPadding()==false) return(false);
 if (TestCreateDeltaWeightAndBias()==false) return(false);
 if (TestBackwardConvolutionWithStepAndPadding()==false) return(false);

 /*

 {
  SYSTEM::PutMessageToConsole("Тест функции ForwardConvolution (0) .");
  CTensor<type_t> cTensor_KernelA(2,2,2);
  //ядро A
  cTensor_KernelA.SetElement(0,0,0,1);
  cTensor_KernelA.SetElement(0,0,1,2);

  cTensor_KernelA.SetElement(0,1,0,3);
  cTensor_KernelA.SetElement(0,1,1,4);

  cTensor_KernelA.SetElement(1,0,0,5);
  cTensor_KernelA.SetElement(1,0,1,6);

  cTensor_KernelA.SetElement(1,1,0,7);
  cTensor_KernelA.SetElement(1,1,1,8);

  //создаём тензор тензоров ядер
  CTensor<type_t> cTensor_Kernel=CTensor<type_t>(1,1,cTensor_KernelA.GetSizeX()*cTensor_KernelA.GetSizeY()*cTensor_KernelA.GetSizeZ());
  uint32_t pos=0;
  for(uint32_t z=0;z<cTensor_KernelA.GetSizeZ();z++)
  {
   for(uint32_t y=0;y<cTensor_KernelA.GetSizeY();y++)
   {
    for(uint32_t x=0;x<cTensor_KernelA.GetSizeX();x++,pos++)
    {
     cTensor_Kernel.SetElement(0,0,pos,cTensor_KernelA.GetElement(z,y,x));
    }
   }
  }

  //создаём тензор смещений
  CTensor<type_t> cTensor_Bias=CTensor<type_t>(cTensor_Kernel.GetSizeY(),1,1);
  cTensor_Bias.Zero();

  //входное изображение
  CTensor<type_t> cTensor_Image(2,3,4);

  cTensor_Image.SetElement(0,0,0,1);
  cTensor_Image.SetElement(0,0,1,2);
  cTensor_Image.SetElement(0,0,2,3);
  cTensor_Image.SetElement(0,0,3,4);
  cTensor_Image.SetElement(0,1,0,5);
  cTensor_Image.SetElement(0,1,1,6);
  cTensor_Image.SetElement(0,1,2,7);
  cTensor_Image.SetElement(0,1,3,8);
  cTensor_Image.SetElement(0,2,0,9);
  cTensor_Image.SetElement(0,2,1,10);
  cTensor_Image.SetElement(0,2,2,11);
  cTensor_Image.SetElement(0,2,3,12);

  cTensor_Image.SetElement(1,0,0,12);
  cTensor_Image.SetElement(1,0,1,11);
  cTensor_Image.SetElement(1,0,2,10);
  cTensor_Image.SetElement(1,0,3,9);
  cTensor_Image.SetElement(1,1,0,8);
  cTensor_Image.SetElement(1,1,1,7);
  cTensor_Image.SetElement(1,1,2,6);
  cTensor_Image.SetElement(1,1,3,5);
  cTensor_Image.SetElement(1,2,0,4);
  cTensor_Image.SetElement(1,2,1,3);
  cTensor_Image.SetElement(1,2,2,2);
  cTensor_Image.SetElement(1,2,3,1);


  //выходной тензор свёртки
  CTensor<type_t> cTensor_Output(1,2,3);
  //проверочный тензор свёртки
  CTensor<type_t> cTensor_Control(1,2,3);

  cTensor_Control.SetElement(0,0,0,282);
  cTensor_Control.SetElement(0,0,1,266);
  cTensor_Control.SetElement(0,0,2,250);

  cTensor_Control.SetElement(0,1,0,218);
  cTensor_Control.SetElement(0,1,1,202);
  cTensor_Control.SetElement(0,1,2,186);


  //выполняем прямую свёртку
  CTensorConv<type_t>::ForwardConvolution(cTensor_Output,cTensor_Image,cTensor_Kernel,cTensor_KernelA.GetSizeX(),cTensor_KernelA.GetSizeY(),cTensor_KernelA.GetSizeZ(),cTensor_Kernel.GetSizeY(),cTensor_Bias,1,1,0,0);
  //сравниваем полученный тензор
  if (cTensor_Output.Compare(cTensor_Control,"")==false) return(false);
  SYSTEM::PutMessageToConsole("Успешно.");
  SYSTEM::PutMessageToConsole("");
 }
























 {
  SYSTEM::PutMessageToConsole("Тест функции BackwardConvolution.");
  CTensor<type_t> cTensor_KernelA(3,2,2);
  CTensor<type_t> cTensor_KernelB(3,2,2);
  //ядро A
  cTensor_KernelA.SetElement(0,0,0,1);
  cTensor_KernelA.SetElement(0,0,1,2);

  cTensor_KernelA.SetElement(0,1,0,3);
  cTensor_KernelA.SetElement(0,1,1,4);


  cTensor_KernelA.SetElement(1,0,0,5);
  cTensor_KernelA.SetElement(1,0,1,6);

  cTensor_KernelA.SetElement(1,1,0,7);
  cTensor_KernelA.SetElement(1,1,1,8);


  cTensor_KernelA.SetElement(2,0,0,9);
  cTensor_KernelA.SetElement(2,0,1,10);

  cTensor_KernelA.SetElement(2,1,0,11);
  cTensor_KernelA.SetElement(2,1,1,12);
  //ядро B
  cTensor_KernelB.SetElement(0,0,0,12);
  cTensor_KernelB.SetElement(0,0,1,11);

  cTensor_KernelB.SetElement(0,1,0,10);
  cTensor_KernelB.SetElement(0,1,1,9);


  cTensor_KernelB.SetElement(1,0,0,8);
  cTensor_KernelB.SetElement(1,0,1,7);

  cTensor_KernelB.SetElement(1,1,0,6);
  cTensor_KernelB.SetElement(1,1,1,5);


  cTensor_KernelB.SetElement(2,0,0,4);
  cTensor_KernelB.SetElement(2,0,1,3);

  cTensor_KernelB.SetElement(2,1,0,2);
  cTensor_KernelB.SetElement(2,1,1,1);
  //создаём вектор тензоров ядер
  std::vector<CTensor<type_t>> cTensor_Kernel;
  cTensor_Kernel.push_back(cTensor_KernelA);
  cTensor_Kernel.push_back(cTensor_KernelB);

  //входное изображение
  CTensor<type_t> cTensor_Delta(2,3,3);

  cTensor_Delta.SetElement(0,0,0,1);
  cTensor_Delta.SetElement(0,0,1,2);
  cTensor_Delta.SetElement(0,0,2,3);
  cTensor_Delta.SetElement(0,1,0,4);
  cTensor_Delta.SetElement(0,1,1,5);
  cTensor_Delta.SetElement(0,1,2,6);
  cTensor_Delta.SetElement(0,2,0,7);
  cTensor_Delta.SetElement(0,2,1,8);
  cTensor_Delta.SetElement(0,2,2,9);

  cTensor_Delta.SetElement(1,0,0,10);
  cTensor_Delta.SetElement(1,0,1,11);
  cTensor_Delta.SetElement(1,0,2,12);
  cTensor_Delta.SetElement(1,1,0,13);
  cTensor_Delta.SetElement(1,1,1,14);
  cTensor_Delta.SetElement(1,1,2,15);
  cTensor_Delta.SetElement(1,2,0,16);
  cTensor_Delta.SetElement(1,2,1,17);
  cTensor_Delta.SetElement(1,2,2,18);

  //выходной тензор обратной свёртки
  CTensor<type_t> cTensor_Output(3,4,4);
  //проверочный тензор обратной свёртки
  CTensor<type_t> cTensor_Control(3,4,4);
  //выполняем обратную свёртку
  std::vector<type_t> back_bias(cTensor_Kernel.size(),0);
  CTensorConv<type_t>::BackwardConvolution(cTensor_Output,cTensor_Delta,cTensor_Kernel,back_bias,1,1,0,0);

  cTensor_Control.SetElement(0,0,0,121);
  cTensor_Control.SetElement(0,0,1,246);
  cTensor_Control.SetElement(0,0,2,272);
  cTensor_Control.SetElement(0,0,3,138);

  cTensor_Control.SetElement(0,1,0,263);
  cTensor_Control.SetElement(0,1,1,534);
  cTensor_Control.SetElement(0,1,2,586);
  cTensor_Control.SetElement(0,1,3,297);

  cTensor_Control.SetElement(0,2,0,341);
  cTensor_Control.SetElement(0,2,1,690);
  cTensor_Control.SetElement(0,2,2,742);
  cTensor_Control.SetElement(0,2,3,375);

  cTensor_Control.SetElement(0,3,0,181);
  cTensor_Control.SetElement(0,3,1,366);
  cTensor_Control.SetElement(0,3,2,392);
  cTensor_Control.SetElement(0,3,3,198);


  cTensor_Control.SetElement(1,0,0,85);
  cTensor_Control.SetElement(1,0,1,174);
  cTensor_Control.SetElement(1,0,2,200);
  cTensor_Control.SetElement(1,0,3,102);

  cTensor_Control.SetElement(1,1,0,191);
  cTensor_Control.SetElement(1,1,1,390);
  cTensor_Control.SetElement(1,1,2,442);
  cTensor_Control.SetElement(1,1,3,225);

  cTensor_Control.SetElement(1,2,0,269);
  cTensor_Control.SetElement(1,2,1,546);
  cTensor_Control.SetElement(1,2,2,598);
  cTensor_Control.SetElement(1,2,3,303);

  cTensor_Control.SetElement(1,3,0,145);
  cTensor_Control.SetElement(1,3,1,294);
  cTensor_Control.SetElement(1,3,2,320);
  cTensor_Control.SetElement(1,3,3,162);


  cTensor_Control.SetElement(2,0,0,49);
  cTensor_Control.SetElement(2,0,1,102);
  cTensor_Control.SetElement(2,0,2,128);
  cTensor_Control.SetElement(2,0,3,66);

  cTensor_Control.SetElement(2,1,0,119);
  cTensor_Control.SetElement(2,1,1,246);
  cTensor_Control.SetElement(2,1,2,298);
  cTensor_Control.SetElement(2,1,3,153);

  cTensor_Control.SetElement(2,2,0,197);
  cTensor_Control.SetElement(2,2,1,402);
  cTensor_Control.SetElement(2,2,2,454);
  cTensor_Control.SetElement(2,2,3,231);

  cTensor_Control.SetElement(2,3,0,109);
  cTensor_Control.SetElement(2,3,1,222);
  cTensor_Control.SetElement(2,3,2,248);
  cTensor_Control.SetElement(2,3,3,126);

  //сравниваем полученный тензор
  if (cTensor_Output.Compare(cTensor_Control,"")==false) return(false);
  SYSTEM::PutMessageToConsole("Успешно.");
  SYSTEM::PutMessageToConsole("");
 }

 {
  SYSTEM::PutMessageToConsole("Тест функции TransponseMul.");
  CTensor<type_t> cTensorA(1,3,2);
  CTensor<type_t> cTensorAt(1,2,3);
  cTensorA.SetElement(0,0,0,1);
  cTensorA.SetElement(0,0,1,2);
  cTensorA.SetElement(0,1,0,3);
  cTensorA.SetElement(0,1,1,4);
  cTensorA.SetElement(0,2,0,5);
  cTensorA.SetElement(0,2,1,6);

  CTensor<type_t> cTensorB(1,3,4);
  cTensorB.SetElement(0,0,0,7);
  cTensorB.SetElement(0,0,1,8);
  cTensorB.SetElement(0,0,2,13);
  cTensorB.SetElement(0,0,3,16);

  cTensorB.SetElement(0,1,0,9);
  cTensorB.SetElement(0,1,1,10);
  cTensorB.SetElement(0,1,2,14);
  cTensorB.SetElement(0,1,3,17);

  cTensorB.SetElement(0,2,0,11);
  cTensorB.SetElement(0,2,1,12);
  cTensorB.SetElement(0,2,2,15);
  cTensorB.SetElement(0,2,3,18);

  CTensor<type_t> cTensorC(1,2,4);
  CTensor<type_t> cTensorD(1,2,4);

  CTensorMath<type_t>::TransponseMul(cTensorC,cTensorA,cTensorB);
  CTensorMath<type_t>::Transponse(cTensorAt,cTensorA);

  CTensorMath<type_t>::Mul(cTensorD,cTensorAt,cTensorB);

  //сравниваем
  for(uint32_t x=0;x<cTensorC.GetSizeX();x++)
  {
   for(uint32_t y=0;y<cTensorC.GetSizeY();y++)
   {
    type_t e1=cTensorC.GetElement(0,y,x);
    type_t e2=cTensorD.GetElement(0,y,x);

    if (fabs(e1-e2)>0.00001) return(false);
   }
  }
  SYSTEM::PutMessageToConsole("Успешно.");
 }

 {
  SYSTEM::PutMessageToConsole("Тест функции SummXY.");
  CTensor<type_t> cTensor_Input(3,100,200);
  CTensor<type_t> cTensor_Output(3,1,1);
  CTensor<type_t> cTensor_Control(3,1,1);

  for(uint32_t z=0;z<cTensor_Input.GetSizeZ();z++)
  {
   type_t summ=0;
   for(uint32_t y=0;y<cTensor_Input.GetSizeY();y++)
   {
    for(uint32_t x=0;x<cTensor_Input.GetSizeX();x++)
	{
     type_t e=(x-y+z*2)%10;
	 summ+=e;
	 cTensor_Input.SetElement(z,y,x,e);
	}
   }
   cTensor_Control.SetElement(z,0,0,summ);
  }

  CTensorMath<type_t>::SummXY(cTensor_Output,cTensor_Input);

  //сравниваем полученный тензор
  if (cTensor_Output.Compare(cTensor_Control,"")==false) return(false);
  SYSTEM::PutMessageToConsole("Успешно.");
  SYSTEM::PutMessageToConsole("");
 }*/
 return(true);
}

#endif
