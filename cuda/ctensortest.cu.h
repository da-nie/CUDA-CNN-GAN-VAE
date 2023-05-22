#ifndef C_TENSOR_TEST_H
#define C_TENSOR_TEST_H

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
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------------------------

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
 {
  SYSTEM::PutMessageToConsole("Тест функции CreateDeltaWeightAndBias.");
  {
   CTensor<type_t> cTensor_KernelA(3,3,3);
   CTensor<type_t> cTensor_KernelB(3,3,3);

   //ядро A
   cTensor_KernelA.SetElement(0,0,0,-1);
   cTensor_KernelA.SetElement(0,0,1,1);
   cTensor_KernelA.SetElement(0,0,2,1);

   cTensor_KernelA.SetElement(0,1,0,-1);
   cTensor_KernelA.SetElement(0,1,1,1);
   cTensor_KernelA.SetElement(0,1,2,1);

   cTensor_KernelA.SetElement(0,2,0,0);
   cTensor_KernelA.SetElement(0,2,1,0);
   cTensor_KernelA.SetElement(0,2,2,1);


   cTensor_KernelA.SetElement(1,0,0,0);
   cTensor_KernelA.SetElement(1,0,1,-1);
   cTensor_KernelA.SetElement(1,0,2,1);

   cTensor_KernelA.SetElement(1,1,0,-1);
   cTensor_KernelA.SetElement(1,1,1,0);
   cTensor_KernelA.SetElement(1,1,2,-1);

   cTensor_KernelA.SetElement(1,2,0,1);
   cTensor_KernelA.SetElement(1,2,1,0);
   cTensor_KernelA.SetElement(1,2,2,0);


   cTensor_KernelA.SetElement(2,0,0,1);
   cTensor_KernelA.SetElement(2,0,1,-1);
   cTensor_KernelA.SetElement(2,0,2,0);

   cTensor_KernelA.SetElement(2,1,0,-1);
   cTensor_KernelA.SetElement(2,1,1,0);
   cTensor_KernelA.SetElement(2,1,2,-1);

   cTensor_KernelA.SetElement(2,2,0,0);
   cTensor_KernelA.SetElement(2,2,1,1);
   cTensor_KernelA.SetElement(2,2,2,-1);

   //ядро B
   cTensor_KernelB.SetElement(0,0,0,0);
   cTensor_KernelB.SetElement(0,0,1,0);
   cTensor_KernelB.SetElement(0,0,2,-1);

   cTensor_KernelB.SetElement(0,1,0,1);
   cTensor_KernelB.SetElement(0,1,1,0);
   cTensor_KernelB.SetElement(0,1,2,0);

   cTensor_KernelB.SetElement(0,2,0,0);
   cTensor_KernelB.SetElement(0,2,1,-1);
   cTensor_KernelB.SetElement(0,2,2,0);


   cTensor_KernelB.SetElement(1,0,0,1);
   cTensor_KernelB.SetElement(1,0,1,-1);
   cTensor_KernelB.SetElement(1,0,2,0);

   cTensor_KernelB.SetElement(1,1,0,-1);
   cTensor_KernelB.SetElement(1,1,1,1);
   cTensor_KernelB.SetElement(1,1,2,0);

   cTensor_KernelB.SetElement(1,2,0,-1);
   cTensor_KernelB.SetElement(1,2,1,1);
   cTensor_KernelB.SetElement(1,2,2,1);


   cTensor_KernelB.SetElement(2,0,0,1);
   cTensor_KernelB.SetElement(2,0,1,0);
   cTensor_KernelB.SetElement(2,0,2,1);

   cTensor_KernelB.SetElement(2,1,0,1);
   cTensor_KernelB.SetElement(2,1,1,-1);
   cTensor_KernelB.SetElement(2,1,2,1);

   cTensor_KernelB.SetElement(2,2,0,-1);
   cTensor_KernelB.SetElement(2,2,1,0);
   cTensor_KernelB.SetElement(2,2,2,0);

   //создаём вектор тензоров ядер
   std::vector<CTensor<type_t>> cTensor_Kernel;
   cTensor_Kernel.push_back(cTensor_KernelA);
   cTensor_Kernel.push_back(cTensor_KernelB);
   //создаём вектор смещений
   std::vector<type_t> bias;
   bias.push_back(1);
   bias.push_back(0);

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
   CTensorConv<type_t>::ForwardConvolution(cTensor_Convolution,cTensor_Image,cTensor_Kernel,bias,2,2,1,1);

   cTensor_Convolution.Print("Результат свёртки 1");
  }
  {
   CTensor<type_t> cTensor_KernelA(1,3,3);
   //ядро A
   cTensor_KernelA.SetElement(0,0,0,1);
   cTensor_KernelA.SetElement(0,0,1,4);
   cTensor_KernelA.SetElement(0,0,2,1);

   cTensor_KernelA.SetElement(0,1,0,1);
   cTensor_KernelA.SetElement(0,1,1,4);
   cTensor_KernelA.SetElement(0,1,2,3);

   cTensor_KernelA.SetElement(0,2,0,3);
   cTensor_KernelA.SetElement(0,2,1,3);
   cTensor_KernelA.SetElement(0,2,2,1);

   //создаём вектор тензоров ядер
   std::vector<CTensor<type_t>> cTensor_Kernel;
   cTensor_Kernel.push_back(cTensor_KernelA);
   //создаём вектор смещений
   std::vector<type_t> bias;
   bias.push_back(0);

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
   CTensorConv<type_t>::ForwardConvolution(cTensor_Convolution,cTensor_Image,cTensor_Kernel,bias,1,1,0,0);

   cTensor_Convolution.Print("Результат свёртки 2");

   CTensor<type_t> cTensor_dOut(1,2,2);
   cTensor_dOut.SetElement(0,0,0,2);
   cTensor_dOut.SetElement(0,0,1,1);

   cTensor_dOut.SetElement(0,1,0,4);
   cTensor_dOut.SetElement(0,1,1,4);

   //делаем обратное распространение
   CTensor<type_t> cTensor_Delta(1,4,4);

   std::vector<type_t> back_bias(cTensor_Kernel.size(),0);
   CTensorConv<type_t>::BackwardConvolution(cTensor_Delta,cTensor_dOut,cTensor_Kernel,back_bias);
   cTensor_Delta.Print("Тензор поправок дельта");

   //создаём вектор тензоров поправок к ядрам
   std::vector<CTensor<type_t>> cTensor_dKernel;
   cTensor_KernelA.Zero();
   cTensor_dKernel.push_back(cTensor_KernelA);
   for(size_t n=0;n<cTensor_dKernel.size();n++) cTensor_dKernel[n].Zero();
   cTensor_dKernel[0].Print("Тензор поправок ядра A после обнуления");
   //создаём вектор попрвок смещений
   std::vector<type_t> dbias;
   dbias.push_back(0);

   CTensorConv<type_t>::CreateDeltaWeightAndBias(cTensor_dKernel,dbias,cTensor_Image,cTensor_dOut);

   cTensor_dKernel[0].Print("Тензор поправок ядра A после расчёта");
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

   if (cTensor_dKernel[0].Compare(cTensor_Control,"")==false) return(false);
  }
  SYSTEM::PutMessageToConsole("Успешно.");
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
  CTensorConv<type_t>::BackwardConvolution(cTensor_Output,cTensor_Delta,cTensor_Kernel,back_bias);

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
 }


 {
  SYSTEM::PutMessageToConsole("Тест функции ForwardConvolution.");
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
  //создаём вектор смещений
  std::vector<type_t> bias;
  bias.push_back(0);
  bias.push_back(0);

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
  CTensorConv<type_t>::ForwardConvolution(cTensor_Output,cTensor_Image,cTensor_Kernel,bias,1,1,0,0);
  //сравниваем полученный тензор
  if (cTensor_Output.Compare(cTensor_Control,"")==false) return(false);
  SYSTEM::PutMessageToConsole("Успешно.");
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
  for(size_t x=0;x<cTensorC.GetSizeX();x++)
  {
   for(size_t y=0;y<cTensorC.GetSizeY();y++)
   {
    type_t e1=cTensorC.GetElement(0,y,x);
    type_t e2=cTensorD.GetElement(0,y,x);

    if (fabs(e1-e2)>0.00001) return(false);
   }
  }
  SYSTEM::PutMessageToConsole("Успешно.");
 }


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

 //умножение на число справа
 SYSTEM::PutMessageToConsole("Тест умножения на число справа.");
 cTensorC=cTensorA*static_cast<type_t>(2.0);
 if (cTensorC.GetElement(0,0,0)!=2) return(false);
 if (cTensorC.GetElement(0,0,1)!=4) return(false);
 if (cTensorC.GetElement(0,1,0)!=6) return(false);
 if (cTensorC.GetElement(0,1,1)!=8) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");

 //умножение на число слева
 SYSTEM::PutMessageToConsole("Тест умножения на число слева.");
 cTensorC=static_cast<type_t>(2.0)*cTensorA;
 if (cTensorC.GetElement(0,0,0)!=2) return(false);
 if (cTensorC.GetElement(0,0,1)!=4) return(false);
 if (cTensorC.GetElement(0,1,0)!=6) return(false);
 if (cTensorC.GetElement(0,1,1)!=8) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");

 //умножение тензоров
 SYSTEM::PutMessageToConsole("Тест умножения тензоров.");
 cTensorC=cTensorA*cTensorB;
 if (cTensorC.GetElement(0,0,0)!=7) return(false);
 if (cTensorC.GetElement(0,0,1)!=10) return(false);
 if (cTensorC.GetElement(0,1,0)!=15) return(false);
 if (cTensorC.GetElement(0,1,1)!=22) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");

 //вычитание тензоров
 SYSTEM::PutMessageToConsole("Тест вычитания тензоров.");
 cTensorC=cTensorA-cTensorB;
 if (cTensorC.GetElement(0,0,0)!=0) return(false);
 if (cTensorC.GetElement(0,0,1)!=0) return(false);
 if (cTensorC.GetElement(0,1,0)!=0) return(false);
 if (cTensorC.GetElement(0,1,1)!=0) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");

 //сложение тензоров
 SYSTEM::PutMessageToConsole("Тест сложения тензоров.");
 cTensorC=cTensorA+cTensorB;
 if (cTensorC.GetElement(0,0,0)!=2) return(false);
 if (cTensorC.GetElement(0,0,1)!=4) return(false);
 if (cTensorC.GetElement(0,1,0)!=6) return(false);
 if (cTensorC.GetElement(0,1,1)!=8) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");

 //транспонирование тензора
 SYSTEM::PutMessageToConsole("Тест транспонирования тензоров.");
 cTensorC=CTensorMath<type_t>::Transpose(cTensorA);
 if (cTensorC.GetElement(0,0,0)!=1) return(false);
 if (cTensorC.GetElement(0,0,1)!=3) return(false);
 if (cTensorC.GetElement(0,1,0)!=2) return(false);
 if (cTensorC.GetElement(0,1,1)!=4) return(false);
 SYSTEM::PutMessageToConsole("Успешно.");

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

 return(true);
}

#endif
