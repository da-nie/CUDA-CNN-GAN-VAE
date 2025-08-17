#ifndef C_MODEL_BASIC_GAN_H
#define C_MODEL_BASIC_GAN_H

//****************************************************************************************************
//Класс-основа для сетей GAN
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../../system/system.h"
#include "../../common/tga.h"
#include "../../common/ccolormodel.h"
#include "../ctimestamp.cu.h"


#include "cmodelmain.cu.h"

//****************************************************************************************************
//Класс-основа для сетей GAN
//****************************************************************************************************
template<class type_t>
class CModelBasicGAN:public CModelMain<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 protected:
  //-структуры------------------------------------------------------------------------------------------
  //-переменные-----------------------------------------------------------------------------------------
  uint32_t IMAGE_WIDTH;///<ширина входных изображений
  uint32_t IMAGE_HEIGHT;///<высота входных изображений
  uint32_t IMAGE_DEPTH;///<глубина входных изображений
  uint32_t NOISE_LAYER_SIDE_X;///<размерность стороны слоя шума по X
  uint32_t NOISE_LAYER_SIDE_Y;///<размерность стороны слоя шума по Y
  uint32_t NOISE_LAYER_SIDE_Z;///<размерность стороны слоя шума по Z
  uint32_t NOISE_LAYER_SIZE;///<размерность слоя шума

  uint32_t BATCH_AMOUNT;///<количество пакетов
  uint32_t BATCH_SIZE;///<размер пакета
  uint32_t ITERATION_OF_SAVE_IMAGE;///<какую итерацию сохранять изображения
  uint32_t ITERATION_OF_SAVE_NET;///<какую итерацию сохранять сеть

  double SPEED_DISCRIMINATOR;///<скорость обучения дискриминатора
  double SPEED_GENERATOR;///<скорость обучения генератора

  std::vector<std::shared_ptr<INetLayer<type_t> > > GeneratorNet;///<сеть генератора
  std::vector<std::shared_ptr<INetLayer<type_t> > > DiscriminatorNet;///<сеть дискриминатора

  CTensor<type_t> cTensor_Generator_Output;
  CTensor<type_t> cTensor_Discriminator_Output;
  CTensor<type_t> cTensor_Discriminator_Error;

  std::vector< std::vector<type_t> > RealImage;///<образы истинных изображений
  std::vector<uint32_t> RealImageIndex;///<индексы изображений в обучающем наборе

  using CModelMain<type_t>::STRING_BUFFER_SIZE;
  using CModelMain<type_t>::CUDA_PAUSE_MS;

  using CModelMain<type_t>::SafeLog;
  using CModelMain<type_t>::CrossEntropy;
  using CModelMain<type_t>::IsExit;
  using CModelMain<type_t>::SetExitState;
  using CModelMain<type_t>::LoadMNISTImage;
  using CModelMain<type_t>::LoadImage;
  using CModelMain<type_t>::SaveNetLayers;
  using CModelMain<type_t>::LoadNetLayers;
  using CModelMain<type_t>::SaveNetLayersTrainingParam;
  using CModelMain<type_t>::LoadNetLayersTrainingParam;
  using CModelMain<type_t>::ExchangeImageIndex;
  using CModelMain<type_t>::SaveImage;
  using CModelMain<type_t>::SpeedTest;

  uint32_t Iteration;///<итерация

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelBasicGAN(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelBasicGAN();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Execute(void);///<выполнить
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  virtual void CreateGenerator(void)=0;///<создать сеть генератора
  virtual void CreateDiscriminator(void)=0;///<создать сеть дискриминатора
  void LoadNet(void);///<загрузить сети
  void SaveNet(void);///<сохранить сети
  void LoadTrainingParam(void);///<загрузить параметры обучения
  void SaveTrainingParam(void);///<сохранить параметры обучения
  void CreateFakeImage(CTensor<type_t> &cTensor_Generator_Image);///<создать мнимое изображение с помощью генератора
  void TrainingDiscriminatorFakeAndGenerator(double &disc_cost,double &gen_cost);///<обучение дискриминатора и генератора на фальшивом изображения
  void TrainingDiscriminatorFake(double &disc_cost);///<обучение дискриминатора на фальшивом изображения
  void TrainingDiscriminatorReal(uint32_t mini_batch_index,double &cost);///<обучение дискриминатора на настоящих изображениях
  void TrainingGenerator(double &cost,double &middle_answer);///<обучение генератора
  void SaveRandomImage(void);///<сохранить случайное изображение с генератора
  void SaveKitImage(void);///<сохранить изображение из набора
  void Training(void);///<обучение нейросети
  void TrainingSeparable(void);///<раздельное обучение нейросети
  virtual void TrainingNet(bool mnist);///<запуск обучения нейросети
  void TestTrainingGenerator(void);///<тест обучения генератора
  void TestTrainingGeneratorNet(bool mnist);///<запуск теста обучения генератора
};

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicGAN<type_t>::CModelBasicGAN(void)
{
 BATCH_AMOUNT=0;
 BATCH_SIZE=0;

 IMAGE_WIDTH=0;
 IMAGE_HEIGHT=0;
 IMAGE_DEPTH=0;
 NOISE_LAYER_SIDE_X=0;
 NOISE_LAYER_SIDE_Y=0;
 NOISE_LAYER_SIDE_Z=0;
 NOISE_LAYER_SIZE=NOISE_LAYER_SIDE_X*NOISE_LAYER_SIDE_Y*NOISE_LAYER_SIDE_Z;

 SPEED_DISCRIMINATOR=0;
 SPEED_GENERATOR=0;
 BATCH_SIZE=1;

 ITERATION_OF_SAVE_IMAGE=1;
 ITERATION_OF_SAVE_NET=1;

 Iteration=0;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicGAN<type_t>::~CModelBasicGAN()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************
//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::LoadNet(void)
{
 FILE *file=fopen("disc_neuronet.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("disc_neuronet.net",false));
  LoadNetLayers(iDataStream_Disc_Ptr.get(),DiscriminatorNet,true);
  SYSTEM::PutMessageToConsole("Сеть дискриминатора загружена.");
 }
 file=fopen("gen_neuronet.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("gen_neuronet.net",false));
  LoadNetLayers(iDataStream_Gen_Ptr.get(),GeneratorNet,false);
  SYSTEM::PutMessageToConsole("Сеть генератора загружена.");
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::SaveNet(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("disc_neuronet.net",true));
 SaveNetLayers(iDataStream_Disc_Ptr.get(),DiscriminatorNet,true);

 std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("gen_neuronet.net",true));
 SaveNetLayers(iDataStream_Gen_Ptr.get(),GeneratorNet,false);
}
//----------------------------------------------------------------------------------------------------
//загрузить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::LoadTrainingParam(void)
{
 FILE *file=fopen("disc_training_param.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("disc_training_param.net",false));
  LoadNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),DiscriminatorNet,Iteration,true);
  SYSTEM::PutMessageToConsole("Параметры обучения дискриминатора загружены.");
 }
 file=fopen("gen_training_param.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("gen_training_param.net",false));
  LoadNetLayersTrainingParam(iDataStream_Gen_Ptr.get(),GeneratorNet,Iteration,false);
  SYSTEM::PutMessageToConsole("Параметры обучения генератора загружены.");
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::SaveTrainingParam(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("disc_training_param.net",true));
 SaveNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),DiscriminatorNet,Iteration,true);

 std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("gen_training_param.net",true));
 SaveNetLayersTrainingParam(iDataStream_Gen_Ptr.get(),GeneratorNet,Iteration,false);
}

//----------------------------------------------------------------------------------------------------
//создать мнимое изображение с помощью генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::CreateFakeImage(CTensor<type_t> &cTensor_Generator_Image)
{
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(BATCH_SIZE,1,NOISE_LAYER_SIZE,1);
 if (IsExit()==true) throw("Стоп");
 CRandom<type_t>::SetRandomNormal(cTensor_Generator_Input,-1,1);//входной вектор
 GeneratorNet[0]->SetOutput(cTensor_Generator_Input);//входной вектор
 //выполняем прямой проход по сети
 for(uint32_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
 //получаем ответ сети
 cTensor_Generator_Image=GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor();
}

//----------------------------------------------------------------------------------------------------
//обучение дискриминатора и генератора на фальшивом наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TrainingDiscriminatorFakeAndGenerator(double &disc_cost,double &gen_cost)
{
/*
 CTimeStamp cTimeStamp_Main("Обучение дискриминатора и генератора на фальшивых изображениях:",true);
 char str[STRING_BUFFER_SIZE];
 double fake_output=0;
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(1,NOISE_LAYER_SIZE,1);

 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  CRandom<type_t>::SetRandomNormal(cTensor_Generator_Input,-1,1);
  GeneratorNet[0]->SetOutput(cTensor_Generator_Input);//входной вектор
  //выполняем прямой проход по сети генератора
  {
   CTimeStamp cTimeStamp("Вычисление генератора:");
   for(uint32_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
  }

  {
   CTimeStamp cTimeStamp("Вычисление дискриминатора:");
   for(uint32_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
  }

  //вычисляем ошибку
  {
   CTimeStamp cTimeStamp("Получение ответа:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output);
  }
  {
   CTimeStamp cTimeStamp("Расчёт ошибки дискриминатора:");
   fake_output=cTensor_Discriminator_Output.GetElement(0,0,0);
   //double disc_error=SafeLog(fake_output);//прямая метка
   double disc_error=(fake_output-1);//прямая метка

   gen_cost+=disc_error*disc_error;
   cTensor_Discriminator_Error.SetElement(0,0,0,disc_error);

   if (b==0)
   {
    sprintf(str,"Ответ дискриминатора для генератора [%i]:%f",static_cast<int>(b),cTensor_Discriminator_Output.GetElement(0,0,0));
    SYSTEM::PutMessageToConsole(str);
   }
  }
  {
   CTimeStamp cTimeStamp("Задание ошибки для генератора:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error);
  }
  //выполняем вычисление ошибок дискриминатора без обновления весов
  {
   CTimeStamp cTimeStamp("Обучение дискриминатора (получение ошибок):");
   for(uint32_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward(false);
  }

  //выполняем вычисление весов генератора
  {
   CTimeStamp cTimeStamp("Обучение генератора:");
   for(uint32_t m=0,n=GeneratorNet.size()-1;m<GeneratorNet.size();m++,n--) GeneratorNet[n]->TrainingBackward();
  }


  //обучение дискриминатора (ещё один проход в обратном направлении)

  //вычисляем ошибку последнего слоя
  {
   CTimeStamp cTimeStamp("Расчёт ошибки для дискриминатора:");
   fake_output=cTensor_Discriminator_Output.GetElement(0,0,0);
   //double disc_fake_error=-SafeLog(1.0-fake_output);//прямая метка
   double disc_fake_error=(fake_output-0);//прямая метка

   disc_cost+=disc_fake_error*disc_fake_error;
   cTensor_Discriminator_Error.SetElement(0,0,0,disc_fake_error);
  }
  //задаём ошибку дискриминатору
  {
   CTimeStamp cTimeStamp("Задание ошибки для дискриминатора:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error);
  }
  //выполняем вычисление весов
  {
   CTimeStamp cTimeStamp("Обучение дискриминатора:");
   for(uint32_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward();
  }
 }
 */
}

//----------------------------------------------------------------------------------------------------
//обучение дискриминатора на фальшивом наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TrainingDiscriminatorFake(double &disc_cost)
{
 CTimeStamp cTimeStamp_Main("Обучение дискриминатора на фальшивых изображениях:",true);
 char str[STRING_BUFFER_SIZE];
 double fake_output=0;
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(BATCH_SIZE,1,NOISE_LAYER_SIZE,1);

 CRandom<type_t>::SetRandomNormal(cTensor_Generator_Input,-1,1);
 GeneratorNet[0]->SetOutput(cTensor_Generator_Input);//входной вектор
 //выполняем прямой проход по сети генератора
 {
  CTimeStamp cTimeStamp("Вычисление генератора:");
  for(uint32_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
 }
 {
  CTimeStamp cTimeStamp("Вычисление дискриминатора:");
  for(uint32_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
 }
 {
  CTimeStamp cTimeStamp("Получение ответа:");
  DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output);
 }


 //вычисляем ошибку
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  //вычисляем ошибку последнего слоя
  {
   CTimeStamp cTimeStamp("Расчёт ошибки для дискриминатора:");
   fake_output=cTensor_Discriminator_Output.GetElement(b,0,0,0);
   //double disc_fake_error=-SafeLog(1.0-fake_output);//прямая метка
   //double disc_fake_error=(fake_output-0);//прямая метка
   double disc_fake_error=1+fake_output;
   if (disc_fake_error<0) disc_fake_error=0;

   if (b==0)
   {
    sprintf(str,"Ответ дискриминатора для генератора [%i]:%f (->%f)",static_cast<int>(b),fake_output,disc_fake_error);
    SYSTEM::PutMessageToConsole(str);
   }
   disc_cost+=disc_fake_error*disc_fake_error;
   cTensor_Discriminator_Error.SetElement(b,0,0,0,disc_fake_error);
  }
 }

 //задаём ошибку дискриминатору
 {
  CTimeStamp cTimeStamp("Задание ошибки для дискриминатора:");
  DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error);
 }
 //выполняем вычисление весов
 {
  CTimeStamp cTimeStamp("Обучение дискриминатора:");
  for(uint32_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward();
 }
}


//----------------------------------------------------------------------------------------------------
//обучение дискриминатора на настоящем наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TrainingDiscriminatorReal(uint32_t mini_batch_index,double &cost)
{
 CTimeStamp cTimeStamp_Main("Обучение дискриминатора на истинных изображениях:",true);
 char str[STRING_BUFFER_SIZE];
 double real_output=0;
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  //подаём на вход генератора истинное изображение
  {
   CTimeStamp cTimeStamp("Задание изображения:");
   //дискриминатор подключён к изображению
   uint32_t size=RealImage[RealImageIndex[b+mini_batch_index*BATCH_SIZE]].size();
   type_t *ptr=&RealImage[RealImageIndex[b+mini_batch_index*BATCH_SIZE]][0];
   GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().CopyItemLayerWToDevice(b,ptr,size);
  }
 }
 //вычисляем сеть
 {
  CTimeStamp cTimeStamp("Вычисление дискриминатора:");
  for(uint32_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
 }
 //вычисляем ошибку последнего слоя
 {
  CTimeStamp cTimeStamp("Получение ответа дискриминатора:");
  DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output);
 }
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  CTimeStamp cTimeStamp("Вычисление ошибки:");
  real_output=cTensor_Discriminator_Output.GetElement(b,0,0,0);
  //double disc_real_error=SafeLog(real_output);//прямая метка
//   double disc_real_error=(real_output-1);//прямая метка
  double disc_real_error=1-real_output;//прямая метка
  if (disc_real_error<0) disc_real_error=0;
  disc_real_error=-disc_real_error;

  if (b==0)
  {
   sprintf(str,"Ответ дискриминатора на истину [%i]:%f (->:%f)",static_cast<int>(b),real_output,disc_real_error);
   SYSTEM::PutMessageToConsole(str);
  }

  cost+=disc_real_error*disc_real_error;

  cTensor_Discriminator_Error.SetElement(b,0,0,0,disc_real_error);
 }
 {
  CTimeStamp cTimeStamp("Задание ошибки:");
  //задаём ошибку дискриминатору
  DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error);
 }

 //выполняем вычисление весов
 {
  CTimeStamp cTimeStamp("Обучение дискриминатора:");
  for(uint32_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward();
 }
}





//----------------------------------------------------------------------------------------------------
//обучение генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TrainingGenerator(double &cost,double &middle_answer)
{
 CTimeStamp cTimeStamp_Main("Обучение генератора:",true);

 char str[STRING_BUFFER_SIZE];
 double fake_output=0;
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(BATCH_SIZE,1,NOISE_LAYER_SIZE,1);

 CRandom<type_t>::SetRandomNormal(cTensor_Generator_Input,-1,1);
 GeneratorNet[0]->SetOutput(cTensor_Generator_Input);//входной вектор
 //выполняем прямой проход по сети генератора
 {
  CTimeStamp cTimeStamp("Вычисление генератора:");
  for(uint32_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
 }
 {
  CTimeStamp cTimeStamp("Вычисление дискриминатора:");
  for(uint32_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
 }

 //вычисляем ошибку
 {
  CTimeStamp cTimeStamp("Получение ответа:");
  DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output);
 }

 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  middle_answer+=cTensor_Discriminator_Output.GetElement(b,0,0,0);
  {
   CTimeStamp cTimeStamp("Расчёт ошибки дискриминатора:");
   fake_output=cTensor_Discriminator_Output.GetElement(b,0,0,0);
   //double disc_error=SafeLog(fake_output);//прямая метка
   double disc_error=-fake_output;//прямая метка

   disc_error=-disc_error;

   if (b==0)
   {
    sprintf(str,"Ответ дискриминатора для генератора [%i]:%f (->:%f)",static_cast<int>(b),fake_output,disc_error);
    SYSTEM::PutMessageToConsole(str);
   }

   cost+=disc_error*disc_error;
   cTensor_Discriminator_Error.SetElement(b,0,0,0,disc_error);
  }
 }
 {
  CTimeStamp cTimeStamp("Задание ошибки:");
  DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error);
 }

 //выполняем вычисление весов дискриминатораals
 {
  CTimeStamp cTimeStamp("Обучение дискриминатора (получение ошибок):");
  for(uint32_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward(false);
 }

 //выполняем вычисление весов генератора
 {
  CTimeStamp cTimeStamp("Обучение генератора:");
  for(uint32_t m=0,n=GeneratorNet.size()-1;m<GeneratorNet.size();m++,n--) GeneratorNet[n]->TrainingBackward();
 }

 middle_answer/=static_cast<double>(BATCH_SIZE);
}
//----------------------------------------------------------------------------------------------------
//сохранить случайное изображение с генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::SaveRandomImage(void)
{
 static uint32_t counter=0;
 char str[STRING_BUFFER_SIZE];
 CreateFakeImage(cTensor_Generator_Output);

 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  counter=0;
  sprintf(str,"Test/test%05i-%03i.tga",static_cast<int>(counter),static_cast<int>(n));
  SaveImage(cTensor_Generator_Output,str,n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor_Generator_Output,"Test/test-current.tga",n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  sprintf(str,"Test/test%03i.txt",n);
  cTensor_Generator_Output.PrintToFile(str,"Изображение",true);
 }
 counter++;

}
//----------------------------------------------------------------------------------------------------
//сохранить изображение из набора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::SaveKitImage(void)
{
 char str[STRING_BUFFER_SIZE];
 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/real%03i.tga",static_cast<int>(n));
  type_t *ptr=&RealImage[RealImageIndex[n]][0];
  uint32_t size=RealImage[RealImageIndex[n]].size();
  cTensor_Generator_Output.CopyItemToHost(ptr,size);
  //SaveImage(cTensor_Generator_Output,str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::Training(void)
{
 char str_b[STRING_BUFFER_SIZE];

 const double disc_speed=SPEED_DISCRIMINATOR;
 const double gen_speed=SPEED_GENERATOR;
 uint32_t max_iteration=1000000000;//максимальное количество итераций обучения
 const double clip=0.5;

 uint32_t image_amount=RealImage.size();

 std::string str;

 CCUDATimeSpent cCUDATimeSpent;

 while(Iteration<max_iteration)
 {
  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string(static_cast<long double>(Iteration+1)));

  ExchangeImageIndex(RealImageIndex);

  if (Iteration%ITERATION_OF_SAVE_IMAGE==0)
  {
   SaveRandomImage();
   SYSTEM::PutMessageToConsole("Save image.");
  }

  if (Iteration%ITERATION_OF_SAVE_NET==0)
  {
   SYSTEM::PutMessageToConsole("Save net.");
   SaveNet();
   SaveTrainingParam();
   SaveKitImage();
   SYSTEM::PutMessageToConsole("");
  }

  for(uint32_t batch=0;batch<BATCH_AMOUNT;batch++)
  {
   if (IsExit()==true) throw("Стоп");

   str="Итерация:";
   str+=std::to_string(static_cast<long double>(Iteration+1));
   str+=" минипакет:";
   str+=std::to_string(static_cast<long double>(batch+1));
   str+=" из ";
   str+=std::to_string(static_cast<long double>(BATCH_AMOUNT));
   SYSTEM::PutMessageToConsole(str);

   {
    cCUDATimeSpent.Start();


   //long double begin_time=SYSTEM::GetSecondCounter();

   //обучаем дискриминатор и генератор
   double disc_cost=0;
   double gen_cost=0;

   for(uint32_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingResetDeltaWeight();
   for(uint32_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingResetDeltaWeight();
   TrainingDiscriminatorFakeAndGenerator(disc_cost,gen_cost);

   //корректируем веса дискриминатора
   {
    CTimeStamp cTimeStamp("Обновление весов дискриминатора на фальшивых изображениях:");
    for(uint32_t n=0;n<DiscriminatorNet.size();n++)
    {
     DiscriminatorNet[n]->TrainingUpdateWeight(disc_speed,Iteration+1);
     //DiscriminatorNet[n]->ClipWeight(-clip,clip);
    }
   }
   //корректируем веса генератора
   {
    CTimeStamp cTimeStamp("Обновление весов генератора:");
    for(uint32_t n=0;n<GeneratorNet.size();n++)
    {
     GeneratorNet[n]->TrainingUpdateWeight(gen_speed,Iteration+1);
     //GeneratorNet[n]->ClipWeight(-clip,clip);//не нужно делать для генератора!
    }
   }



   for(uint32_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingResetDeltaWeight();
   TrainingDiscriminatorReal(batch,disc_cost);
   //корректируем веса дискриминатора
   {
    CTimeStamp cTimeStamp("Обновление весов дискриминатора на настоящих изображениях:");
    for(uint32_t n=0;n<DiscriminatorNet.size();n++)
    {
     DiscriminatorNet[n]->TrainingUpdateWeight(disc_speed,Iteration+1);
     //DiscriminatorNet[n]->ClipWeight(-clip,clip);
    }
   }


   str="Ошибка дискриминатора:";
   str+=std::to_string(static_cast<long double>(disc_cost));
   SYSTEM::PutMessageToConsole(str);

   str="Ошибка генератора:";
   str+=std::to_string(static_cast<long double>(gen_cost));
   SYSTEM::PutMessageToConsole(str);
   }

   float gpu_time=cCUDATimeSpent.Stop();
   sprintf(str_b,"На минипакет ушло:%.2f мс.",gpu_time);
   SYSTEM::PutMessageToConsole(str_b);
   SYSTEM::PutMessageToConsole("");

  }
  Iteration++;
 }
}

//----------------------------------------------------------------------------------------------------
//раздельное обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TrainingSeparable(void)
{
 char str_b[STRING_BUFFER_SIZE];

 const double disc_speed=SPEED_DISCRIMINATOR;
 const double gen_speed=SPEED_GENERATOR;
 uint32_t max_iteration=1000000000;//максимальное количество итераций обучения
 const double clip=0.5;

 uint32_t image_amount=RealImage.size();

 std::string str;

 CCUDATimeSpent cCUDATimeSpent;

 while(Iteration<max_iteration)
 {
  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string(static_cast<long double>(Iteration+1)));

  ExchangeImageIndex(RealImageIndex);

  if (Iteration%ITERATION_OF_SAVE_IMAGE==0)
  {
   SaveRandomImage();
   SYSTEM::PutMessageToConsole("Save image.");
  }

  if (Iteration%ITERATION_OF_SAVE_NET==0)
  {
   SYSTEM::PutMessageToConsole("Save net.");
   SaveNet();
   SaveTrainingParam();
   SaveKitImage();
   SYSTEM::PutMessageToConsole("");
  }

  for(uint32_t batch=0;batch<BATCH_AMOUNT;batch++)
  {
   if (IsExit()==true) throw("Стоп");

   str="Итерация:";
   str+=std::to_string(static_cast<long double>(Iteration+1));
   str+=" минипакет:";
   str+=std::to_string(static_cast<long double>(batch+1));
   str+=" из ";
   str+=std::to_string(static_cast<long double>(BATCH_AMOUNT));
   SYSTEM::PutMessageToConsole(str);

   if (batch%10==0)
   {
    SaveRandomImage();
    SYSTEM::PutMessageToConsole("Save image.");
   }

   {
    cCUDATimeSpent.Start();

    //обучаем дискриминатор и генератор
    double disc_cost=0;
    double gen_cost=0;
    double middle_answer=0;

    for(uint32_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingResetDeltaWeight();
    TrainingDiscriminatorFake(disc_cost);
    TrainingDiscriminatorReal(batch,disc_cost);

    //корректируем веса дискриминатора
    {
     CTimeStamp cTimeStamp("Обновление весов дискриминатора:");
     for(uint32_t n=0;n<DiscriminatorNet.size();n++)
     {
      DiscriminatorNet[n]->TrainingUpdateWeight(disc_speed,Iteration+1);
      //DiscriminatorNet[n]->ClipWeight(-clip,clip);
     }
    }


    for(uint32_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingResetDeltaWeight();
    TrainingGenerator(gen_cost,middle_answer);
    //корректируем веса генератора
    {
     CTimeStamp cTimeStamp("Обновление весов генератора:");
     for(uint32_t n=0;n<GeneratorNet.size();n++)
     {
      GeneratorNet[n]->TrainingUpdateWeight(gen_speed,Iteration+1);
      //GeneratorNet[n]->ClipWeight(-clip,clip);//не нужно делать для генератора!
     }
    }


    str="Ошибка дискриминатора:";
    str+=std::to_string((long double)disc_cost);
    SYSTEM::PutMessageToConsole(str);

    str="Ошибка генератора:";
    str+=std::to_string((long double)gen_cost);
    SYSTEM::PutMessageToConsole(str);
   }
   float gpu_time=cCUDATimeSpent.Stop();
   sprintf(str_b,"На минипакет ушло:%.2f мс.",gpu_time);
   SYSTEM::PutMessageToConsole(str_b);
   SYSTEM::PutMessageToConsole("");

  }
  Iteration++;
 }
}


//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TrainingNet(bool mnist)
{
 char str[STRING_BUFFER_SIZE];
 SYSTEM::MakeDirectory("Test");

 cTensor_Discriminator_Output=CTensor<type_t>(BATCH_SIZE,1,1,1);
 cTensor_Discriminator_Error=CTensor<type_t>(BATCH_SIZE,1,1,1);
 cTensor_Generator_Output=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);

 CreateGenerator();
 CreateDiscriminator();

 for(uint32_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->Reset();
 for(uint32_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->Reset();
 LoadNet();

 //включаем обучение
 for(uint32_t n=0;n<GeneratorNet.size();n++)
 {
  //GeneratorNet[n]->TrainingModeAdam(0.5,0.9);
  GeneratorNet[n]->TrainingStart();
 }
 for(uint32_t n=0;n<DiscriminatorNet.size();n++)
 {
  //DiscriminatorNet[n]->TrainingModeAdam(0.5,0.9);
  DiscriminatorNet[n]->TrainingStart();
 }

 //загружаем изображения
 //if (LoadMNISTImage("mnist.bin",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,RealImageIndex)==false)
 if (LoadImage("RealImage",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,RealImageIndex)==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 SYSTEM::PutMessage("Образы изображений загружены.");
 //дополняем набор до кратного размеру пакета
 uint32_t image_amount=RealImage.size();
 BATCH_AMOUNT=image_amount/BATCH_SIZE;
 if (BATCH_AMOUNT==0) BATCH_AMOUNT=1;
 if (image_amount%BATCH_SIZE!=0)
 {
  uint32_t index=0;
  for(uint32_t n=image_amount%BATCH_SIZE;n<BATCH_SIZE;n++,index++)
  {
   RealImageIndex.push_back(RealImageIndex[index%image_amount]);
  }
  image_amount=RealImageIndex.size();
  BATCH_AMOUNT=image_amount/BATCH_SIZE;
 }
 sprintf(str,"Изображений:%i Минипакетов:%i",static_cast<int>(image_amount),static_cast<int>(BATCH_AMOUNT));
 SYSTEM::PutMessageToConsole(str);

 //загружаем параметры обучения
 LoadTrainingParam();
 //запускаем обучение
 //Training();

 TrainingSeparable();
 //отключаем обучение
 for(uint32_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingStop();
 for(uint32_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingStop();

 SaveNet();
}










//----------------------------------------------------------------------------------------------------
//тест обучения нейросети генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TestTrainingGenerator(void)
{
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(BATCH_SIZE,1,NOISE_LAYER_SIZE,1);
 static CTensor<type_t> cTensor_Generator_Error=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);

 //учим первому изображению
 CTensor<type_t> cTensor_Generator_Etalon=CTensor<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 uint32_t size=RealImage[0].size();
 type_t *ptr=&RealImage[0][0];
 cTensor_Generator_Etalon.CopyItemToDevice(ptr,size);

 char str_b[STRING_BUFFER_SIZE];

 const double gen_speed=SPEED_GENERATOR/BATCH_SIZE;
 uint32_t max_iteration=1000000000;//максимальное количество итераций обучения

 uint32_t image_amount=RealImage.size();

 std::string str;

 while(Iteration<max_iteration)
 {
  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string(static_cast<long double>(Iteration+1)));

  if (Iteration%ITERATION_OF_SAVE_NET==0)
  {
   SYSTEM::PutMessageToConsole("Save net.");
   SaveNet();
   SaveTrainingParam();
  }

  if (Iteration%ITERATION_OF_SAVE_IMAGE==0)
  {
   SYSTEM::PutMessageToConsole("Save image.");
   //SaveRandomImage();
   for(uint32_t n=0;n<BATCH_SIZE;n++)
   {
    CRandom<type_t>::SetRandomNormal(cTensor_Generator_Input,-1,1);
    GeneratorNet[0]->SetOutput(n,cTensor_Generator_Input);//входной вектор
   }
   //выполняем прямой проход по сети
   for(uint32_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
   SaveImage(GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor(0),"Test/test-current.tga",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
   SaveImage(cTensor_Generator_Etalon,"Test/etalon.tga",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
   SaveKitImage();
   SYSTEM::PutMessageToConsole("");
  }

  if (IsExit()==true) throw("Стоп");

  long double begin_time=SYSTEM::GetSecondCounter();

  //обучаем генератор
  double gen_cost=0;
  for(uint32_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingResetDeltaWeight();
  for(uint32_t n=0;n<BATCH_SIZE;n++)
  {
   CRandom<type_t>::SetRandomNormal(cTensor_Generator_Input,-1,1);
   GeneratorNet[0]->SetOutput(n,cTensor_Generator_Input);//входной вектор
  }
  //выполняем прямой проход по сети генератора
  {
   CTimeStamp cTimeStamp("Вычисление генератора:");
   for(uint32_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
  }
  //вычисляем ошибку
  {
   CTimeStamp cTimeStamp("Расчёт ошибки генератора:");
   for(uint32_t n=0;n<BATCH_SIZE;n++)
   {
    CTensorMath<type_t>::Sub(cTensor_Generator_Error,GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor(n),cTensor_Generator_Etalon);
    GeneratorNet[GeneratorNet.size()-1]->SetOutputError(n,cTensor_Generator_Error);
    cTensor_Generator_Error.CopyFromDevice();

    type_t *ptr=cTensor_Generator_Error.GetColumnPtr(0,0);
    for(uint32_t z=0;z<cTensor_Generator_Error.GetSizeZ();z++)
    {
     for(uint32_t y=0;y<cTensor_Generator_Error.GetSizeY();y++)
 	 {
      for(uint32_t x=0;x<cTensor_Generator_Error.GetSizeX();x++,ptr++)
	  {
       type_t delta=*ptr;
	   delta=delta*delta;
	   gen_cost+=delta;
	  }
	 }
	}
   }
  }
  //выполняем вычисление весов генератора
  {
   CTimeStamp cTimeStamp("Обучение генератора:");
   for(uint32_t m=0,n=GeneratorNet.size()-1;m<GeneratorNet.size();m++,n--) GeneratorNet[n]->TrainingBackward();
  }

  //корректируем веса генератора
  {
   CTimeStamp cTimeStamp("Обновление весов генератора:");
   for(uint32_t n=0;n<GeneratorNet.size();n++)
   {
    GeneratorNet[n]->TrainingUpdateWeight(gen_speed,Iteration+1);
   }
  }
  str="Ошибка генератора:";
  str+=std::to_string((long double)gen_cost);
  SYSTEM::PutMessageToConsole(str);

  long double end_time=SYSTEM::GetSecondCounter();
  float cpu_time=static_cast<float>((end_time-begin_time)*1000.0);

  sprintf(str_b,"На минипакет ушло: %.2f мс.\r\n",cpu_time);
  SYSTEM::PutMessageToConsole(str_b);
  SYSTEM::PutMessageToConsole("");

  Iteration++;
 }
}




//----------------------------------------------------------------------------------------------------
//запуск теста обучения генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TestTrainingGeneratorNet(bool mnist)
{
 char str[STRING_BUFFER_SIZE];
 SYSTEM::MakeDirectory("Test");

 ITERATION_OF_SAVE_IMAGE=100;
 ITERATION_OF_SAVE_NET=100;
 BATCH_SIZE=1;

 //загружаем изображения
 //if (LoadMNISTImage("mnist.bin",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,RealImageIndex)==false)
 if (LoadImage("RealImage",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,RealImageIndex)==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 SYSTEM::PutMessage("Образы изображений загружены.");
 //дополняем набор до кратного размеру пакета
 uint32_t image_amount=RealImage.size();
 BATCH_AMOUNT=image_amount/BATCH_SIZE;
 if (BATCH_AMOUNT==0) BATCH_AMOUNT=1;
 if (image_amount%BATCH_SIZE!=0)
 {
  uint32_t index=0;
  for(uint32_t n=image_amount%BATCH_SIZE;n<BATCH_SIZE;n++,index++)
  {
   RealImageIndex.push_back(RealImageIndex[index%image_amount]);
  }
  image_amount=RealImageIndex.size();
  BATCH_AMOUNT=image_amount/BATCH_SIZE;
 }
 sprintf(str,"Изображений:%i Минипакетов:%i",image_amount,BATCH_AMOUNT);
 SYSTEM::PutMessageToConsole(str);

 cTensor_Generator_Output=CTensor<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);

 CreateGenerator();

 for(uint32_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->Reset();
 LoadNet();

 //включаем обучение
 for(uint32_t n=0;n<GeneratorNet.size();n++)
 {
  GeneratorNet[n]->TrainingModeAdam();
  GeneratorNet[n]->TrainingStart();
 }

 //загружаем параметры обучения
 LoadTrainingParam();

 TestTrainingGenerator();

 //отключаем обучение
 for(uint32_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingStop();

 SaveNet();
}


//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelBasicGAN<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);
 //if (CTensorTest<type_t>::Test()==false) throw("Класс тензоров провалил тестирование!");
 //TestTrainingGeneratorNet(true);
 TrainingNet(true);
}

#endif
