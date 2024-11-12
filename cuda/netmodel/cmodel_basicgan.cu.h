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
  size_t IMAGE_WIDTH;///<ширина входных изображений
  size_t IMAGE_HEIGHT;///<высота входных изображений
  size_t IMAGE_DEPTH;///<глубина входных изображений
  size_t NOISE_LAYER_SIDE_X;///<размерность стороны слоя шума по X
  size_t NOISE_LAYER_SIDE_Y;///<размерность стороны слоя шума по Y
  size_t NOISE_LAYER_SIDE_Z;///<размерность стороны слоя шума по Z
  size_t NOISE_LAYER_SIZE;///<размерность слоя шума

  size_t BATCH_AMOUNT;///<количество пакетов
  size_t BATCH_SIZE;///<размер пакета

  double SPEED_DISCRIMINATOR;///<скорость обучения дискриминатора
  double SPEED_GENERATOR;///<скорость обучения генератора

  std::vector<std::shared_ptr<INetLayer<type_t> > > GeneratorNet;///<сеть генератора
  std::vector<std::shared_ptr<INetLayer<type_t> > > DiscriminatorNet;///<сеть дискриминатора

  CTensor<type_t> cTensor_Generator_Output;
  CTensor<type_t> cTensor_Discriminator_Output;
  CTensor<type_t> cTensor_Discriminator_Error;

  std::vector< std::vector<type_t> > RealImage;//образы истинных изображений
  std::vector<size_t> RealImageIndex;//индексы изображений в обучающем наборе

  using CModelMain<type_t>::STRING_BUFFER_SIZE;
  using CModelMain<type_t>::CUDA_PAUSE_MS;

  using CModelMain<type_t>::GetRandValue;
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

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelBasicGAN(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelBasicGAN();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Execute(void);//выполнить
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  virtual void CreateGenerator(void)=0;//создать сеть генератора
  virtual void CreateDiscriminator(void)=0;//создать сеть дискриминатора
  void LoadNet(void);//загрузить сети
  void SaveNet(void);//сохранить сети
  void LoadTrainingParam(void);//загрузить параметры обучения
  void SaveTrainingParam(void);//сохранить параметры обучения
  void CreateFakeImage(CTensor<type_t> &cTensor_Generator_Image);//создать мнимое изображение с помощью генератора
  void TrainingDiscriminatorFakeAndGenerator(double &disc_cost,double &gen_cost);//обучение дискриминатора и генератора на фальшивом изображения
  void TrainingDiscriminatorReal(size_t mini_batch_index,double &cost);//обучение дискриминатора на настоящих изображениях
  void TrainingGenerator(double &cost,double &max_disc_answer);//обучение генератора
  void SaveRandomImage(void);//сохранить случайное изображение с генератора
  void SaveKitImage(void);//сохранить изображение из набора
  void Training(void);//обучение нейросети
  virtual void TrainingNet(bool mnist);//запуск обучения нейросети
  void TestTrainingGenerator(void);//тест обучения генератора
  void TestTrainingGeneratorNet(bool mnist);//запуск теста обучения генератора
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
  LoadNetLayers(iDataStream_Disc_Ptr.get(),DiscriminatorNet);
  SYSTEM::PutMessageToConsole("Сеть дискриминатора загружена.");
 }
 file=fopen("gen_neuronet.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("gen_neuronet.net",false));
  LoadNetLayers(iDataStream_Gen_Ptr.get(),GeneratorNet);
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
 SaveNetLayers(iDataStream_Disc_Ptr.get(),DiscriminatorNet);

 std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("gen_neuronet.net",true));
 SaveNetLayers(iDataStream_Gen_Ptr.get(),GeneratorNet);
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
  LoadNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),DiscriminatorNet);
  SYSTEM::PutMessageToConsole("Параметры обучения дискриминатора загружены.");
 }
 file=fopen("gen_training_param.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("gen_training_param.net",false));
  LoadNetLayersTrainingParam(iDataStream_Gen_Ptr.get(),GeneratorNet);
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
 SaveNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),DiscriminatorNet);

 std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("gen_training_param.net",true));
 SaveNetLayersTrainingParam(iDataStream_Gen_Ptr.get(),GeneratorNet);
}

//----------------------------------------------------------------------------------------------------
//создать мнимое изображение с помощью генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::CreateFakeImage(CTensor<type_t> &cTensor_Generator_Image)
{
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(1,NOISE_LAYER_SIZE,1);
 if (IsExit()==true) throw("Стоп");
 type_t *ptr=cTensor_Generator_Input.GetColumnPtr(0,0);
 for(uint32_t n=0;n<NOISE_LAYER_SIZE;n++,ptr++)
 {
  type_t r=GetRandValue(20.0)-10.0;
  r/=10.0;
  *ptr=r;
 }
 GeneratorNet[0]->SetOutput(cTensor_Generator_Input);//входной вектор
 //выполняем прямой проход по сети
 for(size_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
 //получаем ответ сети
 cTensor_Generator_Image=GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor();
}

//----------------------------------------------------------------------------------------------------
//обучение дискриминатора и генератора на фальшивом наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TrainingDiscriminatorFakeAndGenerator(double &disc_cost,double &gen_cost)
{
 char str[STRING_BUFFER_SIZE];
 double fake_output=0;
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(1,NOISE_LAYER_SIZE,1);

 for(size_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");

  for(uint32_t n=0;n<NOISE_LAYER_SIZE;n++)
  {
   type_t r=GetRandValue(20.0)-10.0;
   r/=10.0;
   cTensor_Generator_Input.SetElement(0,n,0,r);
  }
  GeneratorNet[0]->SetOutput(cTensor_Generator_Input);//входной вектор
  //выполняем прямой проход по сети генератора
  {
   CTimeStamp cTimeStamp("Вычисление генератора:");
   for(size_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
  {
   CTimeStamp cTimeStamp("Вычисление дискриминатора:");
   for(size_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
  //вычисляем ошибку
  {
   CTimeStamp cTimeStamp("Получение ответа:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output);

   if (b==0)
   {
    sprintf(str,"Ответ дискриминатора для генератора [%i]:%f",b,cTensor_Discriminator_Output.GetElement(0,0,0));
    SYSTEM::PutMessageToConsole(str);
   }
  }
  {
   CTimeStamp cTimeStamp("Расчёт ошибки дискриминатора:");
   fake_output=cTensor_Discriminator_Output.GetElement(0,0,0);
   double disc_error=SafeLog(fake_output);//прямая метка
   //double disc_error=-SafeLog(1.0-fake_output);//инверсная метка
   //double disc_error=(fake_output-1);//прямая метка

   gen_cost+=disc_error*disc_error;
   cTensor_Discriminator_Error.SetElement(0,0,0,disc_error);
  }
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error);
  }
  //выполняем вычисление ошибок дискриминатора
  {
   CTimeStamp cTimeStamp("Обучение дискриминатора (получение ошибок):");
   for(size_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward(false);
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
  //выполняем вычисление весов генератора
  {
   //if (fake_output<0.5)
   {
    CTimeStamp cTimeStamp("Обучение генератора:");
    for(size_t m=0,n=GeneratorNet.size()-1;m<GeneratorNet.size();m++,n--) GeneratorNet[n]->TrainingBackward();
   }
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту

  //обучение дискриминатора

  //вычисляем ошибку последнего слоя
  {
   CTimeStamp cTimeStamp("Расчёт ошибки:");
   fake_output=cTensor_Discriminator_Output.GetElement(0,0,0);
   double disc_fake_error=-SafeLog(1.0-fake_output);//прямая метка
   //double disc_fake_error=(fake_output-0);//прямая метка

   if (GetRandValue(100)>95)//инвертируем метки, чтобы избежать переобучения генератора
   {
    //disc_fake_error=SafeLog(fake_output);//инверсная метка
   }

   disc_cost+=disc_fake_error*disc_fake_error;
   cTensor_Discriminator_Error.SetElement(0,0,0,disc_fake_error);
  }
  //задаём ошибку дискриминатору
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error);
  }
  //выполняем вычисление весов
  {
   //if (fake_output>0.5)
   {
    CTimeStamp cTimeStamp("Обучение дискриминатора:");
    for(size_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward();
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//обучение дискриминатора на настоящем наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TrainingDiscriminatorReal(size_t mini_batch_index,double &cost)
{
 char str[STRING_BUFFER_SIZE];
 double real_output=0;
 for(size_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");

  //подаём на вход генератора истинное изображение
  {
   CTimeStamp cTimeStamp("Задание изображения:");
   //дискриминатор подключён к изображению
   size_t size=RealImage[RealImageIndex[b+mini_batch_index*BATCH_SIZE]].size();
   type_t *ptr=&RealImage[RealImageIndex[b+mini_batch_index*BATCH_SIZE]][0];
   GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().CopyItemToDevice(ptr,size);
  }
  //вычисляем сеть
  {
   CTimeStamp cTimeStamp("Вычисление дискриминатора:");
   for(size_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
  //вычисляем ошибку последнего слоя
  {
   CTimeStamp cTimeStamp("Получение ответа дискриминатора:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output);
   if (b==0)
   {
    sprintf(str,"Ответ дискриминатора на истину [%i]:%f",b,cTensor_Discriminator_Output.GetElement(0,0,0));
    SYSTEM::PutMessageToConsole(str);
   }
  }
  {
   CTimeStamp cTimeStamp("Вычисление ошибки:");
   real_output=cTensor_Discriminator_Output.GetElement(0,0,0);
   double disc_real_error=SafeLog(real_output);//прямая метка
   //double disc_real_error=(real_output-1);//прямая метка

   if (GetRandValue(100)>95)//инвертируем метки, чтобы избежать переобучения генератора
   {
    //disc_real_error=-SafeLog(1.0-real_output);//инверсная метка
   }
   cost+=disc_real_error*disc_real_error;
   cTensor_Discriminator_Error.SetElement(0,0,0,disc_real_error);
  }
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   //задаём ошибку дискриминатору
   DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error);
  }
  //выполняем вычисление весов
  {
   //if (real_output<0.5)
   {
    CTimeStamp cTimeStamp("Обучение дискриминатора:");
    for(size_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward();
   }
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
 }
}





//----------------------------------------------------------------------------------------------------
//обучение генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TrainingGenerator(double &cost,double &max_disc_answer)
{
 char str[STRING_BUFFER_SIZE];
 double fake_output=0;
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(1,NOISE_LAYER_SIZE,1);

 max_disc_answer=0;

 for(size_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");

  for(uint32_t n=0;n<NOISE_LAYER_SIZE;n++)
  {
   type_t r=GetRandValue(20.0)-10.0;
   r/=10.0;
   cTensor_Generator_Input.SetElement(0,n,0,r);
  }
  GeneratorNet[0]->SetOutput(cTensor_Generator_Input);//входной вектор
  //выполняем прямой проход по сети генератора
  {
   CTimeStamp cTimeStamp("Вычисление генератора:");
   for(size_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
  {
   CTimeStamp cTimeStamp("Вычисление дискриминатора:");
   for(size_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
  //вычисляем ошибку
  {
   CTimeStamp cTimeStamp("Получение ответа:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output);

   if (max_disc_answer<cTensor_Discriminator_Output.GetElement(0,0,0)) max_disc_answer=cTensor_Discriminator_Output.GetElement(0,0,0);

   if (b==0)
   {
    sprintf(str,"Ответ дискриминатора для генератора [%i]:%f",b,cTensor_Discriminator_Output.GetElement(0,0,0));
    SYSTEM::PutMessageToConsole(str);
   }
  }
  {
   CTimeStamp cTimeStamp("Расчёт ошибки дискриминатора:");
   fake_output=cTensor_Discriminator_Output.GetElement(0,0,0);
   double disc_error=SafeLog(fake_output);//прямая метка
   //double disc_error=-SafeLog(1.0-fake_output);//инверсная метка
   //double disc_error=(fake_output-1);//прямая метка

   cost+=disc_error*disc_error;
   cTensor_Discriminator_Error.SetElement(0,0,0,disc_error);
  }
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error);
  }
  //выполняем вычисление весов дискриминатора
  {
   CTimeStamp cTimeStamp("Обучение дискриминатора (получение ошибок):");
   for(size_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward();
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
  //выполняем вычисление весов генератора
  {
   //if (fake_output<0.5)
   {
    CTimeStamp cTimeStamp("Обучение генератора:");
    for(size_t m=0,n=GeneratorNet.size()-1;m<GeneratorNet.size();m++,n--) GeneratorNet[n]->TrainingBackward();
   }
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
 }

}
//----------------------------------------------------------------------------------------------------
//сохранить случайное изображение с генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::SaveRandomImage(void)
{
 static size_t counter=0;

 char str[STRING_BUFFER_SIZE];
 for(size_t n=0;n<BATCH_SIZE*0+1;n++)
 {
  CreateFakeImage(cTensor_Generator_Output);
  sprintf(str,"Test/test%05i-%03i.tga",counter,n);
  //SaveImage(cTensor_Generator_Output,str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor_Generator_Output,"Test/test-current.tga",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  //sprintf(str,"Test/test%03i.txt",n);
  //cTensor_Generator_Output.PrintToFile(str,"Изображение",true);
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
 for(size_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/real%03i.tga",n);
  type_t *ptr=&RealImage[RealImageIndex[n]][0];
  size_t size=RealImage[RealImageIndex[n]].size();
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
 size_t max_iteration=1000000000;//максимальное количество итераций обучения
 uint32_t iteration=0;
 const double clip=0.5;

 size_t image_amount=RealImage.size();

 std::string str;

 CCUDATimeSpent cCUDATimeSpent;

 size_t get_training=0;

 while(iteration<max_iteration)
 {
  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string((long double)iteration+1));

  ExchangeImageIndex(RealImageIndex);

  if (iteration%1==0)
  {
   SaveRandomImage();
   SYSTEM::PutMessageToConsole("Save image.");
  }

  if (iteration%1==0)
  {
  SYSTEM::PutMessageToConsole("Save net.");
  SaveNet();
  SaveTrainingParam();
  SaveKitImage();
  SYSTEM::PutMessageToConsole("");
  }

  for(uint32_t batch=0;batch<BATCH_AMOUNT;batch++,get_training++)
  {
   if (IsExit()==true) throw("Стоп");

   str="Итерация:";
   str+=std::to_string((long double)iteration+1);
   str+=" минипакет:";
   str+=std::to_string((long double)batch+1);
   str+=" из ";
   str+=std::to_string((long double)BATCH_AMOUNT);
   SYSTEM::PutMessageToConsole(str);

   {
    cCUDATimeSpent.Start();


   //long double begin_time=SYSTEM::GetSecondCounter();

   //обучаем дискриминатор и генератор
   double disc_cost=0;
   double gen_cost=0;

   for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingResetDeltaWeight();
   for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingResetDeltaWeight();
   TrainingDiscriminatorFakeAndGenerator(disc_cost,gen_cost);

   //корректируем веса дискриминатора
   {
    CTimeStamp cTimeStamp("Обновление весов дискриминатора на фальшивых изображениях:");
    for(size_t n=0;n<DiscriminatorNet.size();n++)
    {
     DiscriminatorNet[n]->TrainingUpdateWeight(disc_speed/(static_cast<double>(BATCH_SIZE)),iteration+1);
    // DiscriminatorNet[n]->ClipWeight(-clip,clip);
    }
   }
   //корректируем веса генератора
   {
    CTimeStamp cTimeStamp("Обновление весов генератора:");
    for(size_t n=0;n<GeneratorNet.size();n++)
    {
     GeneratorNet[n]->TrainingUpdateWeight(gen_speed/(static_cast<double>(BATCH_SIZE)),iteration+1);
     //GeneratorNet[n]->ClipWeight(-clip,clip);//не нужно делать для генератора!
    }
   }



   for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingResetDeltaWeight();
   TrainingDiscriminatorReal(batch,disc_cost);
   //корректируем веса дискриминатора
   {
    CTimeStamp cTimeStamp("Обновление весов дискриминатора на настоящих изображениях:");
    for(size_t n=0;n<DiscriminatorNet.size();n++)
    {
     DiscriminatorNet[n]->TrainingUpdateWeight(disc_speed/(static_cast<double>(BATCH_SIZE)),iteration+1);
     //DiscriminatorNet[n]->ClipWeight(-clip,clip);
    }
   }
   SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту

   str="Ошибка дискриминатора:";
   str+=std::to_string((long double)disc_cost);
   SYSTEM::PutMessageToConsole(str);

   str="Ошибка генератора:";
   str+=std::to_string((long double)gen_cost);
   SYSTEM::PutMessageToConsole(str);

   get_training%=5;

//   if (get_training==0)
/*
   {
    //обучаем генератор
    double gen_cost=0;
    for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingResetDeltaWeight();
    double max_disc_answer=0;
    TrainingGenerator(gen_cost,max_disc_answer);

    //корректируем веса генератора
    {
     CTimeStamp cTimeStamp("Обновление весов генератора:");
     for(size_t n=0;n<GeneratorNet.size();n++)
     {
      GeneratorNet[n]->TrainingUpdateWeight(gen_speed/(static_cast<double>(BATCH_SIZE)),iteration+1);
      //GeneratorNet[n]->ClipWeight(-clip,clip);//не нужно делать для генератора!
     }
    }
    SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту

    str="Ошибка генератора:";
    str+=std::to_string((long double)gen_cost);
    str+=" Лучший ответ дискриминатора:";
    str+=std::to_string((long double)max_disc_answer);
    SYSTEM::PutMessageToConsole(str);
   }*/

   //long double end_time=SYSTEM::GetSecondCounter();
   //float cpu_time=static_cast<float>((end_time-begin_time)*1000.0);

   //sprintf(str_b,"На минипакет ушло: %.2f мс.\r\n",cpu_time);
   //SYSTEM::PutMessageToConsole(str_b);
   }

   float gpu_time=cCUDATimeSpent.Stop();
   sprintf(str_b,"На минипакет ушло:%.2f мс.",gpu_time);
   SYSTEM::PutMessageToConsole(str_b);
   SYSTEM::PutMessageToConsole("");

  }
  iteration++;
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

 cTensor_Discriminator_Output=CTensor<type_t>(1,1,1);
 cTensor_Discriminator_Error=CTensor<type_t>(1,1,1);
 cTensor_Generator_Output=CTensor<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);

 CreateGenerator();
 CreateDiscriminator();

 for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->Reset();
 for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->Reset();
 LoadNet();

 //включаем обучение
 for(size_t n=0;n<GeneratorNet.size();n++)
 {
  GeneratorNet[n]->TrainingModeAdam();
  GeneratorNet[n]->TrainingStart();
 }
 for(size_t n=0;n<DiscriminatorNet.size();n++)
 {
  DiscriminatorNet[n]->TrainingModeAdam();
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
 size_t image_amount=RealImage.size();
 BATCH_AMOUNT=image_amount/BATCH_SIZE;
 if (BATCH_AMOUNT==0) BATCH_AMOUNT=1;
 if (image_amount%BATCH_SIZE!=0)
 {
  size_t index=0;
  for(size_t n=image_amount%BATCH_SIZE;n<BATCH_SIZE;n++,index++)
  {
   RealImageIndex.push_back(RealImageIndex[index%image_amount]);
  }
  image_amount=RealImageIndex.size();
  BATCH_AMOUNT=image_amount/BATCH_SIZE;
 }
 sprintf(str,"Изображений:%i Минипакетов:%i",image_amount,BATCH_AMOUNT);
 SYSTEM::PutMessageToConsole(str);

 //загружаем параметры обучения
 LoadTrainingParam();
 //запускаем обучение
 Training();
 //отключаем обучение
 for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingStop();
 for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingStop();

 SaveNet();
}










//----------------------------------------------------------------------------------------------------
//тест обучения нейросети генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicGAN<type_t>::TestTrainingGenerator(void)
{
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(1,NOISE_LAYER_SIZE,1);
 static CTensor<type_t> cTensor_Generator_Error=CTensor<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);

 char str_b[STRING_BUFFER_SIZE];

 const double gen_speed=SPEED_GENERATOR;
 size_t max_iteration=1000000000;//максимальное количество итераций обучения
 uint32_t iteration=0;

 size_t image_amount=RealImage.size();

 std::string str;

  while(iteration<max_iteration)
 {
  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string((long double)iteration+1));

  if (iteration%100==0)
  {
   SYSTEM::PutMessageToConsole("Save net.");
   SaveNet();
   SYSTEM::PutMessageToConsole("Save image.");
   SaveRandomImage();
   SaveKitImage();
   SYSTEM::PutMessageToConsole("");
  }

  if (IsExit()==true) throw("Стоп");

  long double begin_time=SYSTEM::GetSecondCounter();

  //обучаем генератор
  double gen_cost=0;
  for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingResetDeltaWeight();

  for(uint32_t n=0;n<NOISE_LAYER_SIZE;n++)
  {
   type_t r=GetRandValue(20.0)-10.0;
   cTensor_Generator_Input.SetElement(0,n,0,r);
  }

  GeneratorNet[0]->SetOutput(cTensor_Generator_Input);//входной вектор
  //выполняем прямой проход по сети генератора
  {
   CTimeStamp cTimeStamp("Вычисление генератора:");
   for(size_t layer=0;layer<GeneratorNet.size();layer++) GeneratorNet[layer]->Forward();
  }
  //вычисляем ошибку
  {
   CTimeStamp cTimeStamp("Расчёт ошибки генератора:");
   CTensorMath<type_t>::Sub(cTensor_Generator_Error,GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor(),RealImage[0]);//учим первому изображению
   GeneratorNet[GeneratorNet.size()-1]->SetOutputError(cTensor_Generator_Error);
  }
  //выполняем вычисление весов генератора
  {
   CTimeStamp cTimeStamp("Обучение генератора:");
   for(size_t m=0,n=GeneratorNet.size()-1;m<GeneratorNet.size();m++,n--) GeneratorNet[n]->TrainingBackward();
  }

  //корректируем веса генератора
  {
   CTimeStamp cTimeStamp("Обновление весов генератора:");
   for(size_t n=0;n<GeneratorNet.size();n++)
   {
    GeneratorNet[n]->TrainingUpdateWeight(gen_speed);
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

  iteration++;
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
 if (LoadImage("RealImage",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_WIDTH,RealImage,RealImageIndex)==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 SYSTEM::PutMessage("Образы изображений загружены.");
 //дополняем набор до кратного размеру пакета
 size_t image_amount=RealImage.size();
 BATCH_AMOUNT=image_amount/BATCH_SIZE;
 if (BATCH_AMOUNT==0) BATCH_AMOUNT=1;
 if (image_amount%BATCH_SIZE!=0)
 {
  size_t index=0;
  for(size_t n=image_amount%BATCH_SIZE;n<BATCH_SIZE;n++,index++)
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

 for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->Reset();
 LoadNet();

 //включаем обучение
 for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingStart();

 TestTrainingGenerator();

 //отключаем обучение
 for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingStop();

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
 if (CTensorTest<type_t>::Test()==false) throw("Класс тензоров провалил тестирование!");
 //TestTrainingGeneratorNet(true);
 TrainingNet(true);
}

#endif