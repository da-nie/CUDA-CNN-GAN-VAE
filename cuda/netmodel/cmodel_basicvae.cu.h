#ifndef C_MODEL_BASIC_VAE_H
#define C_MODEL_BASIC_VAE_H

//****************************************************************************************************
//Класс-основа для сетей VAE
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
//Класс-основа для сетей VAE
//****************************************************************************************************
template<class type_t>
class CModelBasicVAE:public CModelMain<type_t>
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

  double SPEED;///<скорость обучения

  std::vector<std::shared_ptr<INetLayer<type_t> > > CoderNet;///<сеть кодера
  std::vector<std::shared_ptr<INetLayer<type_t> > > DecoderNet;///<сеть декодера

  CTensor<type_t> cTensor_Image;
  CTensor<type_t> cTensor_Error;

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
  CModelBasicVAE(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelBasicVAE();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Execute(void);//выполнить
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  virtual void CreateCoder(void)=0;//создать сеть генератора
  virtual void CreateDecoder(void)=0;//создать сеть дискриминатора
  void LoadNet(void);//загрузить сети
  void SaveNet(void);//сохранить сети
  void LoadTrainingParam(void);//загрузить параметры обучения
  void SaveTrainingParam(void);//сохранить параметры обучения
  void CreateFakeImage(CTensor<type_t> &cTensor_Image);//создать мнимое изображение с помощью декодировщика
  void TrainingCoderAndDecoder(size_t mini_batch_index,double &cost);//обучение кодировщика и декодировщика
  void SaveRandomImage(void);//сохранить случайное изображение с генератора
  void SaveKitImage(void);//сохранить изображение из набора
  void Training(void);//обучение нейросети
  virtual void TrainingNet(bool mnist);//запуск обучения нейросети
  void TestTrainingCoder(void);//тест обучения генератора
  void TestTrainingCoderNet(bool mnist);//запуск теста обучения генератора
};

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicVAE<type_t>::CModelBasicVAE(void)
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

 SPEED=0;

 BATCH_SIZE=1;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicVAE<type_t>::~CModelBasicVAE()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************
//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicVAE<type_t>::LoadNet(void)
{
 FILE *file=fopen("coder_neuronet.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("coder_neuronet.net",false));
  LoadNetLayers(iDataStream_Disc_Ptr.get(),CoderNet);
  SYSTEM::PutMessageToConsole("Сеть кодировщика загружена.");
 }
 file=fopen("decoder_neuronet.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("decoder_neuronet.net",false));
  LoadNetLayers(iDataStream_Gen_Ptr.get(),DecoderNet);
  SYSTEM::PutMessageToConsole("Сеть декодировщика загружена.");
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicVAE<type_t>::SaveNet(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("coder_neuronet.net",true));
 SaveNetLayers(iDataStream_Disc_Ptr.get(),CoderNet);

 std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("decoder_neuronet.net",true));
 SaveNetLayers(iDataStream_Gen_Ptr.get(),DecoderNet);
}
//----------------------------------------------------------------------------------------------------
//загрузить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicVAE<type_t>::LoadTrainingParam(void)
{
 FILE *file=fopen("coder_training_param.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("coder_training_param.net",false));
  LoadNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),CoderNet);
  SYSTEM::PutMessageToConsole("Параметры обучения кодировщика загружены.");
 }
 file=fopen("decoder_training_param.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("decoder_training_param.net",false));
  LoadNetLayersTrainingParam(iDataStream_Gen_Ptr.get(),DecoderNet);
  SYSTEM::PutMessageToConsole("Параметры обучения декодировщика загружены.");
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicVAE<type_t>::SaveTrainingParam(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("coder_training_param.net",true));
 SaveNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),CoderNet);

 std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("decoder_training_param.net",true));
 SaveNetLayersTrainingParam(iDataStream_Gen_Ptr.get(),DecoderNet);
}

//----------------------------------------------------------------------------------------------------
//создать мнимое изображение с помощью декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicVAE<type_t>::CreateFakeImage(CTensor<type_t> &cTensor_Image)
{
 CTensor<type_t> cTensor_Input=CTensor<type_t>(1,NOISE_LAYER_SIZE,1);
 if (IsExit()==true) throw("Стоп");
 type_t *ptr=cTensor_Input.GetColumnPtr(0,0);
 for(uint32_t n=0;n<NOISE_LAYER_SIZE;n++,ptr++)
 {
  type_t r=GetRandValue(20.0)-10.0;
  r/=10.0;
  *ptr=r;
 }
 cTensor_Input.SetHostOnChange();
 CoderNet[CoderNet.size()-1]->SetOutput(cTensor_Input);//входной вектор
 //выполняем прямой проход по сети
 for(size_t layer=0;layer<DecoderNet.size();layer++) DecoderNet[layer]->Forward();
 //получаем ответ сети
 cTensor_Image=DecoderNet[DecoderNet.size()-1]->GetOutputTensor();
}

//----------------------------------------------------------------------------------------------------
//обучение кодировщика и декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicVAE<type_t>::TrainingCoderAndDecoder(size_t mini_batch_index,double &cost)
{
 char str[STRING_BUFFER_SIZE];

 for(size_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  //задаём изображение для кодировщика
  {
   CTimeStamp cTimeStamp("Задание изображения:");
   //кодер подключён к изображению
   size_t size=RealImage[RealImageIndex[b+mini_batch_index*BATCH_SIZE]].size();
   type_t *ptr=&RealImage[RealImageIndex[b+mini_batch_index*BATCH_SIZE]][0];
   CoderNet[0]->GetOutputTensor().CopyItemToDevice(ptr,size);
   cTensor_Image.CopyItemToDevice(ptr,size);
  }
  //вычисляем сеть кодировщика
  {
   CTimeStamp cTimeStamp("Вычисление кодировщика:");
   for(size_t layer=0;layer<CoderNet.size();layer++) CoderNet[layer]->Forward();
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
  //вычисляем сеть кодировщика
  {
   CTimeStamp cTimeStamp("Вычисление декодировщика:");
   for(size_t layer=0;layer<DecoderNet.size();layer++) DecoderNet[layer]->Forward();
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
  {
   CTimeStamp cTimeStamp("Вычисление ошибки:");
   CTensorMath<type_t>::Sub(cTensor_Error,DecoderNet[DecoderNet.size()-1]->GetOutputTensor(),cTensor_Image);
   cTensor_Error.CopyFromDevice();
   for(size_t x=0;x<cTensor_Error.GetSizeX();x++)
   {
    for(size_t y=0;y<cTensor_Error.GetSizeY();y++)
    {
     for(size_t z=0;z<cTensor_Error.GetSizeZ();z++)
     {
      type_t c=cTensor_Error.GetElement(z,y,x);
      cost+=c*c;
     }
    }
   }
  }
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   DecoderNet[DecoderNet.size()-1]->SetOutputError(cTensor_Error);
  }
  //выполняем вычисление весов декодировщика
  {
   CTimeStamp cTimeStamp("Обучение декодировщика:");
   for(size_t m=0,n=DecoderNet.size()-1;m<DecoderNet.size();m++,n--) DecoderNet[n]->TrainingBackward();
  }
  //выполняем вычисление весов кодировщика
  {
   CTimeStamp cTimeStamp("Обучение кодировщика:");
   for(size_t m=0,n=CoderNet.size()-1;m<CoderNet.size();m++,n--) CoderNet[n]->TrainingBackward();
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить случайное изображение с декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicVAE<type_t>::SaveRandomImage(void)
{
 static size_t counter=0;

 CTensor<type_t> cTensor;

 char str[STRING_BUFFER_SIZE];
 for(size_t n=0;n<BATCH_SIZE*0+1;n++)
 {
  CreateFakeImage(cTensor);
  sprintf(str,"Test/test%05i-%03i.tga",counter,n);
  //SaveImage(cTensor,str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor,"Test/test-current.tga",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  //sprintf(str,"Test/test%03i.txt",n);
  //cTensor.PrintToFile(str,"Изображение",true);
 }
 counter++;
}
//----------------------------------------------------------------------------------------------------
//сохранить изображение из набора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicVAE<type_t>::SaveKitImage(void)
{
 char str[STRING_BUFFER_SIZE];
 for(size_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/real%03i.tga",n);
  type_t *ptr=&RealImage[RealImageIndex[n]][0];
  size_t size=RealImage[RealImageIndex[n]].size();
  //cTensor_Coder_Output.CopyItemToHost(ptr,size);
  //SaveImage(cTensor_Coder_Output,str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicVAE<type_t>::Training(void)
{
 char str_b[STRING_BUFFER_SIZE];

 const double speed=SPEED;
 size_t max_iteration=1000000000;//максимальное количество итераций обучения
 uint32_t iteration=0;

 size_t image_amount=RealImage.size();

 std::string str;

 CCUDATimeSpent cCUDATimeSpent;

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

  for(uint32_t batch=0;batch<BATCH_AMOUNT;batch++)
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
	//обучаем кодер и декодер
    double cost=0;

    for(size_t n=0;n<CoderNet.size();n++) CoderNet[n]->TrainingResetDeltaWeight();
    for(size_t n=0;n<DecoderNet.size();n++) DecoderNet[n]->TrainingResetDeltaWeight();

    TrainingCoderAndDecoder(batch,cost);
    //корректируем веса
    {
     CTimeStamp cTimeStamp("Обновление весов кодировщика:");
     for(size_t n=0;n<CoderNet.size();n++)
     {
      CoderNet[n]->TrainingUpdateWeight(speed/(static_cast<double>(BATCH_SIZE)),iteration+1);
     }
    }
    //корректируем веса генератора
    {
     CTimeStamp cTimeStamp("Обновление весов декодировщика:");
     for(size_t n=0;n<DecoderNet.size();n++)
     {
      DecoderNet[n]->TrainingUpdateWeight(speed/(static_cast<double>(BATCH_SIZE)),iteration+1);
     }
    }
    SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту

    str="Ошибка:";
    str+=std::to_string((long double)cost);
    SYSTEM::PutMessageToConsole(str);
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
void CModelBasicVAE<type_t>::TrainingNet(bool mnist)
{
 char str[STRING_BUFFER_SIZE];
 SYSTEM::MakeDirectory("Test");


 cTensor_Image=CTensor<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 cTensor_Error=CTensor<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);

 CreateCoder();
 CreateDecoder();

 for(size_t n=0;n<CoderNet.size();n++) CoderNet[n]->Reset();
 for(size_t n=0;n<DecoderNet.size();n++) DecoderNet[n]->Reset();

 LoadNet();

 //включаем обучение
 for(size_t n=0;n<CoderNet.size();n++)
 {
  CoderNet[n]->TrainingModeAdam();
  CoderNet[n]->TrainingStart();
 }
 for(size_t n=0;n<DecoderNet.size();n++)
 {
  DecoderNet[n]->TrainingModeAdam();
  DecoderNet[n]->TrainingStart();
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
 for(size_t n=0;n<CoderNet.size();n++) CoderNet[n]->TrainingStop();
 for(size_t n=0;n<DecoderNet.size();n++) DecoderNet[n]->TrainingStop();

 SaveNet();
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelBasicVAE<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);
 if (CTensorTest<type_t>::Test()==false) throw("Класс тензоров провалил тестирование!");
 //TestTrainingCoderNet(true);
 TrainingNet(true);
}

#endif