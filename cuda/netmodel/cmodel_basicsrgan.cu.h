#ifndef C_MODEL_BASIC_SR_GAN_H
#define C_MODEL_BASIC_SR_GAN_H

//****************************************************************************************************
//Класс-основа для сетей SR-GAN
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
//Класс-основа для сетей SR-GAN
//****************************************************************************************************
template<class type_t>
class CModelBasicSR_GAN:public CModelMain<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 protected:
  //-структуры------------------------------------------------------------------------------------------
  //-переменные-----------------------------------------------------------------------------------------
  uint32_t INPUT_IMAGE_WIDTH;///<ширина входных изображений (сниженное разрешение)
  uint32_t INPUT_IMAGE_HEIGHT;///<высота входных изображений (сниженное разрешение)
  uint32_t INPUT_IMAGE_DEPTH;///<глубина входных изображений (сниженное разрешение)

  uint32_t OUTPUT_IMAGE_WIDTH;///<ширина выходных изображений (высокое разрешение)
  uint32_t OUTPUT_IMAGE_HEIGHT;///<высота выходных изображений (высокое разрешение)
  uint32_t OUTPUT_IMAGE_DEPTH;///<глубина выходных изображений (высокое разрешение)

  uint32_t BATCH_AMOUNT;///<количество пакетов
  uint32_t BATCH_SIZE;///<размер пакета

  uint32_t ITERATION_OF_SAVE_IMAGE;///<какую итерацию сохранять изображения
  uint32_t ITERATION_OF_SAVE_NET;///<какую итерацию сохранять сеть

  double SPEED;///<скорость обучения

  std::vector<std::shared_ptr<INetLayer<type_t> > > Net;///<сеть увеличения разрешения

  CTensor<type_t> cTensor_Image;///<тензор выходного изображения
  CTensor<type_t> cTensor_Error;///<тензор ошибки выходного изображения

  std::vector< std::vector<type_t> > RealHiResImage;///<образы истинных изображений в полном разрешении
  std::vector< std::vector<type_t> > RealLoResImage;///<образы истинных изображений в сниженном разрешении
  std::vector<uint32_t> RealImageIndex;///<индексы изображений в обучающем наборе

  std::vector< std::vector<type_t> > TestImage;///<образы тестовых изображений

  using CModelMain<type_t>::STRING_BUFFER_SIZE;
  using CModelMain<type_t>::CUDA_PAUSE_MS;

  using CModelMain<type_t>::SafeLog;
  using CModelMain<type_t>::CrossEntropy;
  using CModelMain<type_t>::IsExit;
  using CModelMain<type_t>::SetExitState;
  using CModelMain<type_t>::LoadMNISTImage;
  using CModelMain<type_t>::LoadImage;
  using CModelMain<type_t>::CreateResamplingImage;
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
  CModelBasicSR_GAN(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelBasicSR_GAN();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Execute(void);///<выполнить
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  virtual void CreateSRGAN(void)=0;///<создать сеть
  void LoadNet(void);///<загрузить сети
  void SaveNet(void);///<сохранить сети
  void LoadTrainingParam(void);///<загрузить параметры обучения
  void SaveTrainingParam(void);///<сохранить параметры обучения
  void CreateHiResImage(CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_Image);///<создать изображение высокого разрешения
  void TrainingSRGAN(uint32_t mini_batch_index,double &cost);///<обучение
  void SaveRandomImage(void);///<сохранить случайное изображение
  void SaveKitImage(void);///<сохранить изображение из набора
  void Training(void);///<обучение нейросети
  virtual void TrainingNet(bool mnist);///<запуск обучения нейросети
};

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicSR_GAN<type_t>::CModelBasicSR_GAN(void)
{
 BATCH_AMOUNT=0;
 BATCH_SIZE=0;

 INPUT_IMAGE_WIDTH=0;
 INPUT_IMAGE_HEIGHT=0;
 INPUT_IMAGE_DEPTH=0;

 OUTPUT_IMAGE_WIDTH=0;
 OUTPUT_IMAGE_HEIGHT=0;
 OUTPUT_IMAGE_DEPTH=0;

 SPEED=0;

 BATCH_SIZE=1;

 ITERATION_OF_SAVE_IMAGE=1;
 ITERATION_OF_SAVE_NET=1;

 Iteration=0;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicSR_GAN<type_t>::~CModelBasicSR_GAN()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************
//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicSR_GAN<type_t>::LoadNet(void)
{
 FILE *file=fopen("srgan_neuronet.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile("srgan_neuronet.net",false));
  LoadNetLayers(iDataStream_Ptr.get(),Net);
  SYSTEM::PutMessageToConsole("Сеть увеличения разрешения загружена.");
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicSR_GAN<type_t>::SaveNet(void)
{
 std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile("srgan_neuronet.net",true));
 SaveNetLayers(iDataStream_Ptr.get(),Net);
}
//----------------------------------------------------------------------------------------------------
//загрузить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicSR_GAN<type_t>::LoadTrainingParam(void)
{
 FILE *file=fopen("srgan_training_param.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile("srgan_training_param.net",false));
  LoadNetLayersTrainingParam(iDataStream_Ptr.get(),Net,Iteration);
  SYSTEM::PutMessageToConsole("Параметры обучения сети увеличения разрешения загружены.");
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicSR_GAN<type_t>::SaveTrainingParam(void)
{
 std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile("srgan_training_param.net",true));
 SaveNetLayersTrainingParam(iDataStream_Ptr.get(),Net,Iteration);
}

//----------------------------------------------------------------------------------------------------
//создать изображение высокого разрешения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicSR_GAN<type_t>::CreateHiResImage(CTensor<type_t> &cTensor_Input,CTensor<type_t> &cTensor_Image)
{
 if (IsExit()==true) throw("Стоп");
 Net[0]->SetOutput(cTensor_Input);//входной вектор
 //выполняем прямой проход по сети
 for(uint32_t layer=0;layer<Net.size();layer++) Net[layer]->Forward();
 //получаем ответ сети
 cTensor_Image=Net[Net.size()-1]->GetOutputTensor();
}

//----------------------------------------------------------------------------------------------------
//обучение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicSR_GAN<type_t>::TrainingSRGAN(uint32_t mini_batch_index,double &cost)
{
 char str[STRING_BUFFER_SIZE];

 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  //задаём выходное изображение
  {
   uint32_t index=RealImageIndex[b+mini_batch_index*BATCH_SIZE];

   CTimeStamp cTimeStamp("Задание изображения:");
   uint32_t size=RealLoResImage[index].size();
   type_t *ptr=&RealLoResImage[index][0];
   Net[0]->GetOutputTensor().CopyItemToDevice(ptr,size);
   ptr=&RealHiResImage[index][0];
   size=RealHiResImage[index].size();
   cTensor_Image.CopyItemToDevice(ptr,size);
  }

  //вычисляем сеть
  {
   CTimeStamp cTimeStamp("Вычисление сети:");
   for(uint32_t layer=0;layer<Net.size();layer++) Net[layer]->Forward();
  }

  {
   CTimeStamp cTimeStamp("Вычисление ошибки:");
   CTensorMath<type_t>::Sub(cTensor_Error,Net[Net.size()-1]->GetOutputTensor(),cTensor_Image);
   cTensor_Error.CopyFromDevice();
   for(uint32_t x=0;x<cTensor_Error.GetSizeX();x++)
   {
    for(uint32_t y=0;y<cTensor_Error.GetSizeY();y++)
    {
     for(uint32_t z=0;z<cTensor_Error.GetSizeZ();z++)
     {
      type_t c=cTensor_Error.GetElement(z,y,x);
      cost+=c*c;
     }
    }
   }
  }

  //задаём ошибку
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   Net[Net.size()-1]->SetOutputError(cTensor_Error);
  }

  //выполняем вычисление весов
  {
   CTimeStamp cTimeStamp("Обучение:");
   for(uint32_t m=0,n=Net.size()-1;m<Net.size();m++,n--) Net[n]->TrainingBackward();
  }

 }
}
//----------------------------------------------------------------------------------------------------
//сохранить случайное изображение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicSR_GAN<type_t>::SaveRandomImage(void)
{
 static uint32_t counter=0;

 CTensor<type_t> cTensor;
 CTensor<type_t> cTensor_Input=CTensor<type_t>(INPUT_IMAGE_DEPTH,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH);

 char str[STRING_BUFFER_SIZE];
 for(uint32_t n=0;n<BATCH_SIZE*0+1;n++)
 {
  type_t *ptr=&RealLoResImage[n][0];
  cTensor_Input.CopyItemToDevice(ptr,RealLoResImage[n].size());
  CreateHiResImage(cTensor_Input,cTensor);
  sprintf(str,"Test/test%05i-%03i.tga",static_cast<int>(counter),static_cast<int>(n));
  //SaveImage(cTensor,str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor,"Test/test-current.tga",OUTPUT_IMAGE_WIDTH,OUTPUT_IMAGE_HEIGHT,OUTPUT_IMAGE_DEPTH);
  //sprintf(str,"Test/test%03i.txt",n);
  //cTensor.PrintToFile(str,"Изображение",true);
 }
 counter++;
}
//----------------------------------------------------------------------------------------------------
//сохранить изображение из набора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicSR_GAN<type_t>::SaveKitImage(void)
{
 CTensor<type_t> cTensor_Input=CTensor<type_t>(INPUT_IMAGE_DEPTH,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH);

 char str[STRING_BUFFER_SIZE];
 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/real%03i.tga",static_cast<int>(n));
  type_t *ptr=&RealLoResImage[RealImageIndex[n]][0];
  uint32_t size=RealLoResImage[RealImageIndex[n]].size();
  cTensor_Input.CopyItemToHost(ptr,size);
  SaveImage(cTensor_Input,str,INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_DEPTH);
 }
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicSR_GAN<type_t>::Training(void)
{
 char str_b[STRING_BUFFER_SIZE];

 const double speed=SPEED;
 uint32_t max_iteration=1000000000;//максимальное количество итераций обучения

 uint32_t image_amount=RealLoResImage.size();

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
	//обучаем
    double cost=0;
    for(uint32_t n=0;n<Net.size();n++) Net[n]->TrainingResetDeltaWeight();

    TrainingSRGAN(batch,cost);
    //корректируем веса
    {
     CTimeStamp cTimeStamp("Обновление весов:");
     for(uint32_t n=0;n<Net.size();n++)
     {
      Net[n]->TrainingUpdateWeight(speed,Iteration+1);
     }
    }


    str="Ошибка:";
    str+=std::to_string(static_cast<long double>(cost));
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
void CModelBasicSR_GAN<type_t>::TrainingNet(bool mnist)
{
 char str[STRING_BUFFER_SIZE];
 SYSTEM::MakeDirectory("Test");

 CreateSRGAN();

 cTensor_Image=CTensor<type_t>(OUTPUT_IMAGE_DEPTH,OUTPUT_IMAGE_HEIGHT,OUTPUT_IMAGE_WIDTH);
 cTensor_Error=CTensor<type_t>(OUTPUT_IMAGE_DEPTH,OUTPUT_IMAGE_HEIGHT,OUTPUT_IMAGE_WIDTH);

 for(uint32_t n=0;n<Net.size();n++) Net[n]->Reset();

 LoadNet();

 //включаем обучение
 for(uint32_t n=0;n<Net.size();n++)
 {
  Net[n]->TrainingModeAdam();
  //Net[n]->TrainingModeGradient();
  Net[n]->TrainingStart();
 }

 //загружаем изображения
 //if (LoadMNISTImage("mnist.bin",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,RealImageIndex)==false)
 if (LoadImage("RealImage",OUTPUT_IMAGE_WIDTH,OUTPUT_IMAGE_HEIGHT,OUTPUT_IMAGE_DEPTH,RealHiResImage,RealImageIndex)==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 SYSTEM::PutMessage("Образы изображений загружены.");
 //дополняем набор до кратного размеру пакета
 uint32_t image_amount=RealHiResImage.size();
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
 //создаём изображения меньшего разрешения
 CreateResamplingImage(OUTPUT_IMAGE_WIDTH,OUTPUT_IMAGE_HEIGHT,OUTPUT_IMAGE_DEPTH,RealHiResImage,INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_DEPTH,RealLoResImage);
 SYSTEM::PutMessageToConsole("Созданы изображения меньшего разрешения");

 //загружаем параметры обучения
 LoadTrainingParam();
 //запускаем обучение
 Training();
 //отключаем обучение
 for(uint32_t n=0;n<Net.size();n++) Net[n]->TrainingStop();

 SaveNet();
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelBasicSR_GAN<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);
 if (CTensorTest<type_t>::Test()==false) throw("Класс тензоров провалил тестирование!");
 //TestTrainingCoderNet(true);
 TrainingNet(true);
}

#endif
