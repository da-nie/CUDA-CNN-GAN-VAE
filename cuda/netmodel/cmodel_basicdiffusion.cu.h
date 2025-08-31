#ifndef C_MODEL_BASIC_DIFFUSION_H
#define C_MODEL_BASIC_DIFFUSION_H

//****************************************************************************************************
//Класс-основа для сетей Diffusion
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
class CModelBasicDiffusion:public CModelMain<type_t>
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
  uint32_t HIDDEN_LAYER_SIDE_X;///<размерность стороны скрытого слоя по X
  uint32_t HIDDEN_LAYER_SIDE_Y;///<размерность стороны скрытого слоя по Y
  uint32_t HIDDEN_LAYER_SIDE_Z;///<размерность стороны скрытого слоя по Z
  uint32_t HIDDEN_LAYER_SIZE;///<размерность скрытого слоя

  uint32_t BATCH_AMOUNT;///<количество пакетов
  uint32_t BATCH_SIZE;///<размер пакета

  uint32_t ITERATION_OF_SAVE_IMAGE;///<какую итерацию сохранять изображения
  uint32_t ITERATION_OF_SAVE_NET;///<какую итерацию сохранять сеть

  double SPEED;///<скорость обучения

  std::vector<std::shared_ptr<INetLayer<type_t> > > DiffusionNet;///<сеть кодера-декодера

  CTensor<type_t> cTensor_Image;
  CTensor<type_t> cTensor_Error;

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
  CModelBasicDiffusion(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelBasicDiffusion();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Execute(void);///<выполнить
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  virtual void CreateDiffusionNet(void)=0;///<создать сеть
  void LoadNet(void);///<загрузить сети
  void SaveNet(void);///<сохранить сети
  void LoadTrainingParam(void);///<загрузить параметры обучения
  void SaveTrainingParam(void);///<сохранить параметры обучения
  void TrainingDiffusionNet(uint32_t mini_batch_index,double &cost);///<обучение диффузионной сети
  void SaveRandomImage(void);///<какую итерацию сохранять изображения
  void SaveKitImage(void);///<сохранить изображение из набора
  void Training(void);///<обучение нейросети
  virtual void TrainingNet(bool mnist);///<запуск обучения нейросети
};

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicDiffusion<type_t>::CModelBasicDiffusion(void)
{
 BATCH_AMOUNT=0;
 BATCH_SIZE=0;

 IMAGE_WIDTH=0;
 IMAGE_HEIGHT=0;
 IMAGE_DEPTH=0;
 HIDDEN_LAYER_SIDE_X=0;
 HIDDEN_LAYER_SIDE_Y=0;
 HIDDEN_LAYER_SIDE_Z=0;
 HIDDEN_LAYER_SIZE=HIDDEN_LAYER_SIDE_X*HIDDEN_LAYER_SIDE_Y*HIDDEN_LAYER_SIDE_Z;

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
CModelBasicDiffusion<type_t>::~CModelBasicDiffusion()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************
//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::LoadNet(void)
{
 FILE *file=fopen("diffusion_neuronet.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_neuronet.net",false));
  LoadNetLayers(iDataStream_Disc_Ptr.get(),DiffusionNet);
  SYSTEM::PutMessageToConsole("Сеть диффузионной модели загружена.");
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveNet(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_neuronet.net",true));
 SaveNetLayers(iDataStream_Disc_Ptr.get(),DiffusionNet);
}
//----------------------------------------------------------------------------------------------------
//загрузить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::LoadTrainingParam(void)
{
 FILE *file=fopen("diffusion_training_param.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_training_param.net",false));
  LoadNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),DiffusionNet,Iteration);
  SYSTEM::PutMessageToConsole("Параметры обучения диффузионной сети загружены.");
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveTrainingParam(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_training_param.net",true));
 SaveNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),DiffusionNet,Iteration);
}

//----------------------------------------------------------------------------------------------------
//обучение диффузионной сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::TrainingDiffusionNet(uint32_t mini_batch_index,double &cost)
{
/*
 char str[STRING_BUFFER_SIZE];

 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  //задаём изображение для кодировщика
  {
   CTimeStamp cTimeStamp("Задание изображения:");
   //кодер подключён к изображению
   uint32_t img=RealImageIndex[b+mini_batch_index*BATCH_SIZE];
   uint32_t size=RealImage[img].size();
   type_t *ptr=&RealImage[img][0];
   CoderNet[0]->GetOutputTensor().CopyItemLayerWToDevice(b,ptr,size);
  }
 }
 //вычисляем сеть кодировщика
 {
  CTimeStamp cTimeStamp("Вычисление кодировщика:");
  for(uint32_t layer=0;layer<CoderNet.size();layer++) CoderNet[layer]->Forward();
 }

 //вычисляем сеть кодировщика
 {
  CTimeStamp cTimeStamp("Вычисление декодировщика:");
  for(uint32_t layer=0;layer<DecoderNet.size();layer++) DecoderNet[layer]->Forward();
 }
 {
  CTimeStamp cTimeStamp("Вычисление ошибки:");
  CTensorMath<type_t>::Sub(cTensor_Error,DecoderNet[DecoderNet.size()-1]->GetOutputTensor(),CoderNet[0]->GetOutputTensor());
 }
 {
  CTimeStamp cTimeStamp("Задание ошибки:");
  DecoderNet[DecoderNet.size()-1]->SetOutputError(cTensor_Error);
 }

 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {

  char str[255];
  sprintf(str,"Test/input-%i.tga",b);
  SaveImage(CoderNet[0]->GetOutputTensor(b),str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  sprintf(str,"Test/output-%i.tga",b);
  SaveImage(DecoderNet[DecoderNet.size()-1]->GetOutputTensor(b),str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  sprintf(str,"Test/error-%i.tga",b);
  SaveImage(cTensor_Error,str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);

  //считаем ошибку
  double error=0;
  for(uint32_t x=0;x<cTensor_Error.GetSizeX();x++)
  {
   for(uint32_t y=0;y<cTensor_Error.GetSizeY();y++)
   {
    for(uint32_t z=0;z<cTensor_Error.GetSizeZ();z++)
    {
     type_t c=cTensor_Error.GetElement(b,z,y,x);
     error+=c*c;
    }
   }
  }
  if (error>cost) cost=error;
 }
 //выполняем вычисление весов декодировщика
 {
  CTimeStamp cTimeStamp("Обучение декодировщика:");
  for(uint32_t m=0,n=DecoderNet.size()-1;m<DecoderNet.size();m++,n--) DecoderNet[n]->TrainingBackward();
 }
 //выполняем вычисление весов кодировщика
 {
  CTimeStamp cTimeStamp("Обучение кодировщика:");
  for(uint32_t m=0,n=CoderNet.size()-1;m<CoderNet.size();m++,n--) CoderNet[n]->TrainingBackward();
 }
*/
}
//----------------------------------------------------------------------------------------------------
//сохранить случайное изображение с сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveRandomImage(void)
{
/*
 CTensor<type_t> cTensor_Input=CTensor<type_t>(BATCH_SIZE,1,HIDDEN_LAYER_SIZE,1);
 if (IsExit()==true) throw("Стоп");
 CRandom<type_t>::SetRandomNormal(cTensor_Input,0,1);
 CoderNet[CoderNet.size()-1]->SetOutput(cTensor_Input);//входной вектор

 //выполняем прямой проход по сети
 for(uint32_t layer=0;layer<CoderNet.size();layer++) CoderNet[layer]->Forward();
 //выполняем прямой проход по сети
 for(uint32_t layer=0;layer<DecoderNet.size();layer++) DecoderNet[layer]->Forward();
 //получаем ответ сети
 cTensor_Image=DecoderNet[DecoderNet.size()-1]->GetOutputTensor();
 char str[STRING_BUFFER_SIZE];
 static uint32_t counter=0;
 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/test%05i-%03i.tga",static_cast<int>(counter),static_cast<int>(n));
  //SaveImage(cTensor,str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor_Image,"Test/test-current.tga",n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
 counter++;
 */
}
//----------------------------------------------------------------------------------------------------
//сохранить изображение из набора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveKitImage(void)
{
 char str[STRING_BUFFER_SIZE];
 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/real%03i.tga",static_cast<int>(n));
  type_t *ptr=&RealImage[RealImageIndex[n]][0];
  uint32_t size=RealImage[RealImageIndex[n]].size();
  //cTensor_Coder_Output.CopyItemToHost(ptr,size);
  //SaveImage(cTensor_Coder_Output,str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::Training(void)
{
 char str_b[STRING_BUFFER_SIZE];

 const double speed=SPEED;
 uint32_t max_iteration=1000000000;//максимальное количество итераций обучения

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

  double max_cost=0;
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
	//обучаем сеть
    double cost=0;

    for(uint32_t n=0;n<DiffusionNet.size();n++) DiffusionNet[n]->TrainingResetDeltaWeight();

    TrainingDiffusionNet(batch,cost);
    //корректируем веса
    {
     CTimeStamp cTimeStamp("Обновление весов:");
     for(uint32_t n=0;n<DiffusionNet.size();n++) DiffusionNet[n]->TrainingUpdateWeight(speed,Iteration+1);
    }


    str="Ошибка:";
    str+=std::to_string(static_cast<long double>(cost));
    SYSTEM::PutMessageToConsole(str);

    if (cost>max_cost) max_cost=cost;
   }

   float gpu_time=cCUDATimeSpent.Stop();
   sprintf(str_b,"На минипакет ушло:%.2f мс.",gpu_time);
   SYSTEM::PutMessageToConsole(str_b);
   sprintf(str_b,"Максимальная ошибка:%.2f",max_cost);
   SYSTEM::PutMessageToConsole(str_b);
   SYSTEM::PutMessageToConsole("");
  }
  FILE *file=fopen("cost.txt","ab");
  fprintf(file,"%f\r\n",max_cost);
  fclose(file);
  Iteration++;
 }
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::TrainingNet(bool mnist)
{
 char str[STRING_BUFFER_SIZE];
 SYSTEM::MakeDirectory("Test");


 cTensor_Image=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 cTensor_Error=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);

 CreateDiffusionNet();

 for(uint32_t n=0;n<DiffusionNet.size();n++) DiffusionNet[n]->Reset();

 LoadNet();

 //включаем обучение
 for(uint32_t n=0;n<DiffusionNet.size();n++)
 {
  DiffusionNet[n]->TrainingModeAdam(0.5,0.9);
  DiffusionNet[n]->TrainingStart();
 }

 //загружаем изображения
 //if (LoadMNISTImage("mnist.bin",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,RealImageIndex)==false)
 if (LoadImage("RealImage",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,RealImageIndex)==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 SYSTEM::PutMessage("Образы изображений загружены.");


 //делаем изображения с шумом

 std::vector<type_t> &image=RealImage[0];//входное изображение
 std::vector<type_t> y(image.size());
 for(uint32_t n=0;n<image.size();n++) y[n]=CRandom<type_t>::GetPDF(image[n],0,1);

 static const uint32_t TIMESTEP=800;

 struct SImage
 {
  std::vector<double> Image;
  SImage(uint32_t amount,double scale)
  {
   Image.resize(amount);
   for(uint32_t n=0;n<amount;n++) Image[n]=scale;
  }
 };

 //число проходов порчи изображения
 std::vector<SImage> bt;
 bt.push_back(SImage(TIMESTEP,0.1));
 bt.push_back(SImage(TIMESTEP,0.3));
 bt.push_back(SImage(TIMESTEP,0.5));
 bt.push_back(SImage(TIMESTEP,0.7));
 bt.push_back(SImage(TIMESTEP,0.9));
 bt.push_back(SImage(TIMESTEP,0.999));

 double x0=1;//порча исходного изображения
 for(uint32_t b=0;b<bt.size();b++)
 {
  double xt=x0;//текущее испорченное изображение
  std::vector<type_t> q(image.size());
  double mean;
  double variance;
  for(uint32_t t=0;t<TIMESTEP;t++)
  {
   //среднее
   mean=sqrt(1-bt[b].Image[t])*xt;
   //дисперсия
   variance=bt[b].Image[t];
   //вычисляем нормальное распределение
   xt=CRandom<type_t>::GetGaussRandValue(mean,sqrt(variance));
  }
  for(uint32_t n=0;n<image.size();n++) q[n]=image[n];//CRandom<type_t>::GetPDF(image[n],mean,sqrt(variance));
  //выводим картинку
  sprintf(str,"Test/real%03i.tga",static_cast<int>(b));
  type_t *ptr=&q[0];
  uint32_t size=image.size();
  cTensor_Image.CopyItemLayerWToDevice(0,ptr,size);
  SaveImage(cTensor_Image,str,0,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
 throw("Стоп");

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
 Training();
 //отключаем обучение
 for(uint32_t n=0;n<DiffusionNet.size();n++) DiffusionNet[n]->TrainingStop();

 SaveNet();
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelBasicDiffusion<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);
 TrainingNet(true);
}

#endif
