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
#include <random>

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
  //параметры диффузии
  struct SDiffusion
  {
   std::vector<type_t> Beta;///<параметры шума для каждого шага
   std::vector<type_t> Alpha;///<1-beta
   std::vector<type_t> AlphaBar;///<произведение alpha от 0 до t

   void Init(uint32_t time_counter)///<инициализация
   {
    Beta.resize(time_counter);
    Alpha.resize(time_counter);
    AlphaBar.resize(time_counter);
    //заполняем коэффициенты зашумления
    float beta_start=0.01f;
    float beta_end=0.3f;
    for(uint32_t i=0;i<time_counter;i++)
    {
     Beta[i]=beta_start+(beta_end-beta_start)*i/(time_counter-1);
     Alpha[i]=1-Beta[i];
     if (i==0) AlphaBar[i]=Alpha[i];
          else AlphaBar[i]=AlphaBar[i-1]*Alpha[i];
    }
   }
  };
  //параметры обучающих изображений
  struct STrainingImage
  {
   uint32_t RealImageIndex;///<индекс истинного изображения
   uint32_t TimeStep;///<шаг времени
  };
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

  uint32_t TIME_COUNTER;///<число шагов по времени для получения изображения

  double SPEED;///<скорость обучения

  std::vector<std::shared_ptr<INetLayer<type_t> > > DiffusionNet;///<сеть кодера-декодера

  CTensor<type_t> cTensor_Image;
  CTensor<type_t> cTensor_Error;

  std::vector< std::vector<type_t> > RealImage;///<образы истинных изображений
  std::vector<uint32_t> TrainingImageIndex;///<индексы изображений в обучающем наборе
  std::vector<STrainingImage> TrainingImage;///<изображения в обучающем наборе

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

  SDiffusion sDiffusion;///<параметры диффузии
  std::vector<type_t> NoisyImage;///<зашумлённое изображение
  std::vector<type_t> Noise;///<накладываемый шум
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

  void InitDiffusion(void);///<инициализация параметров диффузии
  void GetNoisyImageAndNoise(uint32_t time_step,const std::vector<type_t> &input_image,std::vector<type_t> &noisy_image,std::vector<type_t> &noise);///<получить зашумлённое изображение и шум
  void GetNoise(std::vector<type_t> &noise);///<получить  шум
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

 TIME_COUNTER=30;
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
 char str[STRING_BUFFER_SIZE];

 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  //задаём изображение
  {
   CTimeStamp cTimeStamp("Задание изображения:");
   //вход сети подключён к изображению с шумом
   uint32_t img=b+mini_batch_index*BATCH_SIZE;
   uint32_t training_index=TrainingImageIndex[img];
   uint32_t real_index=TrainingImage[training_index].RealImageIndex;
   uint32_t time_step=TrainingImage[training_index].TimeStep;
   GetNoisyImageAndNoise(time_step,RealImage[real_index],NoisyImage,Noise);
   //задаём изображение с шумом
   type_t *ptr=&NoisyImage[0];
   uint32_t size=NoisyImage.size();
   DiffusionNet[0]->GetOutputTensor().CopyItemLayerWToDevice(b,ptr,size);
   //задаём шум как требуемый ответ сети
   ptr=&Noise[0];
   size=Noise.size();
   cTensor_Image.CopyItemLayerWToDevice(b,ptr,size);
   //задаём временную метку
   for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->SetTimeStep(b,time_step);
  }
 }
/*
 //сохранение зашумлённого изображения
 {
 cTensor_Image=DiffusionNet[0]->GetOutputTensor();
 char str[STRING_BUFFER_SIZE];
 static uint32_t counter=0;
 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/test%05i-%03i.tga",static_cast<int>(counter),static_cast<int>(n));
  SaveImage(cTensor_Image,str,n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor_Image,"Test/test-current.tga",n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
 throw("стоп");
 }
*/


 //вычисляем сеть
 {
  CTimeStamp cTimeStamp("Вычисление сети:");
  for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->Forward();
 }
 {
  CTimeStamp cTimeStamp("Вычисление ошибки:");
  CTensorMath<type_t>::Sub(cTensor_Error,DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor(),cTensor_Image);
  //cTensor_Image=cTensor_Error;
  //CTensorMath<type_t>::Pow2(cTensor_Error,cTensor_Error,1);
 }
/*
 //сохранение ошибки
 {
 char str[STRING_BUFFER_SIZE];
 static uint32_t counter=0;
 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/test%05i-%03i.tga",static_cast<int>(counter),static_cast<int>(n));
  SaveImage(cTensor_Error,str,n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor_Error,"Test/test-current.tga",n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
 //throw("стоп");
 }
 */

 double error=0;
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
/*
  char str[255];
  sprintf(str,"Test/input-%i.tga",b);
  SaveImage(CoderNet[0]->GetOutputTensor(b),str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  sprintf(str,"Test/output-%i.tga",b);
  SaveImage(DecoderNet[DecoderNet.size()-1]->GetOutputTensor(b),str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  sprintf(str,"Test/error-%i.tga",b);
  SaveImage(cTensor_Error,str,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);*/

  //считаем ошибку
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
 }
 error/=static_cast<double>(BATCH_SIZE);
 if (error>cost) cost=error;


 {
  CTimeStamp cTimeStamp("Задание ошибки:");
  //CTensorMath<type_t>::TensorItemProduction(cTensor_Error,cTensor_Error,cTensor_Image);
  DiffusionNet[DiffusionNet.size()-1]->SetOutputError(cTensor_Error);
 }

 //выполняем вычисление весов
 {
  CTimeStamp cTimeStamp("Обучение сети:");
  for(uint32_t m=0,n=DiffusionNet.size()-1;m<DiffusionNet.size();m++,n--) DiffusionNet[n]->TrainingBackward();
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить случайное изображение с сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveRandomImage(void)
{
 Noise.resize(RealImage[0].size());
 //задаём сети начальный шум
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  GetNoise(Noise);
  //задаём шум
  type_t* ptr=&Noise[0];
  uint32_t size=Noise.size();
  DiffusionNet[0]->GetOutputTensor().CopyItemLayerWToDevice(b,ptr,size);
 }
 for(int32_t t=TIME_COUNTER-1;t>=0;t--)
 {
  //задаём сети момент времени
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->SetTimeStep(b,t);
  }
  //получаем от сети предыдущий шум
  for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->Forward();
  //расшумляем изображение
  float alpha=sDiffusion.Alpha[t];
  float beta=sDiffusion.Beta[t];
  float sqrt_alpha=std::sqrt(alpha);
  float sqrt_alpha_bar=std::sqrt(sDiffusion.AlphaBar[t]);
  float sqrt_one_minus_alpha_bar=std::sqrt(1.0-sDiffusion.AlphaBar[t]);
  type_t k=(1.0/sqrt_alpha);
  CTensor<type_t> &input_noise=DiffusionNet[0]->GetOutputTensor();//входной шум
  CTensor<type_t> &output_noise=DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor();//предсказанный сетью шум
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   uint32_t index=0;
   GetNoise(Noise);//новый добавляемый шум
   for(uint32_t z=0;z<input_noise.GetSizeZ();z++)
   {
    for(uint32_t y=0;y<input_noise.GetSizeY();y++)
    {
     for(uint32_t x=0;x<input_noise.GetSizeX();x++,index++)
     {
      type_t input=input_noise.GetElement(b,z,y,x);
      type_t output=output_noise.GetElement(b,z,y,x);
      type_t net_noise=output*(1.0-alpha)/sqrt_one_minus_alpha_bar;
      type_t prev_noise=k*(input-net_noise);
      //if (t>0) prev_noise+=beta*Noise[index];
      //задаём новый шум
      input_noise.SetElement(b,z,y,x,prev_noise);
     }
    }
   }
  }
 }
 //сохраняем изображения (они на входе сети)
 cTensor_Image=DiffusionNet[0]->GetOutputTensor();
 char str[STRING_BUFFER_SIZE];
 static uint32_t counter=0;
 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/test%05i-%03i.tga",static_cast<int>(counter),static_cast<int>(n));
  SaveImage(cTensor_Image,str,n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor_Image,"Test/test-current.tga",n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
 //cTensor_Image.Print("Image");
 //throw("Стоп");

 //counter++;
}
//----------------------------------------------------------------------------------------------------
//сохранить изображение из набора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveKitImage(void)
{
 char str[STRING_BUFFER_SIZE];
 for(uint32_t n=0;n<TrainingImage.size();n++)
 {
  sprintf(str,"Test/real%03i.tga",static_cast<int>(n));
  uint32_t t_index=TrainingImageIndex[n];
  uint32_t r_index=TrainingImage[t_index].RealImageIndex;
  uint32_t time_step=TrainingImage[t_index].TimeStep;
  GetNoisyImageAndNoise(time_step,RealImage[r_index],NoisyImage,Noise);
  type_t *ptr=&NoisyImage[0];
  uint32_t size=NoisyImage.size();
  cTensor_Image.CopyItemLayerWToDevice(0,ptr,size);
  SaveImage(cTensor_Image,str,0,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
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

  ExchangeImageIndex(TrainingImageIndex);

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
   //SaveKitImage();
   SYSTEM::PutMessageToConsole("");
  }

  double max_cost=0;
  for(uint32_t batch=0;batch<BATCH_AMOUNT;batch++)
  {
   if (IsExit()==true) throw("Стоп");

   if (batch%1000==0) SaveRandomImage();

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

  if (max_cost<10) break;
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
  DiffusionNet[n]->TrainingModeAdam(0.9,0.99);
  DiffusionNet[n]->TrainingStart();
 }

 //загружаем изображения
 //if (LoadMNISTImage("mnist.bin",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,RealImageIndex)==false)
 if (LoadImage("RealImage",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,TrainingImageIndex)==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 SYSTEM::PutMessage("Образы изображений загружены.");
 //инициализируем параметры диффузии
 InitDiffusion();
 //создаём обучающий набор с учётом временных меток
 TrainingImage.resize(RealImage.size()*TIME_COUNTER);
 TrainingImageIndex.resize(RealImage.size()*TIME_COUNTER);
 uint32_t index=0;
 for(uint32_t n=0;n<RealImage.size();n++)
 {
  for(uint32_t t=0;t<TIME_COUNTER;t++,index++)
  {
   TrainingImage[index].RealImageIndex=n;
   TrainingImage[index].TimeStep=t;
   TrainingImageIndex[index]=index;
  }
 }
 //дополняем набор до кратного размеру пакета
 uint32_t image_amount=TrainingImage.size();
 BATCH_AMOUNT=image_amount/BATCH_SIZE;
 if (BATCH_AMOUNT==0) BATCH_AMOUNT=1;
 if (image_amount%BATCH_SIZE!=0)
 {
  uint32_t index=0;
  for(uint32_t n=image_amount%BATCH_SIZE;n<BATCH_SIZE;n++,index++)
  {
   TrainingImageIndex.push_back(TrainingImageIndex[index%image_amount]);
  }
  image_amount=TrainingImageIndex.size();
  BATCH_AMOUNT=image_amount/BATCH_SIZE;
 }
 sprintf(str,"Исходных изображений:%i Обучающих изображений:%i Минипакетов:%i",static_cast<int>(RealImage.size()),static_cast<int>(image_amount),static_cast<int>(BATCH_AMOUNT));
 SYSTEM::PutMessageToConsole(str);

 //загружаем параметры обучения
 LoadTrainingParam();
 //запускаем обучение
 Training();
 //отключаем обучение
 for(uint32_t n=0;n<DiffusionNet.size();n++) DiffusionNet[n]->TrainingStop();

 SaveNet();
}


//----------------------------------------------------------------------------------------------------
//!инициализация параметров диффузии
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::InitDiffusion(void)
{
 sDiffusion.Init(TIME_COUNTER);
}

//----------------------------------------------------------------------------------------------------
//!получить зашумлённое изображение и шум
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::GetNoisyImageAndNoise(uint32_t time_step,const std::vector<type_t> &input_image,std::vector<type_t> &noisy_image,std::vector<type_t> &noise)
{
 noise.resize(input_image.size());
 noisy_image.resize(input_image.size());

 //генерируем шум
 GetNoise(noise);
 //вычисляем коэффициенты для текущего шага
 type_t sqrt_alpha_bar=std::sqrt(sDiffusion.AlphaBar[time_step]);
 type_t sqrt_one_minus_alpha_bar=std::sqrt(1.0-sDiffusion.AlphaBar[time_step]);
 //применяем шум к изображению
 for(uint32_t i=0;i<input_image.size();i++)
 {
  noisy_image[i]=sqrt_alpha_bar*input_image[i]+sqrt_one_minus_alpha_bar*noise[i];
 }
}

//----------------------------------------------------------------------------------------------------
//!получить  шум
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::GetNoise(std::vector<type_t> &noise)
{
 double max=1;
 double min=-1;
 uint32_t size=noise.size();
 double average=0;//(max+min)/2.0;
 double sigma=1;//(average-min)/3.0;
/*
 for(uint32_t n=0;n<size;n++)
 {
  type_t value=static_cast<type_t>(CRandom<type_t>::GetGaussRandValue(average,sigma));
  //есть вероятность (0.3%) что сгенерированное число выйдет за нужный нам диапазон
  while(value<min || value>max) value=static_cast<type_t>(CRandom<type_t>::GetGaussRandValue(average,sigma));//если это произошло генерируем новое число.
  noise[n]=value;
 }*/

 //генератор случайных чисел
 std::random_device rd;
 std::mt19937 gen(rd());
 std::normal_distribution<type_t> dist(average,sigma);

 //генерируем шум
 for(uint32_t i=0;i<noise.size();i++)
 {
  noise[i]=dist(gen);
  //while(noise[i]<min || noise[i]>max) noise[i]=dist(gen);
 }

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
