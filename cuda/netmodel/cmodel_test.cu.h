#ifndef C_MODEL_TEST_H
#define C_MODEL_TEST_H

//****************************************************************************************************
//Класс тестирования
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
//Класс тестирования
//****************************************************************************************************
template<class type_t>
class CModelTest:public CModelMain<type_t>
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

  uint32_t BATCH_SIZE;///<размер пакета

  double SPEED;///<скорость обучения

  std::vector<std::shared_ptr<INetLayer<type_t> > > CoderNet;///<сеть кодера
  std::vector<std::shared_ptr<INetLayer<type_t> > > DecoderNet;///<сеть декодера

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
  CModelTest(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelTest();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Execute(void);///<выполнить
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  void CreateCoder(void);///<создать сеть генератора
  void CreateDecoder(void);///<создать сеть дискриминатора
  void LoadNet(void);///<загрузить сети
  void SaveNet(void);///<сохранить сети
  void Test(double &cost1,double &cost2);///<запуск теста
};

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelTest<type_t>::CModelTest(void)
{
 IMAGE_HEIGHT=128;
 IMAGE_WIDTH=128;
 IMAGE_DEPTH=3;
 NOISE_LAYER_SIDE_X=32;
 NOISE_LAYER_SIDE_Y=32;
 NOISE_LAYER_SIDE_Z=1;
 NOISE_LAYER_SIZE=NOISE_LAYER_SIDE_X*NOISE_LAYER_SIDE_Y*NOISE_LAYER_SIDE_Z;

 SPEED=0.001;

 BATCH_SIZE=1;

 Iteration=0;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelTest<type_t>::~CModelTest()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//создать сеть кодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelTest<type_t>::CreateCoder(void)
{
 CoderNet.clear();

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);


 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor final",false);
}
//----------------------------------------------------------------------------------------------------
//создать сеть декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelTest<type_t>::CreateDecoder(void)
{
 DecoderNet.clear();

 uint32_t convert=CoderNet.size()-1;
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(512*4*4,CoderNet[convert].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 convert=DecoderNet.size()-1;
 DecoderNet[convert]->GetOutputTensor().ReinterpretSize(BATCH_SIZE,512,4,4);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));

 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor final",false);

 DecoderNet[convert]->GetOutputTensor().RestoreSize();
}

//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelTest<type_t>::LoadNet(void)
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
void CModelTest<type_t>::SaveNet(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("coder_neuronet.net",true));
 SaveNetLayers(iDataStream_Disc_Ptr.get(),CoderNet);

 std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("decoder_neuronet.net",true));
 SaveNetLayers(iDataStream_Gen_Ptr.get(),DecoderNet);
}


//----------------------------------------------------------------------------------------------------
//запуск теста
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelTest<type_t>::Test(double &cost1,double &cost2)
{
 char str_b[STRING_BUFFER_SIZE];
 std::string str;

 double speed=SPEED;

 cTensor_Image=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 cTensor_Error=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);

 //включаем обучение
 for(uint32_t n=0;n<CoderNet.size();n++)
 {
  CoderNet[n]->TrainingModeAdam();
  CoderNet[n]->TrainingStart();
 }
 for(uint32_t n=0;n<DecoderNet.size();n++)
 {
  DecoderNet[n]->TrainingModeAdam();
  DecoderNet[n]->TrainingStart();
 }

 //создаём входной вектор
 for(uint32_t w=0;w<cTensor_Image.GetSizeW();w++)
 {
  for(uint32_t z=0;z<cTensor_Image.GetSizeZ();z++)
  {
   for(uint32_t y=0;y<cTensor_Image.GetSizeY();y++)
   {
    for(uint32_t x=0;x<cTensor_Image.GetSizeX();x++)
    {
     type_t value=(x+y*y+z*10)%100;
     value/=100.0;
     cTensor_Image.SetElement(w,z,y,x,value);
    }
   }
  }
 }

 //делаем один проход обучения

 double cost=0;

 for(uint32_t n=0;n<CoderNet.size();n++) CoderNet[n]->TrainingResetDeltaWeight();
 for(uint32_t n=0;n<DecoderNet.size();n++) DecoderNet[n]->TrainingResetDeltaWeight();

 CoderNet[0]->GetOutputTensor()=cTensor_Image;

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
  CTensorMath<type_t>::Sub(cTensor_Error,DecoderNet[DecoderNet.size()-1]->GetOutputTensor(),CoderNet[0]->GetOutputTensor(),2,2);
 }

 {
  CTimeStamp cTimeStamp("Задание ошибки:");
  DecoderNet[DecoderNet.size()-1]->SetOutputError(cTensor_Error);
 }

 double error=0;
 for(uint32_t w=0;w<BATCH_SIZE;w++)
 {
  //считаем ошибку
  for(uint32_t x=0;x<cTensor_Error.GetSizeX();x++)
  {
   for(uint32_t y=0;y<cTensor_Error.GetSizeY();y++)
   {
    for(uint32_t z=0;z<cTensor_Error.GetSizeZ();z++)
    {
     type_t c=cTensor_Error.GetElement(w,z,y,x);
     error+=c*c;
    }
   }
  }
 }
 error/=static_cast<double>(BATCH_SIZE);
 cost=error;

 //выполняем вычисление весов декодировщика
 {
  CTimeStamp cTimeStamp("Обучение декодировщика:");
  for(uint32_t m=0,n=DecoderNet.size()-1;m<DecoderNet.size();m++,n--) DecoderNet[n]->TrainingBackward();
 }

 //CoderNet[CoderNet.size()-1]->GetDeltaTensor().Print("Delta");
 //выполняем вычисление весов кодировщика
 {
  CTimeStamp cTimeStamp("Обучение кодировщика:");
  for(uint32_t m=0,n=CoderNet.size()-1;m<CoderNet.size();m++,n--) CoderNet[n]->TrainingBackward();
 }

 //корректируем веса
 {
  CTimeStamp cTimeStamp("Обновление весов кодировщика:");
  for(uint32_t n=0;n<CoderNet.size();n++) CoderNet[n]->TrainingUpdateWeight(speed,Iteration+1);
 }
 //корректируем веса генератора
 {
  CTimeStamp cTimeStamp("Обновление весов декодировщика:");
  for(uint32_t n=0;n<DecoderNet.size();n++) DecoderNet[n]->TrainingUpdateWeight(speed,Iteration+1);
 }

 cost1=cost;

 CoderNet[0]->GetOutputTensor()=cTensor_Image;

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
  CTensorMath<type_t>::Sub(cTensor_Error,DecoderNet[DecoderNet.size()-1]->GetOutputTensor(),CoderNet[0]->GetOutputTensor(),2,2);
 }

 {
  CTimeStamp cTimeStamp("Задание ошибки:");
  DecoderNet[DecoderNet.size()-1]->SetOutputError(cTensor_Error);
 }

 error=0;
 for(uint32_t w=0;w<BATCH_SIZE;w++)
 {
  //считаем ошибку
  for(uint32_t x=0;x<cTensor_Error.GetSizeX();x++)
  {
   for(uint32_t y=0;y<cTensor_Error.GetSizeY();y++)
   {
    for(uint32_t z=0;z<cTensor_Error.GetSizeZ();z++)
    {
     type_t c=cTensor_Error.GetElement(w,z,y,x);
     error+=c*c;
    }
   }
  }
 }
 error/=static_cast<double>(BATCH_SIZE);
 cost=error;

 cost2=cost;
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelTest<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);

 double cost_a1;
 double cost_a2;
 double cost_b1;
 double cost_b2;

 std::string str;

 SYSTEM::PutMessageToConsole("Тест с размером минипакета 1");
 BATCH_SIZE=1;

 CreateCoder();
 CreateDecoder();

 for(uint32_t n=0;n<CoderNet.size();n++) CoderNet[n]->Reset();
 for(uint32_t n=0;n<DecoderNet.size();n++) DecoderNet[n]->Reset();
 SaveNet();

 LoadNet();
 Test(cost_a1,cost_a2);

 //throw("Стоп");

 SYSTEM::PutMessageToConsole("Тест с размером минипакета 32");

 BATCH_SIZE=32;
 CreateCoder();
 CreateDecoder();

 for(uint32_t n=0;n<CoderNet.size();n++) CoderNet[n]->Reset();
 for(uint32_t n=0;n<DecoderNet.size();n++) DecoderNet[n]->Reset();
 LoadNet();
 Test(cost_b1,cost_b2);


 str="Минипакет 1: Ошибка до:";
 str+=std::to_string(static_cast<long double>(cost_a1));
 SYSTEM::PutMessageToConsole(str);
 str="Минипакет 32: Ошибка до:";
 str+=std::to_string(static_cast<long double>(cost_b1));
 SYSTEM::PutMessageToConsole(str);

 SYSTEM::PutMessageToConsole("");
 str="Минипакет 1: Ошибка после:";
 str+=std::to_string(static_cast<long double>(cost_a2));
 SYSTEM::PutMessageToConsole(str);
 str="Минипакет 32: Ошибка после:";
 str+=std::to_string(static_cast<long double>(cost_b2));
 SYSTEM::PutMessageToConsole(str);

 throw("Стоп");
}

#endif
