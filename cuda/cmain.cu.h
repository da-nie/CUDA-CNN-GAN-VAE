#ifndef C_MAIN_H
#define C_MAIN_H

//****************************************************************************************************
//Главный класс программы
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <string>
#include <vector>
#include <memory>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../system/system.h"
#include "../common/tga.h"
#include "ctimestamp.cu.h"

#include "../netlayer/cnetlayerlinear.cu.h"
#include "../netlayer/cnetlayerconvolution.cu.h"
#include "../netlayer/cnetlayerconvolutioninput.cu.h"
#include "../netlayer/cnetlayerbackconvolution.cu.h"
#include "../netlayer/cnetlayermaxpooling.cu.h"

#include "tensor.cu.h"

//****************************************************************************************************
//Главный класс программы
//****************************************************************************************************
template<class type_t>
class CMain
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
  static const size_t STRING_BUFFER_SIZE=1024;///<размер буфера строки

  static const size_t MNIST_IMAGE_AMOUNT=60000;///<количество обучающих изображений
  static const size_t BATCH_AMOUNT=1000;///<количество пакетов
  static const size_t BATCH_SIZE=MNIST_IMAGE_AMOUNT/BATCH_AMOUNT;///<размер пакета

  static const size_t IMAGE_WIDTH=28; ///<ширина входных изображений
  static const size_t IMAGE_HEIGHT=28; ///<высота входных изображений
  static const size_t NOISE_LAYER_SIDE=16;///<размерность стороны слоя шума
  static const size_t NOISE_LAYER_SIZE=NOISE_LAYER_SIDE*NOISE_LAYER_SIDE;///<размерность слоя шума
 private:
  //-структуры------------------------------------------------------------------------------------------
  struct SProtectedVariables
  {
   bool OnExit;
  };
  //-переменные-----------------------------------------------------------------------------------------
  SProtectedVariables sProtectedVariables;

  std::vector<std::shared_ptr<INetLayer<type_t> > > GeneratorNet;///<сеть генератора
  std::vector<std::shared_ptr<INetLayer<type_t> > > DiscriminatorNet;///<сеть дискриминатора

  std::vector<CTensor<type_t>> cTensor_Generator_Output;
  std::vector<CTensor<type_t>> cTensor_Discriminator_Output;
  std::vector<CTensor<type_t>> cTensor_Discriminator_Error;
  std::vector<CTensor<type_t>> cTensor_Discriminator_Real_Image_Input;

  std::vector<CTensor<type_t>> RealImage;//образы истинных изображений
  std::vector<size_t> RealImageIndex;//индексы изображений в обучающем наборе
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CMain(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CMain();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  bool IsExit(void);//нужно ли выйти из потока
  void SetExitState(bool state);//задать необходимость выхода из потока
  void Execute(void);//выполнить
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
  type_t GetRandValue(type_t max_value);//случайное число
  void CreateGenerator(void);//создать сеть генератора
  void CreateDiscriminator(void);//создать сеть дискриминатора
  bool LoadRealMNISTImage(void);//загрузить образы истинных изображений MNIST
  bool LoadRealImage(void);//загрузить образы истинных изображений
  void CreateMNISTImage(void);//создать набор отдельных картинок
  void SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<сохранить слои сети
  void LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<загрузить слои сети
  void LoadNet(void);//загрузить сети
  void SaveNet(void);//сохранить сети
  void CreateFakeImage(std::vector<CTensor<type_t>> &cTensor_Generator_Output);//создать мнимое изображение с помощью генератора
  void TrainingDiscriminatorFake(double &cost);//обучение дискриминатора на фальшивом изображенияз
  void TrainingDiscriminatorReal(size_t mini_batch_index,double &cost);//обучение дискриминатора на настоящих изображениях
  void TrainingGenerator(double &cost);//обучение генератора
  void ExchangeRealImageIndex(void);//перемешать индексы изображений
  void SaveRandomImage(void);//сохранить случайное изображение с генератора
  void SaveImage(CTensor<type_t> &cTensor_Generator_Output,const std::string &name);//сохранить изображение
  void Training(void);//обучение нейросети
  void TrainingNet(void);//обучение нейросети
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMain<type_t>::CMain(void):RealImage(MNIST_IMAGE_AMOUNT),RealImageIndex(MNIST_IMAGE_AMOUNT)
{
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMain<type_t>::~CMain()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************
template<class type_t>
type_t SafeLog(type_t value)
{
 if (value>0) return(log(value));
 SYSTEM::PutMessageToConsole("Error log!");
 return(-100000);
}
template<class type_t>
type_t CrossEntropy(type_t y,type_t p)
{
 type_t s=y*SafeLog(p)+(1-y)*SafeLog(1-p);
 return(-s);
}

//----------------------------------------------------------------------------------------------------
//случайное число
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CMain<type_t>::GetRandValue(type_t max_value)
{
 return((static_cast<type_t>(rand())*max_value)/static_cast<type_t>(RAND_MAX));
}

//----------------------------------------------------------------------------------------------------
//создать сеть генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::CreateGenerator(void)
{

 GeneratorNet.resize(4);
 GeneratorNet[0]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,NULL));//16x16

 GeneratorNet[0]->GetOutputTensor().ReinterpretSize(1,NOISE_LAYER_SIDE,NOISE_LAYER_SIDE);
 GeneratorNet[1]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,32,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[0].get()));//20x20
 GeneratorNet[0]->GetOutputTensor().RestoreSize();

 GeneratorNet[2]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,64,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[1].get()));//24x24
 GeneratorNet[3]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,1,NNeuron::NEURON_FUNCTION_TANGENCE,GeneratorNet[2].get()));//28x28

/*
 GeneratorNet.resize(5);
 GeneratorNet[0]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,NULL));
 GeneratorNet[1]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(256,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[0].get()));
 GeneratorNet[2]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(512,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[1].get()));
 GeneratorNet[3]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1024,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[2].get()));
 GeneratorNet[4]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(IMAGE_WIDTH*IMAGE_HEIGHT,NNeuron::NEURON_FUNCTION_TANGENCE,GeneratorNet[3].get()));
*/
}
//----------------------------------------------------------------------------------------------------
//создать сеть дискриминатора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::CreateDiscriminator(void)
{
 DiscriminatorNet.resize(3);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().ReinterpretSize(1,IMAGE_HEIGHT,IMAGE_WIDTH);
 DiscriminatorNet[0]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get()));//28x28
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().RestoreSize();
 DiscriminatorNet[1]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,5,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[0].get()));
 DiscriminatorNet[2]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1,NNeuron::NEURON_FUNCTION_SIGMOID,DiscriminatorNet[1].get()));

/*
 DiscriminatorNet.resize(4);
 DiscriminatorNet[0]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1024,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get()));
 DiscriminatorNet[1]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(512,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[0].get()));
 DiscriminatorNet[2]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(256,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[1].get()));
 DiscriminatorNet[3]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1,NNeuron::NEURON_FUNCTION_SIGMOID,DiscriminatorNet[2].get()));
*/
}

//----------------------------------------------------------------------------------------------------
//загрузить образы истинных изображений MNIST
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CMain<type_t>::LoadRealMNISTImage(void)
{
 #pragma pack(1)
 //все числа в big-endian!
 struct SHeader
 {
  uint32_t MagicNumber;
  uint32_t NumberOfImages;
  uint32_t Height;
  uint32_t Width;
 };
 struct SImage
 {
  uint8_t Color[IMAGE_WIDTH*IMAGE_HEIGHT];
 };
 #pragma pack()

 FILE *file=fopen("mnist.bin","rb");
 if (file==NULL)
 {
  SYSTEM::PutMessageToConsole("Отсутствует файл mnist.bin!");
  return(false);
 }
 SHeader sHeader;
 if (fread(&sHeader,sizeof(SHeader),1,file)<1)
 {
  SYSTEM::PutMessageToConsole("Ошибка данных в mnist.bin!");
  fclose(file);
  return(false);
 }
 //переведём заголовок в little-endian
 for(uint32_t n=0;n<4;n++)
 {
  uint8_t *b_ptr=reinterpret_cast<uint8_t*>(&sHeader);
  b_ptr+=sizeof(uint32_t)*n;
  uint8_t *e_ptr=b_ptr+sizeof(uint32_t)-1;
  for(uint32_t m=0;m<(sizeof(uint32_t)>>1);m++,b_ptr++,e_ptr--)
  {
   uint8_t tmp=*b_ptr;
   *b_ptr=*e_ptr;
   *e_ptr=tmp;
  }
 }
 if (sHeader.Width!=IMAGE_WIDTH || sHeader.Height!=IMAGE_HEIGHT)
 {
  SYSTEM::PutMessageToConsole("Размеры изображения в файле MNIST неверные!");
  fclose(file);
  return(false);
 }
 if (sHeader.NumberOfImages<MNIST_IMAGE_AMOUNT)
 {
  SYSTEM::PutMessageToConsole("Слишком мало изображений в файле MNIST!");
  fclose(file);
  return(false);
 }
 for(uint32_t n=0;n<MNIST_IMAGE_AMOUNT;n++)
 {
  RealImage[n]=CTensor<type_t>(1,IMAGE_WIDTH,IMAGE_HEIGHT);
  RealImageIndex[n]=n;
  SImage sImage;
  if (fread(&sImage,sizeof(SImage),1,file)<1) continue;
  size_t index=0;
  for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
  {
   for(uint32_t x=0;x<IMAGE_WIDTH;x++,index++)
   {
    uint32_t offset=(x+y*sHeader.Width);
    float c=sImage.Color[offset];
    c/=255.0;
    //приведём к диапазону [-1,1]
    c*=2.0;
    c-=1.0;
    RealImage[n].SetElement(0,y,x,c);
   }
  }
 }
 SYSTEM::PutMessageToConsole("Образы MNIST загружены успешно.");
 return(true);
}


//----------------------------------------------------------------------------------------------------
/*!сохранить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->Save(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->Load(iDataStream_Ptr);
}

//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::LoadNet(void)
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
void CMain<type_t>::SaveNet(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("disc_neuronet.net",true));
 SaveNetLayers(iDataStream_Disc_Ptr.get(),DiscriminatorNet);

 std::unique_ptr<IDataStream> iDataStream_Gen_Ptr(IDataStream::CreateNewDataStreamFile("gen_neuronet.net",true));
 SaveNetLayers(iDataStream_Gen_Ptr.get(),GeneratorNet);
}

//----------------------------------------------------------------------------------------------------
//создать мнимое изображение с помощью генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::CreateFakeImage(std::vector<CTensor<type_t>> &cTensor_Generator_Output)
{
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(1,NOISE_LAYER_SIZE,1);
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  for(uint32_t n=0;n<NOISE_LAYER_SIZE;n++)
  {
   type_t r=GetRandValue(20.0)-10.0;
   cTensor_Generator_Input.SetElement(0,n,0,r);
  }
  GeneratorNet[0]->SetOutput(cTensor_Generator_Input);//входной вектор
  //выполняем прямой проход по сети
  for(size_t layer=0;layer<GeneratorNet.size();layer++)
  {
   //printf("Layer:%i\r\n",layer);
   GeneratorNet[layer]->Forward();
  }
  //получаем ответ сети
  cTensor_Generator_Output[b]=GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor();
 }
}

//----------------------------------------------------------------------------------------------------
//обучение дискриминатора на фальшивом наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::TrainingDiscriminatorFake(double &cost)
{
 double fake_output=0;
 //создаём изображения с генератора
 {
  CTimeStamp cTimeStamp("Создание шума:");
  CreateFakeImage(cTensor_Generator_Output);
 }

 for(size_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");

  //подаём на вход дискриминатора мнимое изображение
  {
   CTimeStamp cTimeStamp("Задание шума:");
   //дискриминатор подключён к выходу генератора
   GeneratorNet[GeneratorNet.size()-1]->SetOutput(cTensor_Generator_Output[b]);
  }
  //вычисляем сеть
  {
   CTimeStamp cTimeStamp("Вычисление дискриминатора:");
   for(size_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
  }
  //вычисляем ошибку последнего слоя
  {
   CTimeStamp cTimeStamp("Получение ответа дискриминатора:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output[b]);

   if (b==0)
   {
    char str[255];
    sprintf(str,"Ответ дискриминатора на фальшивку [%i]:%f",b,cTensor_Discriminator_Output[b].GetElement(0,0,0));
    SYSTEM::PutMessageToConsole(str);
   }
  }
  {
   CTimeStamp cTimeStamp("Расчёт ошибки:");
   fake_output=cTensor_Discriminator_Output[b].GetElement(0,0,0);
   //double disc_fake_error=-SafeLog(1.0-fake_output);
   //cost+=disc_fake_error*disc_fake_error;
   double disc_fake_error=(fake_output-0);
   cost+=disc_fake_error*disc_fake_error;
   cTensor_Discriminator_Error[b].SetElement(0,0,0,disc_fake_error);
  }
  //задаём ошибку дискриминатору
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error[b]);
  }
  //выполняем вычисление весов
  {
   CTimeStamp cTimeStamp("Обучение дискриминатора:");
   for(size_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward();
  }
 }
}

//----------------------------------------------------------------------------------------------------
//обучение дискриминатора на настоящем наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::TrainingDiscriminatorReal(size_t mini_batch_index,double &cost)
{
 {
  CTimeStamp cTimeStamp("Создание набора изображений:");
  for(size_t n=0;n<BATCH_SIZE;n++)
  {
   //создаём набор настоящих изображений
   size_t image_index=RealImageIndex[n+mini_batch_index*BATCH_SIZE];
   cTensor_Discriminator_Real_Image_Input[n]=RealImage[image_index];
  }
 }

 double real_output=0;
 for(size_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");

  //подаём на вход генератора истинное изображение
  {
   CTimeStamp cTimeStamp("Задание изображения:");
   //дискриминатор подключён к выходу генератора
   GeneratorNet[GeneratorNet.size()-1]->SetOutput(cTensor_Discriminator_Real_Image_Input[b]);
  }
  //вычисляем сеть
  {
   CTimeStamp cTimeStamp("Вычисление дискриминатора:");
   for(size_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
  }
  //вычисляем ошибку последнего слоя
  {
   CTimeStamp cTimeStamp("Получение ответа дискриминатора:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output[b]);
   if (b==0)
   {
    char str[255];
    sprintf(str,"Ответ дискриминатора на истину [%i]:%f",b,cTensor_Discriminator_Output[b].GetElement(0,0,0));
    SYSTEM::PutMessageToConsole(str);
   }
  }
  {
   CTimeStamp cTimeStamp("Вычисление ошибки:");
   real_output=cTensor_Discriminator_Output[b].GetElement(0,0,0);
   //double disc_real_error=SafeLog(real_output);
   //cost+=disc_real_error*disc_real_error;
   double disc_real_error=(real_output-1);
   cost+=disc_real_error*disc_real_error;
   cTensor_Discriminator_Error[b].SetElement(0,0,0,disc_real_error);
  }
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   //задаём ошибку дискриминатору
   DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error[b]);
  }
  //выполняем вычисление весов
  {
   CTimeStamp cTimeStamp("Обучение дискриминатора:");
   for(size_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward();
  }
 }
}

//----------------------------------------------------------------------------------------------------
//обучение генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::TrainingGenerator(double &cost)
{
 double fake_output=0;
 static CTensor<type_t> cTensor_Generator_Input=CTensor<type_t>(1,NOISE_LAYER_SIZE,1);

 for(size_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");

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
  {
   CTimeStamp cTimeStamp("Вычисление дискриминатора:");
   for(size_t layer=0;layer<DiscriminatorNet.size();layer++) DiscriminatorNet[layer]->Forward();
  }
  //вычисляем ошибку
  {
   CTimeStamp cTimeStamp("Получение ответа:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutput(cTensor_Discriminator_Output[b]);
   if (b==0)
   {
    char str[255];
    sprintf(str,"Ответ дискриминатора для генератора [%i]:%f",b,cTensor_Discriminator_Output[b].GetElement(0,0,0));
    SYSTEM::PutMessageToConsole(str);
   }
  }
  {
   CTimeStamp cTimeStamp("Расчёт ошибки дискриминатора:");
   fake_output=cTensor_Discriminator_Output[b].GetElement(0,0,0);
   // double disc_error=SafeLog(fake_output);
   //cost+=disc_error*disc_error;
   double disc_error=(fake_output-1);
   cost+=disc_error*disc_error;
   cTensor_Discriminator_Error[b].SetElement(0,0,0,disc_error);
  }
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   DiscriminatorNet[DiscriminatorNet.size()-1]->SetOutputError(cTensor_Discriminator_Error[b]);
  }
  //выполняем вычисление весов дискриминатора
  {
   CTimeStamp cTimeStamp("Обучение дискриминатора (получение ошибок):");
   for(size_t m=0,n=DiscriminatorNet.size()-1;m<DiscriminatorNet.size();m++,n--) DiscriminatorNet[n]->TrainingBackward();
  }
  //выполняем вычисление весов генератора
  {
   CTimeStamp cTimeStamp("Обучение генератора:");
   for(size_t m=0,n=GeneratorNet.size()-1;m<GeneratorNet.size();m++,n--) GeneratorNet[n]->TrainingBackward();
  }
 }
}

//----------------------------------------------------------------------------------------------------
//перемешать индексы изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::ExchangeRealImageIndex(void)
{
 //делаем перемешивание
 for(size_t n=0;n<MNIST_IMAGE_AMOUNT;n++)
 {
  size_t index_1=n;
  size_t index_2=static_cast<size_t>((rand()*static_cast<double>(MNIST_IMAGE_AMOUNT*10))/static_cast<double>(RAND_MAX));
  index_2%=MNIST_IMAGE_AMOUNT;

  size_t tmp=RealImageIndex[index_1];
  RealImageIndex[index_1]=RealImageIndex[index_2];
  RealImageIndex[index_2]=tmp;
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить случайное изображение с генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::SaveRandomImage(void)
{
 for(size_t n=0;n<BATCH_SIZE;n++) cTensor_Generator_Output[n]=CTensor<type_t>(IMAGE_WIDTH*IMAGE_HEIGHT,1);
 CreateFakeImage(cTensor_Generator_Output);
 //SaveImage(cTensor_Generator_Output[0],"Test/test.tga");
 for(size_t n=0;n<BATCH_SIZE;n++)
 {
  char str[255];
  sprintf(str,"Test/test%03i.tga",n);
  SaveImage(cTensor_Generator_Output[n],str);
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить изображение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::SaveImage(CTensor<type_t> &cTensor_Generator_Output,const std::string &name)
{
 uint8_t image[IMAGE_WIDTH*IMAGE_HEIGHT*4];

 type_t *ptr=cTensor_Generator_Output.GetColumnPtr(0,0);
 for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
 {
  for(uint32_t x=0;x<IMAGE_WIDTH;x++)
  {
   double c=ptr[x+y*IMAGE_WIDTH];

   c+=1.0;
   c/=2.0;

   if (c<0) c=0;
   if (c>1) c=1;
   c*=255.0;
   uint32_t offset=(x+y*IMAGE_WIDTH)*4;
   image[offset+0]=c;
   image[offset+1]=c;
   image[offset+2]=c;
   image[offset+3]=255;
  }
 }
 SaveTGA(name.c_str(),IMAGE_WIDTH,IMAGE_HEIGHT,image);
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::Training(void)
{
 const double disc_speed=0.01;//скорость обучения дискриминатора
 const double gen_speed=disc_speed*2.0;//скорость обучения генератора
 size_t max_iteration=1000000000;//максимальное количество итераций обучения
 uint32_t iteration=0;

 std::string str;
 while(iteration<max_iteration)
 {
  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string((long double)iteration));

  for(uint32_t batch=0;batch<BATCH_AMOUNT;batch++)
  {
   if (IsExit()==true) throw("Стоп");

   if (batch%10==0)
   {
    SYSTEM::PutMessageToConsole("Save net.");
    SaveNet();
    SYSTEM::PutMessageToConsole("Save image.");
    SaveRandomImage();
    SYSTEM::PutMessageToConsole("");
   }

   str="Итерация:";
   str+=std::to_string((long double)iteration+1);
   str+=" минипакет:";
   str+=std::to_string((long double)batch+1);
   SYSTEM::PutMessageToConsole(str);

   long double begin_time=SYSTEM::GetSecondCounter();

   //обучаем дискриминатор
   double disc_cost=0;

   for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingResetDeltaWeight();
   TrainingDiscriminatorFake(disc_cost);
   //корректируем веса дискриминатора
   {
    CTimeStamp cTimeStamp("Обновление весов дискриминатора на фальшивых изображениях:");
    for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingUpdateWeight(disc_speed/(static_cast<double>(BATCH_SIZE)));
   }

   for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingResetDeltaWeight();
   TrainingDiscriminatorReal(batch,disc_cost);
   //корректируем веса дискриминатора
   {
    CTimeStamp cTimeStamp("Обновление весов дискриминатора на настоящих изображениях:");
    for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingUpdateWeight(disc_speed/(static_cast<double>(BATCH_SIZE)));
   }

   str="Ошибка дискриминатора:";
   str+=std::to_string((long double)disc_cost);
   SYSTEM::PutMessageToConsole(str);

   //обучаем генератор
   double gen_cost=0;
   for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingResetDeltaWeight();
   TrainingGenerator(gen_cost);

   //корректируем веса генератора
   {
    CTimeStamp cTimeStamp("Обновление весов генератора:");
    for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingUpdateWeight(gen_speed/(static_cast<double>(BATCH_SIZE)));
   }

   str="Ошибка генератора:";
   str+=std::to_string((long double)gen_cost);
   SYSTEM::PutMessageToConsole(str);

   long double end_time=SYSTEM::GetSecondCounter();
   float cpu_time=static_cast<float>((end_time-begin_time)*1000.0);

   char str[255];
   sprintf(str,"На минипакет ушло: %.2f мс.\r\n",cpu_time);
   SYSTEM::PutMessageToConsole(str);
   SYSTEM::PutMessageToConsole("");
  }
  ExchangeRealImageIndex();
  SYSTEM::PutMessageToConsole("Save net.");
  SaveNet();
  SYSTEM::PutMessageToConsole("Save image.");
  SaveRandomImage();
  iteration++;
 }
}


//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::TrainingNet(void)
{
 SYSTEM::MakeDirectory("Test");
 if (LoadRealMNISTImage()==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }

 cTensor_Generator_Output=std::vector<CTensor<type_t>>(BATCH_SIZE);
 cTensor_Discriminator_Real_Image_Input=std::vector<CTensor<type_t>>(BATCH_SIZE);
 cTensor_Discriminator_Output=std::vector<CTensor<type_t>>(BATCH_SIZE);
 cTensor_Discriminator_Error=std::vector<CTensor<type_t>>(BATCH_SIZE);
 for(size_t n=0;n<BATCH_SIZE;n++)
 {
  cTensor_Generator_Output[n]=CTensor<type_t>(1,IMAGE_WIDTH,IMAGE_HEIGHT);
  cTensor_Discriminator_Real_Image_Input[n]=CTensor<type_t>(1,IMAGE_WIDTH,IMAGE_HEIGHT);
  cTensor_Discriminator_Output[n]=CTensor<type_t>(1,1,1);
  cTensor_Discriminator_Error[n]=CTensor<type_t>(1,1,1);
 }

 CreateGenerator();
 CreateDiscriminator();

 for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->Reset();
 for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->Reset();

 LoadNet();

 //включаем обучение
 for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingStart();
 for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingStart();

 Training();
 //отключаем обучение
 for(size_t n=0;n<GeneratorNet.size();n++) GeneratorNet[n]->TrainingStop();
 for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingStop();

 SaveNet();
}


//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//нужно ли выйти из потока
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CMain<type_t>::IsExit(void)
{
 return(false);
}

//----------------------------------------------------------------------------------------------------
//задать необходимость выхода из потока
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMain<type_t>::SetExitState(bool state)
{

}

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CMain<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);

 if (CTensorTest<type_t>::Test()==false) throw("Класс тензоров провалил тестирование!");

 //измеряем время свёрток


 {
  SYSTEM::PutMessageToConsole("Тест скорости функции ForwardConvolution.");
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
  std::vector<CTensor<type_t>> cTensor_Kernel_Test;
  cTensor_Kernel.push_back(cTensor_KernelA);
  cTensor_Kernel.push_back(cTensor_KernelB);

  for(size_t n=0;n<128;n++)
  {
   cTensor_Kernel_Test.push_back(cTensor_KernelA);
   cTensor_Kernel_Test.push_back(cTensor_KernelB);
  }
  //создаём вектор смещений
  std::vector<type_t> bias;
  std::vector<type_t> bias_test;
  bias.push_back(0);
  bias.push_back(0);

  for(size_t n=0;n<128;n++)
  {
   bias_test.push_back(0);
   bias_test.push_back(0);
  }

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
  {
   CTensor<type_t> cTensor_ImageMax(3,300,300);
   //выходной тензор свёртки
   CTensor<type_t> cTensor_OutputMax(256,299,299);
   CTimeStamp cTimeStamp("Скорость прямой свёрти:");
   {
    for(size_t n=0;n<1;n++)
    {
     CTensorConv<type_t>::ForwardConvolution(cTensor_OutputMax,cTensor_ImageMax,cTensor_Kernel_Test,bias_test,1,1,0,0);
    }
   }
  }

  //выполняем обратную свёртку
  {
   CTensor<type_t> cTensor_DeltaMax(256,299,299);
   //выходной тензор свёртки
   CTensor<type_t> cTensor_OutputDeltaMax(3,300,300);
   CTimeStamp cTimeStamp("Скорость обратной свёрти:");
   {
    for(size_t n=0;n<1;n++)
    {
     CTensorConv<type_t>::BackwardConvolution(cTensor_OutputDeltaMax,cTensor_DeltaMax,cTensor_Kernel_Test,bias_test);
    }
   }
  }


  //сравниваем полученный тензор
  CTensorConv<type_t>::ForwardConvolution(cTensor_Output,cTensor_Image,cTensor_Kernel,bias,1,1,0,0);
  if (cTensor_Output.Compare(cTensor_Control,"")==false) throw("Свёртка неправильная!");
  SYSTEM::PutMessageToConsole("Успешно.");
 }




 TrainingNet();
}



#endif
