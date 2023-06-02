#ifndef C_MODEL_V_H
#define C_MODEL_V_H

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

#include "../../system/system.h"
#include "../../common/tga.h"
#include "../ctimestamp.cu.h"

#include "../../netlayer/cnetlayerlinear.cu.h"
#include "../../netlayer/cnetlayerconvolution.cu.h"
#include "../../netlayer/cnetlayerconvolutioninput.cu.h"
#include "../../netlayer/cnetlayerbackconvolution.cu.h"
#include "../../netlayer/cnetlayermaxpooling.cu.h"
#include "../../netlayer/cnetlayermaxdepooling.cu.h"
#include "../../netlayer/cnetlayerupsampling.cu.h"

#include "../tensor.cu.h"

//****************************************************************************************************
//Главный класс программы
//****************************************************************************************************
template<class type_t>
class CModelV
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
  static const size_t STRING_BUFFER_SIZE=1024;///<размер буфера строки
/*
  static const size_t MNIST_IMAGE_AMOUNT=60000;///<количество обучающих изображений
  static const size_t BATCH_AMOUNT=1000;///<количество пакетов
  static const size_t BATCH_SIZE=MNIST_IMAGE_AMOUNT/BATCH_AMOUNT;///<размер пакета
  */

  static const size_t IMAGE_WIDTH=44; ///<ширина входных изображений
  static const size_t IMAGE_HEIGHT=44; ///<высота входных изображений
  static const size_t NOISE_LAYER_SIDE=16;///<размерность стороны слоя шума
  static const size_t NOISE_LAYER_SIZE=NOISE_LAYER_SIDE*NOISE_LAYER_SIDE*3;///<размерность слоя шума

  size_t BATCH_AMOUNT;///<количество пакетов
  size_t BATCH_SIZE;///<размер пакета
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
  CModelV(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelV();
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
  bool LoadRealMNISTImage(void);//загрузить образы истинных изображений из MNIST
  bool LoadRealImage(void);//загрузить образы истинных изображений
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
CModelV<type_t>::CModelV(void)
{
 BATCH_AMOUNT=0;
 BATCH_SIZE=0;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelV<type_t>::~CModelV()
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
type_t CModelV<type_t>::GetRandValue(type_t max_value)
{
 return((static_cast<type_t>(rand())*max_value)/static_cast<type_t>(RAND_MAX));
}

//----------------------------------------------------------------------------------------------------
//создать сеть генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelV<type_t>::CreateGenerator(void)
{

 GeneratorNet.resize(5);
 GeneratorNet[0]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,NULL));//16x16
 GeneratorNet[0]->GetOutputTensor().ReinterpretSize(3,NOISE_LAYER_SIDE,NOISE_LAYER_SIDE);
 GeneratorNet[1]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[0].get()));
 GeneratorNet[0]->GetOutputTensor().RestoreSize();
 //GeneratorNet[1]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,NULL));//16x16

 GeneratorNet[2]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,32,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[1].get()));//24x24
 GeneratorNet[3]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,64,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[2].get()));//24x24
 GeneratorNet[4]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,3,NNeuron::NEURON_FUNCTION_TANGENCE,GeneratorNet[3].get()));//28x28



/*
 GeneratorNet.resize(6);
 GeneratorNet[0]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,NULL));

 GeneratorNet[1]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(29*29*3,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[0].get()));

 GeneratorNet[1]->GetOutputTensor().ReinterpretSize(3,29,29);
 GeneratorNet[2]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxDePooling<type_t>(2,2,GeneratorNet[1].get()));
 GeneratorNet[1]->GetOutputTensor().RestoreSize();

 GeneratorNet[3]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[2].get()));
 GeneratorNet[4]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,128,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[3].get()));
 GeneratorNet[5]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,3,NNeuron::NEURON_FUNCTION_TANGENCE,GeneratorNet[4].get()));
*/
/*
 GeneratorNet[2]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[1].get()));//48x48
 GeneratorNet[3]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[2].get()));//56x56
 GeneratorNet[4]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[3].get()));//64x64
 GeneratorNet[5]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[4].get()));//72x72
 GeneratorNet[6]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[5].get()));//80x80
 GeneratorNet[7]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[6].get()));//88x88
 GeneratorNet[8]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[7].get()));//96x96
 GeneratorNet[9]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[8].get()));//104x104
 GeneratorNet[10]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[9].get()));//112x112
 GeneratorNet[11]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[10].get()));//120x120
 GeneratorNet[12]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[11].get()));//128x128
 GeneratorNet[13]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[12].get()));//136x136
 GeneratorNet[14]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[13].get()));//144x144
 GeneratorNet[15]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[14].get()));//152x152
 GeneratorNet[16]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[15].get()));//160x160
 GeneratorNet[17]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[16].get()));//168x168
 GeneratorNet[18]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[17].get()));//176x176
 GeneratorNet[19]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,2,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[18].get()));//184x184
 GeneratorNet[20]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,4,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[19].get()));//192x192
 GeneratorNet[21]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,8,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[20].get()));//200x200
 GeneratorNet[22]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,16,NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[21].get()));//208x208
 GeneratorNet[23]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,32,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[22].get()));//212x212
 GeneratorNet[24]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,64,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[23].get()));//216x216
 GeneratorNet[25]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,128,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[24].get()));//220x220
 GeneratorNet[26]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,3,NNeuron::NEURON_FUNCTION_TANGENCE,GeneratorNet[25].get()));//224x224
*/
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
void CModelV<type_t>::CreateDiscriminator(void)
{

 DiscriminatorNet.resize(3);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().ReinterpretSize(3,IMAGE_HEIGHT,IMAGE_WIDTH);
 DiscriminatorNet[0]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get()));//28x28
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().RestoreSize();
 DiscriminatorNet[1]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,5,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[0].get()));
 DiscriminatorNet[2]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1,NNeuron::NEURON_FUNCTION_SIGMOID,DiscriminatorNet[1].get()));

/*
 DiscriminatorNet.resize(4);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().ReinterpretSize(3,IMAGE_HEIGHT,IMAGE_WIDTH);
 DiscriminatorNet[0]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get()));//28x28
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().RestoreSize();
 DiscriminatorNet[1]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[0].get()));
 DiscriminatorNet[2]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[1].get()));
 DiscriminatorNet[3]=std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1,NNeuron::NEURON_FUNCTION_SIGMOID,DiscriminatorNet[2].get()));
*/
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
bool CModelV<type_t>::LoadRealMNISTImage(void)
{
 static const size_t MNIST_IMAGE_AMOUNT=60000;///<количество обучающих изображений
 static const size_t MNIST_IMAGE_WIDTH=28; ///<ширина входных изображений
 static const size_t MNIST_IMAGE_HEIGHT=28; ///<высота входных изображений

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
  uint8_t Color[MNIST_IMAGE_WIDTH*MNIST_IMAGE_HEIGHT];
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
 if (sHeader.Width!=MNIST_IMAGE_WIDTH || sHeader.Height!=MNIST_IMAGE_HEIGHT)
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

 double dx=static_cast<double>(sHeader.Width)/static_cast<double>(IMAGE_WIDTH);
 double dy=static_cast<double>(sHeader.Height)/static_cast<double>(IMAGE_HEIGHT);

 size_t amount=MNIST_IMAGE_AMOUNT;
 amount/=5;
 for(uint32_t n=0;n<amount;n++)
 {
  RealImage.push_back(CTensor<type_t>(3,IMAGE_WIDTH,IMAGE_HEIGHT));
  RealImageIndex.push_back(n);
  SImage sImage;
  if (fread(&sImage,sizeof(SImage),1,file)<1) continue;
  size_t index=0;

  for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
  {
   for(uint32_t x=0;x<IMAGE_WIDTH;x++,index++)
   {
    size_t xp=x*dx;
    size_t yp=y*dy;

    uint32_t offset=(xp+yp*sHeader.Width);
    float c=sImage.Color[offset];
    c/=255.0;
    //приведём к диапазону [-1,1]
    c*=2.0;
    c-=1.0;

    float r=c;
    float g=c;
    float b=c;

    RealImage[n].SetElement(0,y,x,r);
    RealImage[n].SetElement(1,y,x,g);
    RealImage[n].SetElement(2,y,x,b);
   }
  }
 }
 SYSTEM::PutMessageToConsole("Образы MNIST загружены успешно.");
 return(true);
}

//----------------------------------------------------------------------------------------------------
//загрузить образы истинных изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelV<type_t>::LoadRealImage(void)
{
 RealImage.clear();
 RealImageIndex.clear();

 std::string path="RealImage";

 std::vector<std::string> file_list;
 SYSTEM::CreateFileList(path,file_list);
 //обрабатываем файлы
 size_t index=0;
 size_t size=file_list.size();
 RealImage.reserve(size);
 RealImageIndex.reserve(size);

 for(size_t n=0;n<size;n++)
 {
  std::string &file_name=file_list[n];
  //проверяем расширение
  size_t length=file_name.length();
  if (length<4) continue;
  if (file_name[length-4]!='.') continue;
  if (file_name[length-3]!='t' && file_name[length-3]!='T')  continue;
  if (file_name[length-2]!='g' && file_name[length-2]!='G') continue;
  if ((file_name[length-1]!='a' && file_name[length-1]!='A') && (file_name[length-1]!='i' && file_name[length-1]!='I')) continue;//для переименованных в tgi файлов
  //отправляем файл на обработку  int32_t width;
  int32_t height;
  int32_t width;
  std::string name=path+"/"+file_name;
  uint32_t *img_ptr=reinterpret_cast<uint32_t*>(LoadTGAFromFile(name.c_str(),width,height));//загрузить tga-файл
  if (img_ptr==NULL) continue;
  /*if (width!=IMAGE_WIDTH || height!=IMAGE_HEIGHT)
  {
   delete[](img_ptr);
   continue;
  }
  */

  RealImage.push_back(CTensor<type_t>(3,IMAGE_WIDTH,IMAGE_HEIGHT));
  RealImageIndex.push_back(index);

  double dx=static_cast<double>(width)/static_cast<double>(IMAGE_WIDTH);
  double dy=static_cast<double>(height)/static_cast<double>(IMAGE_HEIGHT);

  for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
  {
   for(uint32_t x=0;x<IMAGE_WIDTH;x++)
   {
    size_t xp=x*dx;
    size_t yp=y*dy;

    uint32_t offset=(xp+yp*width);
    uint32_t color=img_ptr[offset];
    float r=(color>>0)&0xff;
    float g=(color>>8)&0xff;
    float b=(color>>16)&0xff;

    r/=255.0;
    g/=255.0;
    b/=255.0;
    //приведём к диапазону [-1,1]
    r*=2.0;
    r-=1.0;

    g*=2.0;
    g-=1.0;

    b*=2.0;
    b-=1.0;

    RealImage[index].SetElement(0,y,x,r);
    RealImage[index].SetElement(1,y,x,g);
    RealImage[index].SetElement(2,y,x,b);
   }
  }
  index++;
  delete[](img_ptr);
 }
 char str[STRING_BUFFER_SIZE];
 sprintf(str,"Загружено реальных изображений:%i",index);
 SYSTEM::PutMessageToConsole(str);
 return(true);
}


//----------------------------------------------------------------------------------------------------
/*!сохранить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelV<type_t>::SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->Save(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelV<type_t>::LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->Load(iDataStream_Ptr);
}

//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelV<type_t>::LoadNet(void)
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
void CModelV<type_t>::SaveNet(void)
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
void CModelV<type_t>::CreateFakeImage(std::vector<CTensor<type_t>> &cTensor_Generator_Output)
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
void CModelV<type_t>::TrainingDiscriminatorFake(double &cost)
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
void CModelV<type_t>::TrainingDiscriminatorReal(size_t mini_batch_index,double &cost)
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
void CModelV<type_t>::TrainingGenerator(double &cost)
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
void CModelV<type_t>::ExchangeRealImageIndex(void)
{
 //делаем перемешивание
 size_t image_amount=RealImage.size();
 for(size_t n=0;n<image_amount;n++)
 {
  size_t index_1=n;
  size_t index_2=static_cast<size_t>((rand()*static_cast<double>(image_amount*10))/static_cast<double>(RAND_MAX));
  index_2%=image_amount;

  size_t tmp=RealImageIndex[index_1];
  RealImageIndex[index_1]=RealImageIndex[index_2];
  RealImageIndex[index_2]=tmp;
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить случайное изображение с генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelV<type_t>::SaveRandomImage(void)
{
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
void CModelV<type_t>::SaveImage(CTensor<type_t> &cTensor_Generator_Output,const std::string &name)
{
 static uint32_t image[IMAGE_WIDTH*IMAGE_HEIGHT];
 type_t *ptr=cTensor_Generator_Output.GetColumnPtr(0,0);
 for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
 {
  for(uint32_t x=0;x<IMAGE_WIDTH;x++)
  {
   double r=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*0];
   double g=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*1];
   double b=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*2];

   r+=1.0;
   r/=2.0;

   if (r<0) r=0;
   if (r>1) r=1;
   r*=255.0;

   g+=1.0;
   g/=2.0;

   if (g<0) g=0;
   if (g>1) g=1;
   g*=255.0;

   b+=1.0;
   b/=2.0;

   if (b<0) b=0;
   if (b>1) b=1;
   b*=255.0;

   uint32_t offset=x+y*IMAGE_WIDTH;
   uint8_t rc=static_cast<uint8_t>(r);
   uint8_t gc=static_cast<uint8_t>(g);
   uint8_t bc=static_cast<uint8_t>(b);

   uint32_t color=0xff;
   color<<=8;
   color|=bc;
   color<<=8;
   color|=gc;
   color<<=8;
   color|=rc;

   image[offset]=color;
  }
 }
 SaveTGA(name.c_str(),IMAGE_WIDTH,IMAGE_HEIGHT,reinterpret_cast<uint8_t*>(image));
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelV<type_t>::Training(void)
{
 const double disc_speed=0.005;//скорость обучения дискриминатора
 const double gen_speed=disc_speed*2;//скорость обучения генератора
 size_t max_iteration=1000000000;//максимальное количество итераций обучения
 uint32_t iteration=0;

 size_t image_amount=RealImage.size();

 std::string str;
 while(iteration<max_iteration)
 {
  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string((long double)iteration));

  for(uint32_t batch=0;batch<BATCH_AMOUNT;batch++)
  {
   if (IsExit()==true) throw("Стоп");

   if (batch%50==0)
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
void CModelV<type_t>::TrainingNet(void)
{
 SYSTEM::MakeDirectory("Test");
// if (LoadRealMNISTImage()==false)
 if (LoadRealImage()==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 //дополняем набор до кратного 10
 size_t batch=10;
 size_t image_amount=RealImage.size();
 for(size_t n=0;n<image_amount%batch;n++)
 {
  RealImage.push_back(RealImage[n]);
  RealImageIndex.push_back(RealImageIndex[n]);
 }
 image_amount=RealImage.size();
 BATCH_SIZE=image_amount/batch;
 BATCH_AMOUNT=batch;

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
bool CModelV<type_t>::IsExit(void)
{
 return(false);
}

//----------------------------------------------------------------------------------------------------
//задать необходимость выхода из потока
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelV<type_t>::SetExitState(bool state)
{

}

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelV<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);
 if (CTensorTest<type_t>::Test()==false) throw("Класс тензоров провалил тестирование!");
 TrainingNet();
}



#endif
