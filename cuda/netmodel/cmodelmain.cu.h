#ifndef C_MAIN_H
#define C_MAIN_H

//****************************************************************************************************
//Класс-основа
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

#include "../../netlayer/cnetlayerfunction.cu.h"
#include "../../netlayer/cnetlayerlinear.cu.h"
#include "../../netlayer/cnetlayerdropout.cu.h"
#include "../../netlayer/cnetlayerbatchnormalization.cu.h"
#include "../../netlayer/cnetlayerconvolution.cu.h"
#include "../../netlayer/cnetlayerconvolutioninput.cu.h"
#include "../../netlayer/cnetlayerbackconvolution.cu.h"
#include "../../netlayer/cnetlayermaxpooling.cu.h"
#include "../../netlayer/cnetlayermaxdepooling.cu.h"
#include "../../netlayer/cnetlayerupsampling.cu.h"
#include "../../netlayer/cnetlayeraveragepooling.cu.h"

#include "../tensor.cu.h"

//****************************************************************************************************
//Класс-основа
//****************************************************************************************************
template<class type_t>
class CModelMain
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
  static const size_t STRING_BUFFER_SIZE=1024;///<размер буфера строки
  static const size_t CUDA_PAUSE_MS=1;///<пауза для CUDA
 protected:
  //-структуры------------------------------------------------------------------------------------------
  struct SProtectedVariables
  {
   bool OnExit;
  };
  //-переменные-----------------------------------------------------------------------------------------
  SProtectedVariables sProtectedVariables;

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
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelMain(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelMain();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  type_t GetRandValue(type_t max_value);//случайное число
  type_t SafeLog(type_t value);//логарифм с ограничением по размеру
  type_t CrossEntropy(type_t y,type_t p);//перекрёстная энтропия
  bool IsExit(void);//нужно ли выйти из потока
  void SetExitState(bool state);//задать необходимость выхода из потока
  void Execute(void);//выполнить
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  virtual void CreateGenerator(void)=0;//создать сеть генератора
  virtual void CreateDiscriminator(void)=0;//создать сеть дискриминатора
  bool LoadRealMNISTImage(void);//загрузить образы истинных изображений из MNIST
  bool LoadRealImage(void);//загрузить образы истинных изображений
  void SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<сохранить слои сети
  void LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<загрузить слои сети
  void SaveNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<сохранить параметры обучения слоёв сети
  void LoadNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<загрузить параметры обучения слоёв сети
  void LoadNet(void);//загрузить сети
  void SaveNet(void);//сохранить сети
  void LoadTrainingParam(void);//загрузить параметры обучения
  void SaveTrainingParam(void);//сохранить параметры обучения
  void CreateFakeImage(CTensor<type_t> &cTensor_Generator_Image);//создать мнимое изображение с помощью генератора
  void TrainingDiscriminatorFake(double &cost);//обучение дискриминатора на фальшивом изображения
  void TrainingDiscriminatorReal(size_t mini_batch_index,double &cost);//обучение дискриминатора на настоящих изображениях
  void TrainingGenerator(double &cost,double &max_disc_answer);//обучение генератора
  void ExchangeRealImageIndex(void);//перемешать индексы изображений
  void SaveRandomImage(void);//сохранить случайное изображение с генератора
  void SaveKitImage(void);//сохранить изображение из набора
  void SaveImage(CTensor<type_t> &cTensor_Generator_Output,const std::string &name);//сохранить изображение
  void Training(void);//обучение нейросети
  virtual void TrainingNet(bool mnist);//запуск обучения нейросети
  void TestTrainingGenerator(void);//тест обучения генератора
  void TestTrainingGeneratorNet(bool mnist);//запуск теста обучения генератора

  void SpeedTest(void);//тест скорости
};

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelMain<type_t>::CModelMain(void)
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
CModelMain<type_t>::~CModelMain()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//загрузить образы истинных изображений MNIST
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelMain<type_t>::LoadRealMNISTImage(void)
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
  RealImage.push_back(std::vector<type_t>(IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_DEPTH));

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
    float gray=sImage.Color[offset];
    gray/=255.0;
    //приведём к диапазону [-1,1]
    gray*=2.0;
    gray-=1.0;

    float r=gray;
    float g=gray;
    float b=gray;

    if (IMAGE_DEPTH==1)
	{
	 RealImage[x+y*IMAGE_WIDTH]=gray;
	}
    if (IMAGE_DEPTH==3)
	{
	 RealImage[x+y*IMAGE_WIDTH+0*IMAGE_WIDTH*IMAGE_HEIGHT]=r;
	 RealImage[x+y*IMAGE_WIDTH+1*IMAGE_WIDTH*IMAGE_HEIGHT]=g;
	 RealImage[x+y*IMAGE_WIDTH+2*IMAGE_WIDTH*IMAGE_HEIGHT]=b;
	}
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
bool CModelMain<type_t>::LoadRealImage(void)
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
  //изображение
  std::vector<type_t> NormalImage=std::vector<type_t>(IMAGE_DEPTH*IMAGE_HEIGHT*IMAGE_WIDTH);
  //зеркальное по ширине изображение
  std::vector<type_t> FlipImage=std::vector<type_t>(IMAGE_DEPTH*IMAGE_HEIGHT*IMAGE_WIDTH);

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

    float gray=(r+g+b)/3.0f;
    gray/=255.0;
    gray*=2.0;
    gray-=1.0;

	float sl;
	float sa;
	float sb;

    CColorModel::RGB2Lab(r,g,b,sl,sa,sb);

    sl*=2.0;
    sl-=1.0;

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

    if (IMAGE_DEPTH==1)
	{
	 NormalImage[x+y*IMAGE_WIDTH]=gray;
	 FlipImage[(IMAGE_WIDTH-x-1)+y*IMAGE_WIDTH]=gray;
	}
    if (IMAGE_DEPTH==3)
	{
     r=sl;
     g=sa;
     b=sb;

     if (r<-1) r=-1;
     if (g<-1) g=-1;
     if (b<-1) b=-1;

     if (r>1) r=1;
     if (g>1) g=1;
     if (b>1) b=1;

	 NormalImage[x+y*IMAGE_WIDTH+0*IMAGE_WIDTH*IMAGE_HEIGHT]=r;
	 NormalImage[x+y*IMAGE_WIDTH+1*IMAGE_WIDTH*IMAGE_HEIGHT]=g;
	 NormalImage[x+y*IMAGE_WIDTH+2*IMAGE_WIDTH*IMAGE_HEIGHT]=b;

	 FlipImage[(IMAGE_WIDTH-x-1)+y*IMAGE_WIDTH+0*IMAGE_WIDTH*IMAGE_HEIGHT]=r;
	 FlipImage[(IMAGE_WIDTH-x-1)+y*IMAGE_WIDTH+1*IMAGE_WIDTH*IMAGE_HEIGHT]=g;
	 FlipImage[(IMAGE_WIDTH-x-1)+y*IMAGE_WIDTH+2*IMAGE_WIDTH*IMAGE_HEIGHT]=b;
	}
   }
  }

  RealImage.push_back(NormalImage);
  RealImageIndex.push_back(index);
  index++;
/*
  RealImage.push_back(FlipImage);
  RealImageIndex.push_back(index);
  index++;
  */

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
void CModelMain<type_t>::SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->Save(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->Load(iDataStream_Ptr);
}

//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоёв сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::SaveNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->SaveTrainingParam(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоёв сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::LoadNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->LoadTrainingParam(iDataStream_Ptr);
}

//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::LoadNet(void)
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
void CModelMain<type_t>::SaveNet(void)
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
void CModelMain<type_t>::LoadTrainingParam(void)
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
void CModelMain<type_t>::SaveTrainingParam(void)
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
void CModelMain<type_t>::CreateFakeImage(CTensor<type_t> &cTensor_Generator_Image)
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
//обучение дискриминатора на фальшивом наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::TrainingDiscriminatorFake(double &cost)
{
 char str[STRING_BUFFER_SIZE];
 double fake_output=0;

 for(size_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  //создаём изображение с генератора
  {
   CTimeStamp cTimeStamp("Создание шума:");
   CreateFakeImage(cTensor_Generator_Output);
  }
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту

  //подаём на вход дискриминатора мнимое изображение
  {
   CTimeStamp cTimeStamp("Задание шума:");
   //дискриминатор подключён к выходу генератора
   GeneratorNet[GeneratorNet.size()-1]->SetOutput(cTensor_Generator_Output);
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
    sprintf(str,"Ответ дискриминатора на фальшивку [%i]:%f",b,cTensor_Discriminator_Output.GetElement(0,0,0));
    SYSTEM::PutMessageToConsole(str);
   }
  }
  {
   CTimeStamp cTimeStamp("Расчёт ошибки:");
   fake_output=cTensor_Discriminator_Output.GetElement(0,0,0);
   double disc_fake_error=-SafeLog(1.0-fake_output);//прямая метка
   //double disc_fake_error=(fake_output-0);//прямая метка

   if (GetRandValue(100)>95)//инвертируем метки, чтобы избежать переобучения генератора
   {
    //disc_fake_error=SafeLog(fake_output);//инверсная метка
   }


   cost+=disc_fake_error*disc_fake_error;
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
  SYSTEM::PauseInMs(CUDA_PAUSE_MS);//чтобы не перегревать видеокарту
 }
}

//----------------------------------------------------------------------------------------------------
//обучение дискриминатора на настоящем наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::TrainingDiscriminatorReal(size_t mini_batch_index,double &cost)
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
void CModelMain<type_t>::TrainingGenerator(double &cost,double &max_disc_answer)
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
//перемешать индексы изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::ExchangeRealImageIndex(void)
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
void CModelMain<type_t>::SaveRandomImage(void)
{
 static size_t counter=0;

 char str[STRING_BUFFER_SIZE];
 for(size_t n=0;n<BATCH_SIZE*0+1;n++)
 {
  CreateFakeImage(cTensor_Generator_Output);
  sprintf(str,"Test/test%05i-%03i.tga",counter,n);
  //SaveImage(cTensor_Generator_Output,str);
  if (n==0) SaveImage(cTensor_Generator_Output,"Test/test-current.tga");
  //sprintf(str,"Test/test%03i.txt",n);
  //cTensor_Generator_Output.PrintToFile(str,"Изображение",true);
 }
 counter++;
}
//----------------------------------------------------------------------------------------------------
//сохранить изображение из набора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::SaveKitImage(void)
{
 char str[STRING_BUFFER_SIZE];
 for(size_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/real%03i.tga",n);
  type_t *ptr=&RealImage[RealImageIndex[n]][0];
  size_t size=RealImage[RealImageIndex[n]].size();
  cTensor_Generator_Output.CopyItemToHost(ptr,size);
  SaveImage(cTensor_Generator_Output,str);
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить изображение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::SaveImage(CTensor<type_t> &cTensor_Generator_Output,const std::string &name)
{
 std::vector<uint32_t> image(IMAGE_WIDTH*IMAGE_HEIGHT);
 type_t *ptr=cTensor_Generator_Output.GetColumnPtr(0,0);
 for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
 {
  for(uint32_t x=0;x<IMAGE_WIDTH;x++)
  {
   float r=0;
   float g=0;
   float b=0;

   float ir=0;
   float ig=0;
   float ib=0;

   if (IMAGE_DEPTH==3)
   {
    ir=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*0];
    ig=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*1];
    ib=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*2];

    //восстановление из RGB
    {
     r=ir;
     g=ib;
     b=ir;

     r+=1.0;
     r/=2.0;

     g+=1.0;
     g/=2.0;

     b+=1.0;
     b/=2.0;

     r*=255.0;
     g*=255.0;
     b*=255.0;
    }

    //восстановление из Lab
    {
     float sl=ir;
     float sa=ig;
     float sb=ib;

	 sl+=1.0;
	 sl/=2.0;

	 CColorModel::Lab2RGB(sl,sa,sb,r,g,b);
    }


    if (r<0) r=0;
    if (r>255) r=255;

    if (g<0) g=0;
    if (g>255) g=255;

    if (b<0) b=0;
    if (b>255) b=255;

   }
   if (IMAGE_DEPTH==1)
   {
    double gray=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*0];
    gray+=1.0;
    gray/=2.0;

    if (gray<0) gray=0;
    if (gray>1) gray=1;
    gray*=255.0;

	r=gray;
	g=gray;
	b=gray;
   }
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
 SaveTGA(name.c_str(),IMAGE_WIDTH,IMAGE_HEIGHT,reinterpret_cast<uint8_t*>(&image[0]));
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::Training(void)
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

  ExchangeRealImageIndex();

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

   //обучаем дискриминатор
   double disc_cost=0;

   for(size_t n=0;n<DiscriminatorNet.size();n++) DiscriminatorNet[n]->TrainingResetDeltaWeight();
   TrainingDiscriminatorFake(disc_cost);

   //корректируем веса дискриминатора
   {
    CTimeStamp cTimeStamp("Обновление весов дискриминатора на фальшивых изображениях:");
    for(size_t n=0;n<DiscriminatorNet.size();n++)
    {
     DiscriminatorNet[n]->TrainingUpdateWeight(disc_speed/(static_cast<double>(BATCH_SIZE)),iteration+1);
    // DiscriminatorNet[n]->ClipWeight(-clip,clip);
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

   get_training%=5;

//   if (get_training==0)
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
   }

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
void CModelMain<type_t>::TrainingNet(bool mnist)
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
 //if (LoadRealMNISTImage()==false)
 if (LoadRealImage()==false)
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
void CModelMain<type_t>::TestTrainingGenerator(void)
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
void CModelMain<type_t>::TestTrainingGeneratorNet(bool mnist)
{
 char str[STRING_BUFFER_SIZE];
 SYSTEM::MakeDirectory("Test");
 if (LoadRealImage()==false)
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
//случайное число
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CModelMain<type_t>::GetRandValue(type_t max_value)
{
 return((static_cast<type_t>(rand())*max_value)/static_cast<type_t>(RAND_MAX));
}

//----------------------------------------------------------------------------------------------------
//логарифм с ограничением по размеру
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CModelMain<type_t>::SafeLog(type_t value)
{
 if (value>0) return(log(value));
 SYSTEM::PutMessageToConsole("Error log!");
 return(-100000);
}
//----------------------------------------------------------------------------------------------------
//перекрёстная энтропия
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CModelMain<type_t>::CrossEntropy(type_t y,type_t p)
{
 type_t s=y*SafeLog(p)+(1-y)*SafeLog(1-p);
 return(-s);
}

//----------------------------------------------------------------------------------------------------
//нужно ли выйти из потока
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelMain<type_t>::IsExit(void)
{
 return(false);
}

//----------------------------------------------------------------------------------------------------
//задать необходимость выхода из потока
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::SetExitState(bool state)
{

}

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelMain<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);
 if (CTensorTest<type_t>::Test()==false) throw("Класс тензоров провалил тестирование!");
 //TestTrainingGeneratorNet(true);
 TrainingNet(true);
}

//----------------------------------------------------------------------------------------------------
//тест скорости
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelMain<type_t>::SpeedTest(void)
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
  CTimeStamp cTimeStamp("Скорость прямой свёртки:");
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
  CTimeStamp cTimeStamp("Скорость обратной свёртки:");
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

#endif
