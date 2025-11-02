#ifndef C_MODEL_MAIN_H
#define C_MODEL_MAIN_H

//****************************************************************************************************
//Класс-основа для моделей сетей
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
#include "../../common/cimage.h"
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
#include "../../netlayer/cnetlayertimeembedding.cu.h"
#include "../../netlayer/cnetlayersplitter.cu.h"
#include "../../netlayer/cnetlayerconcatenator.cu.h"
#include "../../netlayer/cnetlayervaecoderoutput.cu.h"

#include "../tensor.cu.h"

#include "../../common/crandom.h"

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
  static const uint32_t STRING_BUFFER_SIZE=1024;///<размер буфера строки
  static const uint32_t CUDA_PAUSE_MS=1;///<пауза для CUDA
 protected:
  //-структуры------------------------------------------------------------------------------------------
  struct SProtectedVariables
  {
   bool OnExit;
  };
  //-переменные-----------------------------------------------------------------------------------------
  SProtectedVariables sProtectedVariables;
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelMain(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelMain();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  type_t SafeLog(type_t value);///<логарифм с ограничением по размеру
  type_t CrossEntropy(type_t y,type_t p);///<перекрёстная энтропия
  bool IsExit(void);///<нужно ли выйти из потока
  void SetExitState(bool state);///<задать необходимость выхода из потока
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  bool LoadMNISTImage(const std::string &file_name,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector< std::vector<type_t> > &image,std::vector<uint32_t> &index);///<загрузить образы изображений из MNIST
  bool LoadImage(const std::string &path_name,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector< std::vector<type_t> > &image,std::vector<uint32_t> &index);///<загрузить образы изображений
  bool CreateResamplingImage(uint32_t input_image_width,uint32_t input_image_height,uint32_t input_image_depth,const std::vector< std::vector<type_t> > &image_input,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector< std::vector<type_t> > &image_output);///<создать изображения другого разрешения
  void SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,bool backward=false);///<сохранить слои сети
  void LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,bool backward=false);///<загрузить слои сети
  void SaveNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t iteration,bool backward=false);///<сохранить параметры обучения слоёв сети
  void LoadNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t &iteration,bool backward=false);///<загрузить параметры обучения слоёв сети
  void ExchangeImageIndex(std::vector<uint32_t> &index);///<перемешать индексы изображений
  void SaveImage(CTensor<type_t> &cTensor,const std::string &name,uint32_t w,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth);///<сохранить изображение
  void SpeedTest(void);///<тест скорости
};

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelMain<type_t>::CModelMain(void)
{
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
bool CModelMain<type_t>::LoadMNISTImage(const std::string &file_name,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector< std::vector<type_t> > &image,std::vector<uint32_t> &index)
{
 image.clear();
 index.clear();

 static const uint32_t MNIST_IMAGE_AMOUNT=60000;///<количество обучающих изображений
 static const uint32_t MNIST_IMAGE_WIDTH=28; ///<ширина входных изображений
 static const uint32_t MNIST_IMAGE_HEIGHT=28; ///<высота входных изображений

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

 FILE *file=fopen(file_name.c_str(),"rb");
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

 double dx=static_cast<double>(sHeader.Width)/static_cast<double>(output_image_width);
 double dy=static_cast<double>(sHeader.Height)/static_cast<double>(output_image_height);

 uint32_t amount=MNIST_IMAGE_AMOUNT;
 amount/=5;
 for(uint32_t n=0;n<amount;n++)
 {
  image.push_back(std::vector<type_t>(output_image_width*output_image_height*output_image_depth));

  index.push_back(n);
  SImage sImage;
  if (fread(&sImage,sizeof(SImage),1,file)<1) continue;
  uint32_t index=0;

  for(uint32_t y=0;y<output_image_height;y++)
  {
   for(uint32_t x=0;x<output_image_width;x++,index++)
   {
    uint32_t xp=x*dx;
    uint32_t yp=y*dy;

    uint32_t offset=(xp+yp*sHeader.Width);
    float gray=sImage.Color[offset];
    gray/=255.0;
    //приведём к диапазону [-1,1]
    gray*=2.0;
    gray-=1.0;

    float r=gray;
    float g=gray;
    float b=gray;

    if (output_image_depth==1)
	{
	 image[x+y*output_image_width]=gray;
	}
    if (output_image_depth==3)
	{
	 image[x+y*output_image_width+0*output_image_width*output_image_height]=r;
	 image[x+y*output_image_width+1*output_image_width*output_image_height]=g;
	 image[x+y*output_image_width+2*output_image_width*output_image_height]=b;
	}
   }
  }
 }
 SYSTEM::PutMessageToConsole("Образы MNIST загружены успешно.");
 return(true);
}

//----------------------------------------------------------------------------------------------------
//загрузить образы изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelMain<type_t>::LoadImage(const std::string &path_name,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector< std::vector<type_t> > &image,std::vector<uint32_t> &index)
{
 image.clear();
 index.clear();

 std::string path=path_name;

 std::vector<std::string> file_list;
 SYSTEM::CreateFileList(path,file_list);
 //обрабатываем файлы
 uint32_t current_index=0;
 uint32_t size=file_list.size();
 image.reserve(size);
 index.reserve(size);

 for(uint32_t n=0;n<size;n++)
 {
  std::string &file_name=file_list[n];
  //проверяем расширение
  uint32_t length=file_name.length();
  if (length<4) continue;
  if (file_name[length-4]!='.') continue;
  if (file_name[length-3]!='t' && file_name[length-3]!='T')  continue;
  if (file_name[length-2]!='g' && file_name[length-2]!='G') continue;
  if ((file_name[length-1]!='a' && file_name[length-1]!='A') && (file_name[length-1]!='i' && file_name[length-1]!='I')) continue;//для переименованных в tgi файлов
  //отправляем файл на обработку
  std::string name=path+"/"+file_name;
  std::vector<type_t> NormalImage;
  if (CImage<type_t>::LoadImage(name,output_image_width,output_image_height,output_image_depth,NormalImage)==false) continue;
  image.push_back(NormalImage);
  index.push_back(current_index);
  current_index++;
 }
 char str[STRING_BUFFER_SIZE];
 sprintf(str,"Загружено реальных изображений:%i",static_cast<int>(current_index));
 SYSTEM::PutMessageToConsole(str);
 return(true);
}

//----------------------------------------------------------------------------------------------------
//создать изображения другого разрешения
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelMain<type_t>::CreateResamplingImage(uint32_t input_image_width,uint32_t input_image_height,uint32_t input_image_depth,const std::vector< std::vector<type_t> > &image_input,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector< std::vector<type_t> > &image_output)
{
 image_output.clear();
 uint32_t size=image_input.size();
 for(uint32_t n=0;n<size;n++)
 {
  std::vector<type_t> image;
  CImage<type_t>::CreateResamplingImage(input_image_width,input_image_height,input_image_depth,image_input[n],output_image_width,output_image_height,output_image_depth,image);
  image_output.push_back(image);
 }
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!сохранить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,bool backward)
{
 if (backward==false) for(uint32_t n=0;n<net.size();n++) net[n]->Save(iDataStream_Ptr);
                 else for(uint32_t n=net.size();n>0;n--) net[n-1]->Save(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,bool backward)
{
 if (backward==false)
 {
  for(uint32_t n=0;n<net.size();n++)
  {
   if (net[n]->IsMark()==true) continue;
   net[n]->Load(iDataStream_Ptr,true);
  }
 }
 else
 {
  for(uint32_t n=net.size();n>0;n--)
  {
   if (net[n-1]->IsMark()==true) continue;
   net[n-1]->Load(iDataStream_Ptr,true);
  }
 }
}

//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоёв сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::SaveNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t iteration,bool backward)
{
 iDataStream_Ptr->SaveUInt32(iteration);
 if (backward==false) for(uint32_t n=0;n<net.size();n++) net[n]->SaveTrainingParam(iDataStream_Ptr);
                 else for(uint32_t n=net.size();n>0;n--) net[n-1]->SaveTrainingParam(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоёв сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::LoadNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t &iteration,bool backward)
{
 iteration=iDataStream_Ptr->LoadUInt32();
 if (backward==false)
 {
  for(uint32_t n=0;n<net.size();n++)
  {
   if (net[n]->IsMark()==true) continue;
   net[n]->LoadTrainingParam(iDataStream_Ptr);
  }
 }
 else
 {
  for(uint32_t n=net.size();n>0;n--)
  {
   if (net[n-1]->IsMark()==true) continue;
   net[n-1]->LoadTrainingParam(iDataStream_Ptr);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//перемешать индексы изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::ExchangeImageIndex(std::vector<uint32_t> &index)
{
 //делаем перемешивание
 uint32_t image_amount=index.size();
 for(uint32_t n=0;n<image_amount;n++)
 {
  uint32_t index_1=n;
  uint32_t index_2=static_cast<uint32_t>((rand()*static_cast<double>(image_amount*10))/static_cast<double>(RAND_MAX));
  index_2%=image_amount;

  uint32_t tmp=index[index_1];
  index[index_1]=index[index_2];
  index[index_2]=tmp;
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить изображение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelMain<type_t>::SaveImage(CTensor<type_t> &cTensor,const std::string &name,uint32_t w,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth)
{
 CImage<type_t>::SaveImage(cTensor,name,w,output_image_width,output_image_height,output_image_depth);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//логарифм с ограничением по размеру
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CModelMain<type_t>::SafeLog(type_t value)
{
 if (value>0) return(log(value));
 SYSTEM::PutMessageToConsole("Error log!");
 return(-10);
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

 for(uint32_t n=0;n<128;n++)
 {
  cTensor_Kernel_Test.push_back(cTensor_KernelA);
  cTensor_Kernel_Test.push_back(cTensor_KernelB);
 }
 //создаём вектор смещений
 std::vector<type_t> bias;
 std::vector<type_t> bias_test;
 bias.push_back(0);
 bias.push_back(0);

 for(uint32_t n=0;n<128;n++)
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
   for(uint32_t n=0;n<1;n++)
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
   for(uint32_t n=0;n<1;n++)
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
