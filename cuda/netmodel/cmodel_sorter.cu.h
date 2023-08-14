#ifndef C_MODEL_SORTER_H
#define C_MODEL_SORTER_H

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
#include "../../common/ccolormodel.h"
#include "../ctimestamp.cu.h"

#include "../../netlayer/cnetlayerlinear.cu.h"
#include "../../netlayer/cnetlayerdropout.cu.h"
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
class CModelSorter
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
  static const size_t STRING_BUFFER_SIZE=1024;///<размер буфера строки
 private:
  //-структуры------------------------------------------------------------------------------------------
  struct SProtectedVariables
  {
   bool OnExit;
  };
  //-переменные-----------------------------------------------------------------------------------------
  SProtectedVariables sProtectedVariables;

  size_t IMAGE_WIDTH; ///<ширина входных изображений
  size_t IMAGE_HEIGHT; ///<высота входных изображений
  size_t IMAGE_DEPTH; ///<глубина входных изображений

  size_t BATCH_AMOUNT;///<количество пакетов
  size_t BATCH_SIZE;///<размер пакета

  size_t GROUP_SIZE;///<количество групп сортировки

  double SPEED;///<скорость обучения

  type_t END_COST;///<значение ошибки, ниже которого обучение прекращается

  std::vector<std::shared_ptr<INetLayer<type_t> > > SorterNet;///<сеть сортировщика

  CTensor<type_t> cTensor_Sorter_Output;//ответ сортировщика
  struct STrainingImage
  {
   CTensor<type_t> Image;///<изображение
   size_t Group;///<группа классификации
  };

  std::vector<STrainingImage> TrainingImage;//образы изображений
  std::vector<size_t> TrainingImageIndex;//индексы изображений в обучающем наборе
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelSorter(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelSorter();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  bool IsExit(void);//нужно ли выйти из потока
  void SetExitState(bool state);//задать необходимость выхода из потока
  void Execute(void);//выполнить
  type_t GetRandValue(type_t max_value);//случайное число
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
  void CreateSorter(void);//создать сеть сортировщика
  bool LoadTrainingImageInPath(const std::string &path,size_t group,bool added,size_t &index);//загрузить образы изображений из каталога
  bool LoadTrainingImage(void);//загрузить образы изображений
  void SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<сохранить слои сети
  void LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<загрузить слои сети
  void LoadNet(void);//загрузить сети
  void SaveNet(void);//сохранить сети
  void TrainingSorter(size_t mini_batch_index,double &cost);//обучение сортировщика
  void ExchangeTrainingImageIndex(void);//перемешать индексы изображений
  void SaveImage(CTensor<type_t> &cTensor_Image,const std::string &name);//сохранить изображение
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
CModelSorter<type_t>::CModelSorter(void)
{
 IMAGE_WIDTH=200;
 IMAGE_HEIGHT=200;
 IMAGE_DEPTH=3;
 BATCH_SIZE=10;

 GROUP_SIZE=10;

 SPEED=0.001;
 END_COST=0.1;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelSorter<type_t>::~CModelSorter()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//создать сеть сортировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::CreateSorter(void)
{
 SorterNet.clear();
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(8,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(256,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 /*SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(256,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(128,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(64,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,SorterNet[SorterNet.size()-1].get())));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(32,NNeuron::NEURON_FUNCTION_LEAKY_RELU,SorterNet[SorterNet.size()-1].get())));
 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,SorterNet[SorterNet.size()-1].get())));
 */
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(GROUP_SIZE,NNeuron::NEURON_FUNCTION_SIGMOID,SorterNet[SorterNet.size()-1].get())));
}


//----------------------------------------------------------------------------------------------------
//загрузить образы изображений из каталога
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelSorter<type_t>::LoadTrainingImageInPath(const std::string &path,size_t group,bool added,size_t &index)
{
 char str[STRING_BUFFER_SIZE];
 SYSTEM::PutMessageToConsole("Загружается:"+path);

 std::vector<std::string> file_list;
 SYSTEM::CreateFileList(path,file_list);
 //обрабатываем файлы
 size_t size=file_list.size();
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
  if (added==false)
  {
   index++;
   //index++;
   delete[](img_ptr);
   continue;
  }

  STrainingImage sTrainingImage_Based;
  STrainingImage sTrainingImage_FlipHorizontal;

  sTrainingImage_Based.Group=group;
  sTrainingImage_Based.Image=CTensor<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
  sTrainingImage_FlipHorizontal.Group=group;
  sTrainingImage_FlipHorizontal.Image=CTensor<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);

  //sprintf(str,"Загружается %i:",index);
  //SYSTEM::PutMessageToConsole(str+name);

  //зеркальное по ширине изображение
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
    gray*=2.0;
    gray-=1.0;

	r*=2.0;
	r-=1.0;

	g*=2.0;
	g-=1.0;

    b*=2.0;
	b-=1.0;

	if (IMAGE_DEPTH==1)
	{
	 sTrainingImage_Based.Image.SetElement(0,y,x,gray);
	 sTrainingImage_FlipHorizontal.Image.SetElement(0,y,IMAGE_WIDTH-x-1,gray);
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

	 sTrainingImage_Based.Image.SetElement(0,y,x,r);
	 sTrainingImage_Based.Image.SetElement(1,y,x,g);
	 sTrainingImage_Based.Image.SetElement(2,y,x,b);

	 sTrainingImage_FlipHorizontal.Image.SetElement(0,y,IMAGE_WIDTH-x-1,r);
	 sTrainingImage_FlipHorizontal.Image.SetElement(1,y,IMAGE_WIDTH-x-1,g);
	 sTrainingImage_FlipHorizontal.Image.SetElement(2,y,IMAGE_WIDTH-x-1,b);
	}
   }
  }
  TrainingImage[index]=sTrainingImage_Based;
  TrainingImageIndex[index]=index;
  index++;
  //TrainingImage[index]=sTrainingImage_FlipHorizontal;
  //TrainingImageIndex[index]=index;
  //index++;
  delete[](img_ptr);
 }
 return(true);
}

//----------------------------------------------------------------------------------------------------
//загрузить образы изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelSorter<type_t>::LoadTrainingImage(void)
{
 char str[STRING_BUFFER_SIZE];
 TrainingImage.clear();
 TrainingImageIndex.clear();
 std::string path="TrainingImage";
 //считаем изображения
 size_t index=0;
 for(size_t n=0;n<GROUP_SIZE;n++)
 {
  sprintf(str,"/%i",n);
  LoadTrainingImageInPath(path+str,n,false,index);
 }
 sprintf(str,"Найдено изображений:%i",index);
 SYSTEM::PutMessageToConsole(str);
 //добавляем изображения
 TrainingImage.resize(index);
 TrainingImageIndex.resize(index);
 index=0;
 for(size_t n=0;n<GROUP_SIZE;n++)
 {
  sprintf(str,"/%i",n);
  LoadTrainingImageInPath(path+str,n,true,index);
 }
 sprintf(str,"Загружено изображений:%i",TrainingImage.size());
 SYSTEM::PutMessageToConsole(str);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!сохранить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->Save(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(size_t n=0;n<net.size();n++) net[n]->Load(iDataStream_Ptr);
}

//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::LoadNet(void)
{
 FILE *file=fopen("sorter_neuronet.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("sorter_neuronet.net",false));
  LoadNetLayers(iDataStream_Disc_Ptr.get(),SorterNet);
  SYSTEM::PutMessageToConsole("Сеть сортировщика загружена.");
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SaveNet(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("sorter_neuronet.net",true));
 SaveNetLayers(iDataStream_Disc_Ptr.get(),SorterNet);
}

//----------------------------------------------------------------------------------------------------
//обучение сортировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::TrainingSorter(size_t mini_batch_index,double &cost)
{
 char str[STRING_BUFFER_SIZE];

 double real_output=0;
 for(size_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");

  //подаём на вход сортировщика изображение
  {
   CTimeStamp cTimeStamp("Задание изображения:");
   SorterNet[0]->SetOutput(TrainingImage[TrainingImageIndex[b+mini_batch_index*BATCH_SIZE]].Image);
  }
  //вычисляем сеть
  {
   CTimeStamp cTimeStamp("Вычисление сортировщика:");
   for(size_t layer=0;layer<SorterNet.size();layer++) SorterNet[layer]->Forward();
  }
  //вычисляем ошибку последнего слоя
  {
   CTimeStamp cTimeStamp("Получение ответа сортировщика:");
   SorterNet[SorterNet.size()-1]->GetOutput(cTensor_Sorter_Output);
  }
  {
   CTimeStamp cTimeStamp("Вычисление ошибки:");
   size_t group=TrainingImage[TrainingImageIndex[b+mini_batch_index*BATCH_SIZE]].Group;
   std::string answer_str;
   for(size_t n=0;n<cTensor_Sorter_Output.GetSizeY();n++)
   {
    type_t necessities=0;
    if (n==group) necessities=1;
	type_t answer=cTensor_Sorter_Output.GetElement(0,n,0);
	type_t error=-(necessities-answer);
	type_t local_cost=error*error;
	if (local_cost>cost) cost=local_cost;
	cTensor_Sorter_Output.SetElement(0,n,0,error);
	sprintf(str,"%.2f[%.2f] ",answer,necessities);
	answer_str+=str;
   }
   //SYSTEM::PutMessageToConsole(answer_str);
  }
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   //задаём ошибку сортировщику
   SorterNet[SorterNet.size()-1]->SetOutputError(cTensor_Sorter_Output);
  }
  //выполняем вычисление весов
  {
   CTimeStamp cTimeStamp("Обучение сортировщика:");
   for(size_t m=0,n=SorterNet.size()-1;m<SorterNet.size();m++,n--) SorterNet[n]->TrainingBackward();
  }
 }
}

//----------------------------------------------------------------------------------------------------
//перемешать индексы изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::ExchangeTrainingImageIndex(void)
{
 //делаем перемешивание
 size_t image_amount=TrainingImage.size();
 for(size_t n=0;n<image_amount;n++)
 {
  size_t index_1=n;
  size_t index_2=static_cast<size_t>((rand()*static_cast<double>(image_amount*10))/static_cast<double>(RAND_MAX));
  index_2%=image_amount;

  size_t tmp=TrainingImageIndex[index_1];
  TrainingImageIndex[index_1]=TrainingImageIndex[index_2];
  TrainingImageIndex[index_2]=tmp;
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить изображение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SaveImage(CTensor<type_t> &cTensor_Image,const std::string &name)
{
 std::vector<uint32_t> image(IMAGE_WIDTH*IMAGE_HEIGHT);
 type_t *ptr=cTensor_Image.GetColumnPtr(0,0);
 for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
 {
  for(uint32_t x=0;x<IMAGE_WIDTH;x++)
  {
   float r=0;
   float g=0;
   float b=0;
   if (IMAGE_DEPTH==3)
   {
    r=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*0];
    g=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*1];
    b=ptr[x+y*IMAGE_WIDTH+(IMAGE_HEIGHT*IMAGE_WIDTH)*2];
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
void CModelSorter<type_t>::Training(void)
{
 char str_b[STRING_BUFFER_SIZE];

 double speed=SPEED;
 size_t max_iteration=1000000000;//максимальное количество итераций обучения
 uint32_t iteration=0;

 size_t image_amount=TrainingImage.size();

 std::string str;
 while(iteration<max_iteration)
 {
  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string((long double)iteration+1));

  if (iteration%250==0)
  {
   SYSTEM::PutMessageToConsole("Save net.");
   SaveNet();
   SYSTEM::PutMessageToConsole("");
  }

  type_t full_cost=0;
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

   long double begin_time=SYSTEM::GetSecondCounter();

   //обучаем сортировщик
   double cost=0;

   for(size_t n=0;n<SorterNet.size();n++) SorterNet[n]->TrainingResetDeltaWeight();
   TrainingSorter(batch,cost);
   //корректируем веса сортировщика
   {
    CTimeStamp cTimeStamp("Обновление весов сортировщика:");
    for(size_t n=0;n<SorterNet.size();n++)
    {
     SorterNet[n]->TrainingUpdateWeight(speed/(static_cast<double>(BATCH_SIZE)));
    }
   }
   if (cost>full_cost) full_cost=cost;
   long double end_time=SYSTEM::GetSecondCounter();
   float cpu_time=static_cast<float>((end_time-begin_time)*1000.0);

   str="Ошибка сортировщика:";
   str+=std::to_string((long double)cost);
   SYSTEM::PutMessageToConsole(str);
   str="Достигнута ошибка сортировщика:";
   str+=std::to_string((long double)full_cost);
   SYSTEM::PutMessageToConsole(str);
   str="Скорость обучения:";
   str+=std::to_string((long double)speed);
   SYSTEM::PutMessageToConsole(str);

   sprintf(str_b,"На минипакет ушло: %.2f мс.",cpu_time);
   SYSTEM::PutMessageToConsole(str_b);
   SYSTEM::PutMessageToConsole("");
  }
  str="Общая ошибка сортировщика для итерации:";
  str+=std::to_string((long double)full_cost);
  SYSTEM::PutMessageToConsole(str);

  FILE *file=fopen("test.csv","ab");
  fprintf(file,"%i;%f\r\n",iteration,full_cost);
  fclose(file);


  if (full_cost<END_COST) break;
  ExchangeTrainingImageIndex();
  iteration++;
 }
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::TrainingNet(void)
{
 char str[STRING_BUFFER_SIZE];
 if (LoadTrainingImage()==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 SYSTEM::PutMessage("Образы изображений загружены.");
 //дополняем набор до кратного размеру пакета
 size_t image_amount=TrainingImage.size();
 BATCH_AMOUNT=image_amount/BATCH_SIZE;
 if (BATCH_AMOUNT==0) BATCH_AMOUNT=1;
 if (image_amount%BATCH_SIZE!=0)
 {
  size_t index=0;
  for(size_t n=image_amount%BATCH_SIZE;n<BATCH_SIZE;n++,index++)
  {
   TrainingImageIndex.push_back(TrainingImageIndex[index%image_amount]);
  }
  image_amount=TrainingImageIndex.size();
  BATCH_AMOUNT=image_amount/BATCH_SIZE;
 }
 sprintf(str,"Изображений:%i Минипакетов:%i",image_amount,BATCH_AMOUNT);
 SYSTEM::PutMessageToConsole(str);

  cTensor_Sorter_Output=CTensor<type_t>(1,GROUP_SIZE,1);

 CreateSorter();
 for(size_t n=0;n<SorterNet.size();n++) SorterNet[n]->Reset();
 LoadNet();

 //включаем обучение
 for(size_t n=0;n<SorterNet.size();n++) SorterNet[n]->TrainingStart();

 Training();
 //отключаем обучение
 for(size_t n=0;n<SorterNet.size();n++) SorterNet[n]->TrainingStop();

 SaveNet();
}


//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//случайное число
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CModelSorter<type_t>::GetRandValue(type_t max_value)
{
 return((static_cast<type_t>(rand())*max_value)/static_cast<type_t>(RAND_MAX));
}

//----------------------------------------------------------------------------------------------------
//нужно ли выйти из потока
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelSorter<type_t>::IsExit(void)
{
 return(false);
}

//----------------------------------------------------------------------------------------------------
//задать необходимость выхода из потока
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SetExitState(bool state)
{

}

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelSorter<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);
 if (CTensorTest<type_t>::Test()==false) throw("Класс тензоров провалил тестирование!");
 TrainingNet();
}

#endif
