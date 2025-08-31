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
#include "../../libjpeg/jpeglib.h"

static const std::string TRAINING_PARAM_FILE_NAME("sorter_training_param.net");
static const std::string NET_FILE_NAME("sorter_neuronet.net");

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
  static const uint32_t STRING_BUFFER_SIZE=1024;///<размер буфера строки
 private:
  //-структуры------------------------------------------------------------------------------------------
  struct SProtectedVariables
  {
   bool OnExit;
  };
  //-переменные-----------------------------------------------------------------------------------------
  SProtectedVariables sProtectedVariables;

  uint32_t IMAGE_WIDTH; ///<ширина входных изображений
  uint32_t IMAGE_HEIGHT; ///<высота входных изображений
  uint32_t IMAGE_DEPTH; ///<глубина входных изображений

  uint32_t BATCH_AMOUNT;///<количество пакетов
  uint32_t BATCH_SIZE;///<размер пакета

  uint32_t GROUP_SIZE;///<количество групп сортировки

  uint32_t ITERATION_OF_SAVE_NET;///<какую итерацию сохранять сеть

  double SPEED;///<скорость обучения

  type_t END_COST;///<значение ошибки, ниже которого обучение прекращается

  std::vector<std::shared_ptr<INetLayer<type_t> > > SorterNet;///<сеть сортировщика

  CTensor<type_t> cTensor_Sorter_Output;///<ответ сортировщика

  struct SImageSettings
  {
   uint32_t StorageImageIndex;///<индекс изображения из хранилища изображений

   bool FlipHorizontal;///<требуется ли отражение по горизонтали (имеет высший приоритет)
   int32_t OffsetX;///<смещение по X
   int32_t OffsetY;///<смещение по Y

   SImageSettings(void)///<конструктор
   {
    StorageImageIndex=0;
    FlipHorizontal=false;
    OffsetX=0;
    OffsetY=0;
   }
  };

  struct SStorageImage
  {
   std::vector<type_t> Image;///<данные изображения
   uint32_t Group;///<группа классификации
  };



  std::vector<SStorageImage> TrainingImageStorage;///<данные изображений для обучения
  std::vector<SImageSettings> TrainingImageSettings;///<образы изображений для обучения
  std::vector<uint32_t> TrainingImageSettingsIndex;///<индексы изображений в обучающем наборе

  std::vector<SStorageImage> CheckImageStorage;///<данные изображений для тестирования точности
  std::vector<SImageSettings> CheckImageSettings;///<образы изображений для тестирования точности

  uint32_t Iteration;//итерация

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelSorter(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelSorter();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  bool IsExit(void);///<нужно ли выйти из потока
  void SetExitState(bool state);///<задать необходимость выхода из потока
  void Execute(void);///<выполнить
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
  void CreateSorter(void);///<создать сеть сортировщика
  bool LoadStorageImageInPath(const std::string &path,std::vector<SStorageImage> &storage_image_list,uint32_t group);///<загрузить образы изображений из каталога
  bool LoadTrainingImage(void);///<загрузить образы изображений
  bool LoadCheckImage(void);///<загрузить образы изображений для тестирования
  bool LoadJPG(const std::string &file_name,std::vector<uint8_t> &image,uint32_t &width,uint32_t &height);///<загрузка jpg-файла
  void SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<сохранить слои сети
  void LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net);///<загрузить слои сети
  void SaveNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t iteration);///<сохранить параметры обучения слоёв сети
  void LoadNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t &iteration);///<загрузить параметры обучения слоёв сети
  void LoadNet(const std::string &file_name);///<загрузить сети
  void SaveNet(const std::string &file_name);///<сохранить сети
  void LoadTrainingParam(const std::string &file_name);///<загрузить параметры обучения
  void SaveTrainingParam(const std::string &file_name);///<сохранить параметры обучения
  void GetImage(const SImageSettings &sImageSettings,const std::vector<SStorageImage> &storage_image,std::vector<type_t> &image);///<получение изображения с учётом его параметров
  void TrainingSorter(uint32_t mini_batch_index,double &cost);///<обучение сортировщика
  void ExchangeTrainingImageSettingsIndex(void);///<перемешать индексы изображений
  void SaveImage(CTensor<type_t> &cTensor_Image,const std::string &name);///<сохранить изображение
  void SaveImage(type_t *image_ptr,const std::string &name,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth);///<сохранить изображение
  void Training(void);///<обучение нейросети
  void TrainingNet(void);///<обучение нейросети

  void Sorting(void);///<выполнить сортировку
  double Check(void);///<выполнить тестирование

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
 IMAGE_WIDTH=224/2;
 IMAGE_HEIGHT=224/2;
 IMAGE_DEPTH=3;
 BATCH_SIZE=100;

 GROUP_SIZE=10;

 SPEED=0.0001;
 END_COST=0.1;

 ITERATION_OF_SAVE_NET=1;

 Iteration=0;
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


 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(96/2,11,4,4,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);
// SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128/2,5,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);

 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(192/2,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);
 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128/2,3,2,2,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);

/* SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,2,2,2,2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,2,2,2,2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);*/

 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 /*SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,2,2,2,2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);
 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,1,1,2,2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);*/
 //SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));


/*
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(512,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 */
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(2048/2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(256,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(GROUP_SIZE,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Sorter Output tensor",false);

/*
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(8,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,0,0,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet[SorterNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(256,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));

 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(GROUP_SIZE,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 SorterNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,SorterNet[SorterNet.size()-1].get(),BATCH_SIZE)));
 */
}


//----------------------------------------------------------------------------------------------------
//загрузить образы изображений из каталога
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelSorter<type_t>::LoadStorageImageInPath(const std::string &path,std::vector<SStorageImage> &storage_image_list,uint32_t group)
{
 char str[STRING_BUFFER_SIZE];
 SYSTEM::PutMessageToConsole("Загружается:"+path);

 std::vector<std::string> file_list;
 SYSTEM::CreateFileList(path,file_list);

 SStorageImage sStorageImage;
 sStorageImage.Group=group;
 sStorageImage.Image=std::vector<type_t>(IMAGE_DEPTH*IMAGE_HEIGHT*IMAGE_WIDTH);

 //обрабатываем файлы
 uint32_t size=file_list.size();
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
//sprintf(str,"Загружается %i:",index);
  //SYSTEM::PutMessageToConsole(str+name);

  double dx=static_cast<double>(width)/static_cast<double>(IMAGE_WIDTH);
  double dy=static_cast<double>(height)/static_cast<double>(IMAGE_HEIGHT);

  for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
  {
   for(uint32_t x=0;x<IMAGE_WIDTH;x++)
   {
    uint32_t xp=x*dx;
    uint32_t yp=y*dy;

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
	 sStorageImage.Image[x+y*IMAGE_WIDTH]=gray;
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

	 sStorageImage.Image[x+y*IMAGE_WIDTH+0*IMAGE_WIDTH*IMAGE_HEIGHT]=r;
	 sStorageImage.Image[x+y*IMAGE_WIDTH+1*IMAGE_WIDTH*IMAGE_HEIGHT]=g;
	 sStorageImage.Image[x+y*IMAGE_WIDTH+2*IMAGE_WIDTH*IMAGE_HEIGHT]=b;
	}
   }
  }
  storage_image_list.push_back(sStorageImage);
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
 TrainingImageStorage.clear();
 TrainingImageStorage.reserve(2500);
 std::string path="TrainingImage";
 //загружаем изображения
 for(uint32_t n=0;n<GROUP_SIZE;n++)
 {
  sprintf(str,"/%i",static_cast<int>(n));
  LoadStorageImageInPath(path+str,TrainingImageStorage,n);
 }
 sprintf(str,"Загружено изображений:%i",static_cast<int>(TrainingImageStorage.size()));
 SYSTEM::PutMessageToConsole(str);
 TrainingImageSettings.clear();
 TrainingImageSettings.reserve(TrainingImageStorage.size()*8);
 for(uint32_t n=0;n<TrainingImageStorage.size();n++)
 {
  SImageSettings sImageSettings;
  sImageSettings.StorageImageIndex=n;
  TrainingImageSettings.push_back(sImageSettings);
 }

 //создаём смещённые и отражённые варианты
 uint32_t size=TrainingImageSettings.size();
 for(uint32_t n=0;n<size;n++)
 {
  SImageSettings sImageSettings_New=TrainingImageSettings[n];
  for(uint32_t fh=0;fh<1;fh++)//отражение по горизонтали
  {
   for(int32_t dx=-32;dx<32;dx+=16)//смещения по горизонтали
   {
    for(int32_t dy=-32;dy<32;dy+=16)//смещения по вертикали
    {
     if (dx==0 && dy==0 && fh==0) continue;

     //отражение по горизонтали
     sImageSettings_New.FlipHorizontal=false;
     if (fh!=0) sImageSettings_New.FlipHorizontal=true;
     sImageSettings_New.OffsetX=dx;
     sImageSettings_New.OffsetY=dy;
     TrainingImageSettings.push_back(sImageSettings_New);
    }
   }
  }
 }

 TrainingImageSettingsIndex=std::vector<uint32_t>(TrainingImageSettings.size());
 for(uint32_t n=0;n<TrainingImageSettings.size();n++) TrainingImageSettingsIndex[n]=n;
 sprintf(str,"Всего изображений:%i",static_cast<int>(TrainingImageSettings.size()));
 SYSTEM::PutMessageToConsole(str);
 return(true);
}

//----------------------------------------------------------------------------------------------------
//загрузить образы изображений для тестирования
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelSorter<type_t>::LoadCheckImage(void)
{
 char str[STRING_BUFFER_SIZE];
 CheckImageStorage.clear();
 CheckImageStorage.reserve(2500);
 std::string path="CheckImage";
 //загружаем изображения
 for(uint32_t n=0;n<GROUP_SIZE;n++)
 {
  sprintf(str,"/%i",static_cast<int>(n));
  LoadStorageImageInPath(path+str,CheckImageStorage,n);
 }
 sprintf(str,"Загружено изображений для тестирования:%i",static_cast<int>(CheckImageStorage.size()));
 SYSTEM::PutMessageToConsole(str);
 CheckImageSettings.clear();
 CheckImageSettings.reserve(CheckImageStorage.size()*8);
 for(uint32_t n=0;n<CheckImageStorage.size();n++)
 {
  SImageSettings sImageSettings;
  sImageSettings.StorageImageIndex=n;
  CheckImageSettings.push_back(sImageSettings);
 }

 //создаём смещённые и отражённые варианты
 uint32_t size=CheckImageSettings.size();
 for(uint32_t n=0;n<size;n++)
 {
  SImageSettings sImageSettings_New=CheckImageSettings[n];
  for(uint32_t fh=0;fh<1;fh++)//отражение по горизонтали
  {
   for(int32_t dx=-32;dx<32;dx+=16)//смещения по горизонтали
   {
    for(int32_t dy=-32;dy<32;dy+=16)//смещения по вертикали
    {
     if (dx==0 && dy==0 && fh==0) continue;

     //отражение по горизонтали
     sImageSettings_New.FlipHorizontal=false;
     if (fh!=0) sImageSettings_New.FlipHorizontal=true;
     sImageSettings_New.OffsetX=dx;
     sImageSettings_New.OffsetY=dy;
     CheckImageSettings.push_back(sImageSettings_New);
    }
   }
  }
 }

 sprintf(str,"Всего изображений для тестирования:%i",static_cast<int>(CheckImageSettings.size()));
 SYSTEM::PutMessageToConsole(str);

 return(true);
}


//----------------------------------------------------------------------------------------------------
//!загрузка jpg-файла
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelSorter<type_t>::LoadJPG(const std::string &file_name,std::vector<uint8_t> &image,uint32_t &width,uint32_t &height)
{
 image.clear();
 FILE *file=fopen(file_name.c_str(),"rb");
 if (file==NULL) return(false);
 std::vector<uint8_t> jpg_data;
 while(1)
 {
  uint8_t b;
  if (fread(&b,sizeof(uint8_t),1,file)==0) break;
  jpg_data.push_back(b);
 }
 fclose(file);

 uint32_t jpg_size=jpg_data.size();
 unsigned char *jpg_buffer=&jpg_data[0];
 struct jpeg_decompress_struct cinfo;
 struct jpeg_error_mgr jerr;

 int32_t row_stride;
 int32_t pixel_size;
 cinfo.err=jpeg_std_error(&jerr);
 jpeg_create_decompress(&cinfo);
 jpeg_mem_src(&cinfo,jpg_buffer,jpg_size);
 int32_t rc=jpeg_read_header(&cinfo,TRUE);
 if (rc!=1) return(false);//данные не являются нормальным jpeg-изображением
 jpeg_start_decompress(&cinfo);
 width=cinfo.output_width;
 height=cinfo.output_height;
 pixel_size=cinfo.output_components;
 row_stride=width*pixel_size;
 uint32_t bmp_size=width*height*pixel_size;
 image.resize(bmp_size);
 //читаем изображение
 while (cinfo.output_scanline < cinfo.output_height)
 {
  unsigned char *buffer_array[1];
  buffer_array[0]=reinterpret_cast<unsigned char*>(&image[0])+(cinfo.output_scanline)*row_stride;
  jpeg_read_scanlines(&cinfo,buffer_array,1);
 }
 jpeg_finish_decompress(&cinfo);
 jpeg_destroy_decompress(&cinfo);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!сохранить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(uint32_t n=0;n<net.size();n++) net[n]->Save(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net)
{
 for(uint32_t n=0;n<net.size();n++) net[n]->Load(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоёв сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SaveNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t iteration)
{
 for(uint32_t n=0;n<net.size();n++) net[n]->SaveTrainingParam(iDataStream_Ptr);
 iDataStream_Ptr->SaveUInt32(iteration);

}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоёв сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::LoadNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t &iteration)
{
 for(uint32_t n=0;n<net.size();n++) net[n]->LoadTrainingParam(iDataStream_Ptr);
 iteration=iDataStream_Ptr->LoadUInt32();
}
//----------------------------------------------------------------------------------------------------
//загрузить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::LoadTrainingParam(const std::string &file_name)
{
 FILE *file=fopen(file_name.c_str(),"rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile(file_name,false));
  LoadNetLayersTrainingParam(iDataStream_Ptr.get(),SorterNet,Iteration);
  SYSTEM::PutMessageToConsole("Параметры обучения загружены.");
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SaveTrainingParam(const std::string &file_name)
{
 std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile(file_name,true));
 SaveNetLayersTrainingParam(iDataStream_Ptr.get(),SorterNet,Iteration);
}

//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::LoadNet(const std::string &file_name)
{
 FILE *file=fopen(file_name.c_str(),"rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile(file_name,false));
  LoadNetLayers(iDataStream_Disc_Ptr.get(),SorterNet);
  SYSTEM::PutMessageToConsole("Сеть сортировщика загружена.");
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SaveNet(const std::string &file_name)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile(file_name.c_str(),true));
 SaveNetLayers(iDataStream_Disc_Ptr.get(),SorterNet);
}

//----------------------------------------------------------------------------------------------------
//получение изображения с учётом его параметров
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::GetImage(const SImageSettings &sImageSettings,const std::vector<SStorageImage> &storage_image,std::vector<type_t> &image)
{
 image.resize(IMAGE_DEPTH*IMAGE_HEIGHT*IMAGE_WIDTH);
 const type_t *input_ptr=&storage_image[sImageSettings.StorageImageIndex].Image[0];
 type_t *output_ptr=&image[0];
 for(int32_t z=0;z<IMAGE_DEPTH;z++)
 {
  for(int32_t y=0;y<IMAGE_HEIGHT;y++)
  {
   int32_t iy=y;
   iy+=sImageSettings.OffsetY;
   while (iy<0) iy+=IMAGE_HEIGHT;
   iy%=IMAGE_HEIGHT;

   uint32_t in_offset=(iy*IMAGE_WIDTH+z*IMAGE_WIDTH*IMAGE_HEIGHT);
   for(int32_t x=0;x<IMAGE_WIDTH;x++,output_ptr++)
   {
    int32_t ix=x;
    if (sImageSettings.FlipHorizontal==true) ix=(IMAGE_WIDTH-ix-1);
    ix+=sImageSettings.OffsetX;
    while (ix<0) ix+=IMAGE_WIDTH;
    ix%=IMAGE_WIDTH;

    uint32_t in_offset_l=ix+in_offset;

    type_t value=input_ptr[in_offset_l];
    *output_ptr=value;
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//обучение сортировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::TrainingSorter(uint32_t mini_batch_index,double &cost)
{
 static std::vector<type_t> image=std::vector<type_t>(IMAGE_DEPTH*IMAGE_HEIGHT*IMAGE_WIDTH);

 char str[STRING_BUFFER_SIZE];

 double local_cost=0;
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  uint32_t index=b+mini_batch_index*BATCH_SIZE;
  uint32_t image_index=TrainingImageSettingsIndex[index];
  SImageSettings &sImageSettings=TrainingImageSettings[image_index];
  uint32_t group=TrainingImageStorage[sImageSettings.StorageImageIndex].Group;
  //подаём на вход сортировщика изображение
  {
   CTimeStamp cTimeStamp("Задание изображения:");

   GetImage(sImageSettings,TrainingImageStorage,image);
   SorterNet[0]->GetOutputTensor(b).CopyItemToDevice(&image[0],image.size());
  }
 }
 //вычисляем сеть
 {
  CTimeStamp cTimeStamp("Вычисление сортировщика:");
  for(uint32_t layer=0;layer<SorterNet.size();layer++) SorterNet[layer]->Forward();
 }
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  uint32_t index=b+mini_batch_index*BATCH_SIZE;
  uint32_t image_index=TrainingImageSettingsIndex[index];
  SImageSettings &sImageSettings=TrainingImageSettings[image_index];
  uint32_t group=TrainingImageStorage[sImageSettings.StorageImageIndex].Group;

  //вычисляем ошибку последнего слоя
  {
   CTimeStamp cTimeStamp("Получение ответа сортировщика:");
   SorterNet[SorterNet.size()-1]->GetOutput(b,cTensor_Sorter_Output);
  }
  {
   CTimeStamp cTimeStamp("Вычисление ошибки:");
   //std::string answer_str;
   for(uint32_t n=0;n<cTensor_Sorter_Output.GetSizeY();n++)
   {
    type_t necessities=0;
    if (n==group) necessities=1;
    type_t answer=cTensor_Sorter_Output.GetElement(0,n,0);
	type_t error=(answer-necessities);
	cTensor_Sorter_Output.SetElement(0,n,0,error);
	error=fabs(error);
	if (local_cost<error) local_cost=error;
	//sprintf(str,"%.2f[%.2f] ",answer,necessities);
	//answer_str+=str;
   }
   //SYSTEM::PutMessageToConsole(answer_str);
  }
  {
   CTimeStamp cTimeStamp("Задание ошибки:");
   //задаём ошибку сортировщику
   SorterNet[SorterNet.size()-1]->SetOutputError(b,cTensor_Sorter_Output);
  }
 }
 //выполняем вычисление весов
 {
  CTimeStamp cTimeStamp("Обучение сортировщика:");
  for(uint32_t m=0,n=SorterNet.size()-1;m<SorterNet.size();m++,n--) SorterNet[n]->TrainingBackward();
 }
 if (local_cost>cost) cost=local_cost;
}

//----------------------------------------------------------------------------------------------------
//перемешать индексы изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::ExchangeTrainingImageSettingsIndex(void)
{
 //делаем перемешивание
 uint32_t image_amount=TrainingImageSettings.size();
 for(uint32_t n=0;n<image_amount;n++)
 {
  uint32_t index_1=n;
  uint32_t index_2=static_cast<uint32_t>((rand()*static_cast<double>(image_amount*10))/static_cast<double>(RAND_MAX));
  index_2%=image_amount;

  uint32_t tmp=TrainingImageSettingsIndex[index_1];
  TrainingImageSettingsIndex[index_1]=TrainingImageSettingsIndex[index_2];
  TrainingImageSettingsIndex[index_2]=tmp;
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить изображение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SaveImage(CTensor<type_t> &cTensor_Image,const std::string &name)
{
 type_t *ptr=cTensor_Image.GetColumnPtr(0,0);
 SaveImage(ptr,name,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
/*
 std::vector<uint32_t> image(IMAGE_WIDTH*IMAGE_HEIGHT);

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
 */
}

//----------------------------------------------------------------------------------------------------
//сохранить изображение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::SaveImage(type_t *image_ptr,const std::string &name,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth)
{
 std::vector<uint32_t> image(output_image_width*output_image_height);
 type_t *ptr=image_ptr;
 for(uint32_t y=0;y<output_image_height;y++)
 {
  for(uint32_t x=0;x<output_image_width;x++)
  {
   float r=0;
   float g=0;
   float b=0;

   float ir=0;
   float ig=0;
   float ib=0;

   if (output_image_depth==3)
   {
    ir=ptr[x+y*output_image_width+(output_image_height*output_image_width)*0];
    ig=ptr[x+y*output_image_width+(output_image_height*output_image_width)*1];
    ib=ptr[x+y*output_image_width+(output_image_height*output_image_width)*2];

    //восстановление из RGB
    {
     r=ir;
     g=ig;
     b=ib;

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
   if (output_image_depth==1)
   {
    double gray=ptr[x+y*output_image_width+(output_image_height*output_image_width)*0];
    gray+=1.0;
    gray/=2.0;

    if (gray<0) gray=0;
    if (gray>1) gray=1;
    gray*=255.0;

	r=gray;
	g=gray;
	b=gray;
   }
   uint32_t offset=x+y*output_image_width;
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
 SaveTGA(name.c_str(),output_image_width,output_image_height,reinterpret_cast<uint8_t*>(&image[0]));
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::Training(void)
{
 char str_b[STRING_BUFFER_SIZE];

 double speed=SPEED;
 uint32_t max_iteration=1000000000;//максимальное количество итераций обучения

 uint32_t image_amount=TrainingImageSettings.size();

 std::string str;

 double max_acc=0;//максимальная достигнутая точность

 while(Iteration<max_iteration)
 {
  ExchangeTrainingImageSettingsIndex();

  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string(static_cast<long double>(Iteration+1)));

  //проверяем точность сети
  double acc=Check();
  if (acc>max_acc)
  {
   max_acc=acc;
   SYSTEM::PutMessageToConsole("Save max accuracy net.");
   str=std::to_string(static_cast<long double>(acc));
   SaveNet("["+str+"] "+NET_FILE_NAME);
   SaveTrainingParam("["+str+"] "+TRAINING_PARAM_FILE_NAME);
   SYSTEM::PutMessageToConsole("");
  }

  str="Точность:";
  str+=std::to_string(static_cast<long double>(acc));
  str+=" Максимальная точность:";
  str+=std::to_string(static_cast<long double>(max_acc));
  SYSTEM::PutMessageToConsole(str);

  if (Iteration%ITERATION_OF_SAVE_NET==0)
  {
   SYSTEM::PutMessageToConsole("Save net.");
   SaveNet(NET_FILE_NAME);
   SaveTrainingParam(TRAINING_PARAM_FILE_NAME);
   SYSTEM::PutMessageToConsole("");
  }

  type_t full_cost=0;
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

   long double begin_time=SYSTEM::GetSecondCounter();

   //обучаем сортировщик
   double cost=0;
   for(uint32_t n=0;n<SorterNet.size();n++) SorterNet[n]->TrainingResetDeltaWeight();
   TrainingSorter(batch,cost);
   //корректируем веса сортировщика
   {
    CTimeStamp cTimeStamp("Обновление весов сортировщика:");
    for(uint32_t n=0;n<SorterNet.size();n++)
    {
     SorterNet[n]->TrainingUpdateWeight(speed,Iteration+1);
    }
   }
   if (cost>full_cost) full_cost=cost;
   long double end_time=SYSTEM::GetSecondCounter();
   float cpu_time=static_cast<float>((end_time-begin_time)*1000.0);

   str="Ошибка сортировщика:";
   str+=std::to_string(static_cast<long double>(cost));
   SYSTEM::PutMessageToConsole(str);
   str="Достигнута ошибка сортировщика:";
   str+=std::to_string(static_cast<long double>(full_cost));
   SYSTEM::PutMessageToConsole(str);
   str="Скорость обучения:";
   str+=std::to_string(static_cast<long double>(speed));
   SYSTEM::PutMessageToConsole(str);

   sprintf(str_b,"На минипакет ушло: %.2f мс.",cpu_time);
   SYSTEM::PutMessageToConsole(str_b);
   SYSTEM::PutMessageToConsole("");
  }
  str="Общая ошибка сортировщика для итерации:";
  str+=std::to_string((long double)full_cost);
  SYSTEM::PutMessageToConsole(str);

  FILE *file=fopen("test.csv","ab");
  fprintf(file,"%i;%f\r\n",static_cast<int>(Iteration),full_cost);
  fclose(file);


  if (full_cost<END_COST) break;
  Iteration++;
 }
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::TrainingNet(void)
{
 char str[STRING_BUFFER_SIZE];

 cTensor_Sorter_Output=CTensor<type_t>(1,GROUP_SIZE,1);
 CreateSorter();
 for(uint32_t n=0;n<SorterNet.size();n++) SorterNet[n]->Reset();
 //включаем обучение
 for(uint32_t n=0;n<SorterNet.size();n++)
 {
  SorterNet[n]->TrainingModeAdam(0.5,0.9);
  SorterNet[n]->TrainingStart();
 }
 LoadNet(NET_FILE_NAME);
 LoadTrainingParam(TRAINING_PARAM_FILE_NAME);

 //загружаем изображения для тестирования
 if (LoadCheckImage()==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений для тестирования!");
  return;
 }

 //загружаем обучающие изображения
 if (LoadTrainingImage()==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 SYSTEM::PutMessage("Образы изображений загружены.");
 //дополняем набор до кратного размеру пакета
 uint32_t image_amount=TrainingImageSettings.size();
 BATCH_AMOUNT=image_amount/BATCH_SIZE;
 if (BATCH_AMOUNT==0) BATCH_AMOUNT=1;
 if (image_amount%BATCH_SIZE!=0)
 {
  uint32_t index=0;
  for(uint32_t n=image_amount%BATCH_SIZE;n<BATCH_SIZE;n++,index++)
  {
   TrainingImageSettingsIndex.push_back(TrainingImageSettingsIndex[index%image_amount]);
  }
  image_amount=TrainingImageSettingsIndex.size();
  BATCH_AMOUNT=image_amount/BATCH_SIZE;
 }
 sprintf(str,"Изображений:%i Минипакетов:%i",static_cast<int>(image_amount),static_cast<int>(BATCH_AMOUNT));
 SYSTEM::PutMessageToConsole(str);

 /*
 static std::vector<type_t> image=std::vector<type_t>(IMAGE_DEPTH*IMAGE_HEIGHT*IMAGE_WIDTH);
 SImageSettings &sImageSettings=TrainingImageSettings[0];
 GetImage(sImageSettings,image);
 SaveImage(&image[0],"./Check/test.tga",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 return;
 */

 //начинаем обучение
 Training();
 //отключаем обучение
 for(uint32_t n=0;n<SorterNet.size();n++) SorterNet[n]->TrainingStop();

 SaveNet(NET_FILE_NAME);
 SaveTrainingParam(TRAINING_PARAM_FILE_NAME);
}

//----------------------------------------------------------------------------------------------------
//!выполнить сортировку
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSorter<type_t>::Sorting(void)
{
 char str[STRING_BUFFER_SIZE];

 cTensor_Sorter_Output=CTensor<type_t>(1,GROUP_SIZE,1);
 CreateSorter();
 for(uint32_t n=0;n<SorterNet.size();n++)
 {
  SorterNet[n]->Reset();
 }
 LoadNet(NET_FILE_NAME);

 SYSTEM::MakeDirectory("./Output");
 for(uint32_t n=0;n<GROUP_SIZE;n++)
 {
  char path[STRING_BUFFER_SIZE];
  sprintf(path,"./Output/%i",static_cast<int>(n));
  SYSTEM::MakeDirectory(path);
 }

 //выполняем сортировку
 std::string path="Input";
 std::vector<std::string> file_list;
 SYSTEM::CreateFileList(path,file_list);
 //обрабатываем файлы
 uint32_t size=file_list.size();
 for(uint32_t n=0;n<size;n++)
 {
  std::string &file_name=file_list[n];
  //проверяем расширение
  uint32_t length=file_name.length();
  if (length<4) continue;
  if (file_name[length-4]!='.') continue;
  if (file_name[length-3]!='j' && file_name[length-3]!='J')  continue;
  if (file_name[length-2]!='p' && file_name[length-2]!='P') continue;
  if ((file_name[length-1]!='g' && file_name[length-1]!='G')) continue;
  //отправляем файл на обработку
  uint32_t height;
  uint32_t width;
  std::string name=path+"/"+file_name;
  std::vector<uint8_t> image;
  if (LoadJPG(name.c_str(),image,width,height)==false) continue;

  uint8_t *img_ptr=reinterpret_cast<uint8_t*>(&image[0]);

  CTensor<type_t> Image=CTensor<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);


  double dx=static_cast<double>(width)/static_cast<double>(IMAGE_WIDTH);
  double dy=static_cast<double>(height)/static_cast<double>(IMAGE_HEIGHT);

  for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
  {
   for(uint32_t x=0;x<IMAGE_WIDTH;x++)
   {
    uint32_t xp=x*dx;
    uint32_t yp=y*dy;

    uint32_t offset=(xp+yp*width);
    float r=img_ptr[offset*3+0];
    float g=img_ptr[offset*3+1];
    float b=img_ptr[offset*3+2];

    //float r=(color>>0)&0xff;
    //float g=(color>>8)&0xff;
    //float b=(color>>16)&0xff;

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
	 Image.SetElement(0,y,x,gray);
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

	 Image.SetElement(0,y,x,r);
	 Image.SetElement(1,y,x,g);
	 Image.SetElement(2,y,x,b);
	}
   }
  }
  //подаём изображение на вход сети
  SorterNet[0]->SetOutput(0,Image);
  //вычисляем сеть
  for(uint32_t layer=0;layer<SorterNet.size();layer++) SorterNet[layer]->Forward();
  //вычисляем ошибку последнего слоя
  SorterNet[SorterNet.size()-1]->GetOutput(0,cTensor_Sorter_Output);
  uint32_t min_output=0;
  type_t min_answer=fabs(1-cTensor_Sorter_Output.GetElement(0,0,0));
  for(uint32_t n=0;n<cTensor_Sorter_Output.GetSizeY();n++)
  {
   type_t answer=fabs(1-cTensor_Sorter_Output.GetElement(0,n,0));
   if (answer<min_answer)
   {
    min_answer=answer;
    min_output=n;
   }
  }

  sprintf(str,"- > класс: %i Ответ:%.3f",static_cast<int>(min_output),min_answer);
  SYSTEM::PutMessageToConsole("Изображение:"+name+str);

  sprintf(str,"./Output/%i/%s",static_cast<int>(min_output),file_name.c_str());
  SYSTEM::MoveFileTo(name,str);
 }
}


//----------------------------------------------------------------------------------------------------
//!выполнить тестирование
//----------------------------------------------------------------------------------------------------
template<class type_t>
double CModelSorter<type_t>::Check(void)
{
 static const double TO_PERCENT=100;//перевод в проценты
 static std::vector<type_t> image=std::vector<type_t>(IMAGE_DEPTH*IMAGE_HEIGHT*IMAGE_WIDTH);
 uint32_t ok=0;//количество верных ответов сети
 uint32_t image_amount=CheckImageSettings.size();
 uint32_t batch=image_amount/BATCH_SIZE;
 if (batch*BATCH_SIZE<CheckImageSettings.size()) batch++;

 uint32_t index=0;
 for(uint32_t n=0;n<batch;n++)
 {
  if (IsExit()==true) throw("Стоп");
  //подаём на вход сортировщика изображение
  uint32_t check_index=index;
  for(uint32_t b=0;b<BATCH_SIZE;b++,index++)
  {
   SImageSettings &sImageSettings=CheckImageSettings[index%image_amount];
   uint32_t group=CheckImageStorage[sImageSettings.StorageImageIndex].Group;
   GetImage(sImageSettings,CheckImageStorage,image);
   SorterNet[0]->GetOutputTensor(b).CopyItemToDevice(&image[0],image.size());
  }
  //вычисляем сеть
  for(uint32_t layer=0;layer<SorterNet.size();layer++) SorterNet[layer]->Forward();
  //получаем результат
  for(uint32_t b=0;b<BATCH_SIZE;b++,check_index++)
  {
   if (check_index>=image_amount) break;
   SImageSettings &sImageSettings=CheckImageSettings[check_index%image_amount];
   uint32_t group=CheckImageStorage[sImageSettings.StorageImageIndex].Group;

   SorterNet[SorterNet.size()-1]->GetOutput(b,cTensor_Sorter_Output);
   //определяем, куда сеть классифицирует этот результат
   uint32_t answer_group=0;
   type_t max_answer=cTensor_Sorter_Output.GetElement(0,0,0);
   for(uint32_t m=0;m<cTensor_Sorter_Output.GetSizeY();m++)
   {
    type_t answer=cTensor_Sorter_Output.GetElement(0,m,0);
    if (answer>max_answer)
    {
     answer_group=m;
     max_answer=answer;
    }
   }
   if (answer_group==group) ok++;
  }
 }
 double acc=ok*TO_PERCENT;
 acc/=static_cast<double>(image_amount);
 return(acc);
}



//****************************************************************************************************
//открытые функции
//****************************************************************************************************

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
 //Sorting();
 TrainingNet();
}

#endif
