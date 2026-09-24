#ifndef C_NET_LAYER_TIME_EMBEDDING_MLP_H
#define C_NET_LAYER_TIME_EMBEDDING_MLP_H

//****************************************************************************************************
//\file Слой времени с MLP
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdio.h>
#include <fstream>
#include <vector>
#include <math.h>

#include "../common/idatastream.h"
#include "inetlayer.cu.h"
#include "../cuda/tensor.cu.h"
#include "neuron.cu.h"

//****************************************************************************************************
//макроопределения
//****************************************************************************************************

//****************************************************************************************************
//константы
//****************************************************************************************************

//****************************************************************************************************
//предварительные объявления
//****************************************************************************************************

//****************************************************************************************************
//!Слой времени с MLP
//****************************************************************************************************
template<class type_t>
class CNetLayerTimeEmbeddingMLP:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  uint32_t BatchSize;///<размер пакета для обучения

  CTensor<type_t> cTensor_H;///<выходной тензор значений нейронов

  uint32_t InputSize_X;///<размер входного тензора по X
  uint32_t InputSize_Y;///<размер входного тензора по Y
  uint32_t InputSize_Z;///<размер входного тензора по Z

  CTensor<type_t> cTensor_TimeLine;///<тензор моментов времени
  uint32_t TimeSize;///<размер вектора времени
  uint32_t MaxTimeCounter;///<максимальное значение времени

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<тензоры дельты слоя
  CTensor<type_t> cTensor_DeltaLocalNet;///<тензор дельты слоя для обучения встроенной сети

  //внутренняя сеть
  std::vector< std::shared_ptr<INetLayer<type_t> > > LocalNet;///<внутренняя сеть

  //режим усреднения
  using INetLayer<type_t>::EMAEnabled;
  using INetLayer<type_t>::UseEMA;
  using INetLayer<type_t>::EMA_K;
  //ограничение нормы
  using INetLayer<type_t>::ClipByNormThresHold;///<ограничение нормы

  /*
  using INetLayer<type_t>::SetInferenceMode;
  using INetLayer<type_t>::SetUseEMA;
  using INetLayer<type_t>::TrainingModeGradient;
  using INetLayer<type_t>::TrainingModeAdam;
  */
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerTimeEmbeddingMLP(INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t time_size=128,uint32_t max_time_counter=1000,uint32_t batch_size=1);
  CNetLayerTimeEmbeddingMLP(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerTimeEmbeddingMLP();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t time_size=128,uint32_t max_time_counter=1000,uint32_t batch_size=1);///<создать слой
  void Reset(type_t scale=1);///<выполнить инициализацию слоя
  void SetOutput(CTensor<type_t> &output);///<задать выход слоя
  void GetOutput(CTensor<type_t> &output);///<получить выход слоя
  void Forward(void);///<выполнить прямой проход по слою
  CTensor<type_t>& GetOutputTensor(void);///<получить ссылку на выходной тензор
  void SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr);///<задать указатель на последующий слой
  bool Save(IDataStream *iDataStream_Ptr);///<сохранить параметры слоя
  bool Load(IDataStream *iDataStream_Ptr,bool check_size=false);///<загрузить параметры слоя
  bool SaveTrainingParam(IDataStream *iDataStream_Ptr);///<сохранить параметры обучения слоя
  bool LoadTrainingParam(IDataStream *iDataStream_Ptr);///<загрузить параметры обучения слоя

  void TrainingStart(void);///<начать процесс обучения
  void TrainingStop(void);///<завершить процесс обучения
  void TrainingBackward(bool create_delta_weight=true);///<выполнить обратный проход по сети для обучения
  void TrainingResetDeltaWeight(void);///<сбросить поправки к весам
  void TrainingUpdateWeight(double speed,double iteration,double batch_scale=1);///<выполнить обновления весов
  CTensor<type_t>& GetDeltaTensor(void);///<получить ссылку на тензор дельты слоя

  void SetOutputError(CTensor<type_t>& error);///<задать ошибку и расчитать дельту

  void ClipWeight(type_t min,type_t max);///<ограничить веса в диапазон

  void SetTimeStep(uint32_t index,uint32_t time_step);///<задать временной шаг

  void PrintInputTensorSize(const std::string &name);///<вывести размерность входного тензора слоя
  void PrintOutputTensorSize(const std::string &name);///<вывести размерность выходного тензора слоя

  void EnableEMA(bool state);///<разрешить/запретить использование усреднённых весов
  bool LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size=false);///<загрузить усреднённые веса
  bool SaveEMAWeight(IDataStream *iDataStream_Ptr);///<сохранить усреднённые веса

  void SetInferenceMode(bool state) override;///<включить режим вывода
  void SetUseEMA(bool state) override;///<переключиться на усреднённые веса
  void TrainingModeGradient(void) override;///<включить режим обучения "градиентный спуск"
  void TrainingModeAdam(double beta1=0.9,double beta2=0.999,double epsilon=1E-6) override;///<включить режим обучения "алгоритм Adam"
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerTimeEmbeddingMLP<type_t>::CNetLayerTimeEmbeddingMLP(INetLayer<type_t> *prev_layer_ptr,uint32_t time_size,uint32_t max_time_counter,uint32_t batch_size)
{
 Create(prev_layer_ptr,time_size,max_time_counter,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerTimeEmbeddingMLP<type_t>::CNetLayerTimeEmbeddingMLP(void)
{
 Create();
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerTimeEmbeddingMLP<type_t>::~CNetLayerTimeEmbeddingMLP()
{
}

//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
/*!создать слой
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::Create(INetLayer<type_t> *prev_layer_ptr,uint32_t time_size,uint32_t max_time_counter,uint32_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;
 TimeSize=time_size;
 MaxTimeCounter=max_time_counter;

 if (prev_layer_ptr==NULL) throw("Слой времени не может быть входным!");//слой без предшествующего считается входным
 //создаём вектор времни
 cTensor_TimeLine=CTensor<type_t>(BatchSize,TimeSize,1,1);

 //размер входного тензора
 uint32_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 uint32_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 uint32_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 //размер выходного тензора
 uint32_t output_x=input_x;
 uint32_t output_y=input_y;
 uint32_t output_z=input_z;

 //запомним размеры входного тензора, чтобы потом всегда к ним приводить
 InputSize_X=input_x;
 InputSize_Y=input_y;
 InputSize_Z=input_z;

 //создаём внутреннюю сеть
 LocalNet.clear();
 LocalNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolutionInput<type_t>(cTensor_TimeLine.GetSizeZ(),cTensor_TimeLine.GetSizeY(),cTensor_TimeLine.GetSizeX(),cTensor_TimeLine.GetSizeW())));
 LocalNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolution<type_t>(InputSize_Z,1,1,1,0,0,LocalNet.back().get(),BatchSize)));
 LocalNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,LocalNet.back().get(),BatchSize)));
 LocalNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolution<type_t>(InputSize_Z,1,1,1,0,0,LocalNet.back().get(),BatchSize)));
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->Reset();
 LocalNet.back()->Reset(0.001);

 //создаём выходные тензоры
 cTensor_H=CTensor<type_t>(BatchSize,output_z,output_y,output_x);
 //задаём предшествующему слою, что мы его последующий слой
 prev_layer_ptr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::Reset(type_t scale)
{
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->Reset(scale);
 LocalNet.back()->Reset(scale*0.001);
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::SetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerTimeEmbeddingMLP<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerTimeEmbeddingMLP<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerTimeEmbeddingMLP<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerTimeEmbeddingMLP<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerTimeEmbeddingMLP<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerTimeEmbeddingMLP<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerTimeEmbeddingMLP<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerTimeEmbeddingMLP<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::Forward(void)
{
 //выполняем встроенную сеть
 LocalNet[0]->GetOutputTensor()=cTensor_TimeLine;//задаём время
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->Forward();
 //прибавляем ко всем значениям (y,x) одно и то же значение
 CTensorMath<type_t>::AddToXY(cTensor_H,PrevLayerPtr->GetOutputTensor(),LocalNet.back()->GetOutputTensor());
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerTimeEmbeddingMLP<type_t>::GetOutputTensor(void)
{
 return(cTensor_H);
}
//----------------------------------------------------------------------------------------------------
/*!задать указатель на последующий слой
\param[in] next_layer_ptr Указатель на последующий слой
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
{
 NextLayerPtr=next_layer_ptr;
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerTimeEmbeddingMLP<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveInt32(TimeSize);
 iDataStream_Ptr->SaveInt32(MaxTimeCounter);
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->Save(iDataStream_Ptr);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerTimeEmbeddingMLP<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 if (check_size==true)
 {
  if (iDataStream_Ptr->LoadUInt32()!=TimeSize) throw("Ошибка загрузки слоя времени MLP: неверный размер вектора времени.");
  if (iDataStream_Ptr->LoadUInt32()!=MaxTimeCounter) throw("Ошибка загрузки слоя времени MLP: неверный размер максимального значения времени.");
 }
 else
 {
  TimeSize=iDataStream_Ptr->LoadUInt32();
  MaxTimeCounter=iDataStream_Ptr->LoadInt32();
 }
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->Load(iDataStream_Ptr,check_size);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerTimeEmbeddingMLP<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
{
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->SaveTrainingParam(iDataStream_Ptr);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerTimeEmbeddingMLP<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->LoadTrainingParam(iDataStream_Ptr);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 cTensor_Delta=cTensor_H;
 uint32_t x=LocalNet.back().get()->GetOutputTensor().GetSizeX();
 uint32_t y=LocalNet.back().get()->GetOutputTensor().GetSizeY();
 uint32_t z=LocalNet.back().get()->GetOutputTensor().GetSizeZ();
 uint32_t w=LocalNet.back().get()->GetOutputTensor().GetSizeW();
 cTensor_DeltaLocalNet=CTensor<type_t>(w,z,y,x);

 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->TrainingStart();
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
 cTensor_DeltaLocalNet=CTensor<type_t>(1,1,1,1);
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->TrainingStop();
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::TrainingBackward(bool create_delta_weight)
{
 //задаём ошибку предыдущего слоя
 PrevLayerPtr->SetOutputError(cTensor_Delta);
 //обучаем встроенную сеть
 LocalNet.back()->SetOutputError(cTensor_DeltaLocalNet);
 for(uint32_t m=0,n=LocalNet.size()-1;m<LocalNet.size();m++,n--) LocalNet[n]->TrainingBackward(create_delta_weight);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::TrainingResetDeltaWeight(void)
{
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->TrainingResetDeltaWeight();
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->TrainingUpdateWeight(speed,iteration,batch_scale);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerTimeEmbeddingMLP<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::SetOutputError(CTensor<type_t>& error)
{
 cTensor_Delta=error;
 //суммируем градиент
 type_t k=cTensor_H.GetSizeY()*cTensor_H.GetSizeX();
 k=1/k;
 CTensorMath<type_t>::SumXY(cTensor_DeltaLocalNet,cTensor_Delta,1);
}
//----------------------------------------------------------------------------------------------------
/*!ограничить веса в диапазон
\param[in] min Минимальное значение веса
\param[in] max Максимальное значение веса
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::ClipWeight(type_t min,type_t max)
{
 //for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->ClipWeight(min,max);//TODO: не уверен, что в слое времени можно разрешать такую операцию
}

//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
 //заполняем тензор времени
 type_t t=time_step;//static_cast<type_t>(time_step)/static_cast<type_t>(MaxTimeCounter);//нормализуем
 uint32_t d=cTensor_TimeLine.GetSizeZ();
 for(uint32_t z=0;z<d/2;z++)
 {
  type_t exponent=static_cast<type_t>(2*z)/static_cast<type_t>(d);
  type_t freq=std::pow(10000.0f,exponent);
  type_t angle=t/freq;
  cTensor_TimeLine.SetElement(index,2*z,0,0,std::sin(angle));
  cTensor_TimeLine.SetElement(index,2*z+1,0,0,std::cos(angle));
 }
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" TimeEmbeddingMLP: input",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::PrintOutputTensorSize(const std::string &name)
{
 GetOutputTensor().Print(name+" TimeEmbeddingMLP: output",false);
}
//----------------------------------------------------------------------------------------------------
/*!<разрешить/запретить использование усреднённых весов
\param[in] state - разрешить запретить использование усреднённых весов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::EnableEMA(bool state)
{
 EMAEnabled=state;
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->EnableEMA(state);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerTimeEmbeddingMLP<type_t>::LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size)
{
 if (check_size==true)
 {
  if (iDataStream_Ptr->LoadUInt32()!=TimeSize) throw("Ошибка загрузки слоя времени MLP: неверный размер вектора времени.");
  if (iDataStream_Ptr->LoadUInt32()!=MaxTimeCounter) throw("Ошибка загрузки слоя времени MLP: неверный размер максимального значения времени.");
 }
 else
 {
  TimeSize=iDataStream_Ptr->LoadUInt32();
  MaxTimeCounter=iDataStream_Ptr->LoadInt32();
 }
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->LoadEMAWeight(iDataStream_Ptr,check_size);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerTimeEmbeddingMLP<type_t>::SaveEMAWeight(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveInt32(TimeSize);
 iDataStream_Ptr->SaveInt32(MaxTimeCounter);
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->SaveEMAWeight(iDataStream_Ptr);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!включить режим вывода
\param[in] state Состояние режима вывода
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::SetInferenceMode(bool state)
{
 INetLayer<type_t>::SetInferenceMode(state);
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->SetInferenceMode(state);
}
//----------------------------------------------------------------------------------------------------
/*!переключиться на усреднённые веса
\param[in] state Режим усреднённых весов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::SetUseEMA(bool state)
{
 INetLayer<type_t>::SetUseEMA(state);
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->SetUseEMA(state);
}
//----------------------------------------------------------------------------------------------------
/*!включить режим обучения "градиентный спуск"
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::TrainingModeGradient(void)
{
 INetLayer<type_t>::TrainingModeGradient();
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->TrainingModeGradient();
}
//----------------------------------------------------------------------------------------------------
/*!включить режим обучения "алгоритм Adam"
\param[in] beta1 Коэффициент beta1
\param[in] beta2 Коэффициент beta2
\param[in] epsilon Коэффициент epsilon
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbeddingMLP<type_t>::TrainingModeAdam(double beta1,double beta2,double epsilon)
{
 INetLayer<type_t>::TrainingModeAdam(beta1,beta2,epsilon);
 for(uint32_t n=0;n<LocalNet.size();n++) LocalNet[n]->TrainingModeAdam(beta1,beta2,epsilon);
}


#endif
