#ifndef C_NET_LAYER_VIEW_H
#define C_NET_LAYER_VIEW_H

//****************************************************************************************************
//\file Слой преобразования
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdio.h>
#include <fstream>

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
//!Слой преобразования
//****************************************************************************************************
template<class type_t>
class CNetLayerView:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  uint32_t InputX;///<входной слой
  uint32_t InputY;
  uint32_t InputZ;

  uint32_t OutputX;///<выходной слой
  uint32_t OutputY;
  uint32_t OutputZ;

  CTensor<type_t> cTensor_Delta;///<тензор ошибки выхода слоя

  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  uint32_t BatchSize;///<размер пакета для обучения

  //режим усреднения
  using INetLayer<type_t>::EMAEnabled;
  using INetLayer<type_t>::UseEMA;
  using INetLayer<type_t>::EMA_K;
  //ограничение нормы
  using INetLayer<type_t>::ClipByNormThresHold;///<ограничение нормы
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerView(uint32_t size_z,uint32_t size_y,uint32_t size_x,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerView(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerView();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(uint32_t size_z,uint32_t size_y,uint32_t size_x,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
  void Reset(type_t scale=1);///<выполнить инициализацию весов и сдвигов
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
CNetLayerView<type_t>::CNetLayerView(uint32_t size_z,uint32_t size_y,uint32_t size_x,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 Create(size_z,size_y,size_x,prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerView<type_t>::CNetLayerView(void)
{
 Create(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerView<type_t>::~CNetLayerView()
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
\param[in] size_z Глубина выходного слоя
\param[in] size_y Высота выходного слоя
\param[in] size_x Ширина выходного слоя
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::Create(uint32_t size_z,uint32_t size_y,uint32_t size_x,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 if (prev_layer_ptr==NULL) throw "Слой View не может быть входным!";

 InputX=prev_layer_ptr->GetOutputTensor().GetSizeX();
 InputY=prev_layer_ptr->GetOutputTensor().GetSizeY();
 InputZ=prev_layer_ptr->GetOutputTensor().GetSizeZ();

 BatchSize=batch_size;
 OutputX=size_x;
 OutputY=size_y;
 OutputZ=size_z;
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию весов и сдвигов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::Reset(type_t scale)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::SetOutput(CTensor<type_t> &output)
{
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(BatchSize,OutputZ,OutputY,OutputX);
 PrevLayerPtr->GetOutputTensor()=output;
 PrevLayerPtr->GetOutputTensor().RestoreSize();
}
//----------------------------------------------------------------------------------------------------
/*!получить выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=OutputX) throw("void CNetLayerView<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=OutputY) throw("void CNetLayerView<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=OutputZ) throw("void CNetLayerView<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=PrevLayerPtr->GetOutputTensor().GetSizeW()) throw("void CNetLayerView<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=GetOutputTensor();
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::Forward(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerView<type_t>::GetOutputTensor(void)
{
 CTensor<type_t> &cTensor=PrevLayerPtr->GetOutputTensor();
 cTensor.ReinterpretSize(BatchSize,OutputZ,OutputY,OutputX);
 return(cTensor);
}
//----------------------------------------------------------------------------------------------------
/*!задать указатель на последующий слой
\param[in] next_layer_ptr Указатель на последующий слой
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerView<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerView<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerView<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerView<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::TrainingStart(void)
{
 cTensor_Delta=CTensor<type_t>(BatchSize,OutputZ,OutputY,OutputX);
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::TrainingStop(void)
{
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::TrainingBackward(bool create_delta_weight)
{
 cTensor_Delta.ReinterpretSize(BatchSize,InputZ,InputY,InputX);
 PrevLayerPtr->SetOutputError(cTensor_Delta);
 cTensor_Delta.RestoreSize();
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerView<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::SetOutputError(CTensor<type_t>& error)
{
 cTensor_Delta=error;
}
//----------------------------------------------------------------------------------------------------
/*!ограничить веса в диапазон
\param[in] min Минимальное значение веса
\param[in] max Максимальное значение веса
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::ClipWeight(type_t min,type_t max)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::PrintInputTensorSize(const std::string &name)
{
 char str[255];
 SYSTEM::PutMessageToConsole("***** View: input *****");
 sprintf(str,"W:%i Z:%i Y:%i X:%i",static_cast<int>(BatchSize),static_cast<int>(InputZ),static_cast<int>(InputY),static_cast<int>(InputX));
 SYSTEM::PutMessageToConsole(str);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::PrintOutputTensorSize(const std::string &name)
{
 char str[255];
 SYSTEM::PutMessageToConsole("***** View: output *****");
 sprintf(str,"W:%i Z:%i Y:%i X:%i",static_cast<int>(BatchSize),static_cast<int>(OutputZ),static_cast<int>(OutputY),static_cast<int>(OutputX));
 SYSTEM::PutMessageToConsole(str);
}
//----------------------------------------------------------------------------------------------------
/*!<разрешить/запретить использование усреднённых весов
\param[in] state - разрешить запретить использование усреднённых весов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerView<type_t>::EnableEMA(bool state)
{
 EMAEnabled=state;
}
//----------------------------------------------------------------------------------------------------
/*!загрузить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerView<type_t>::LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerView<type_t>::SaveEMAWeight(IDataStream *iDataStream_Ptr)
{
 return(true);
}

#endif
