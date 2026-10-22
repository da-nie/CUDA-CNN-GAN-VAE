#ifndef C_NET_LAYER_GROUP_NORMALIZATION_H
#define C_NET_LAYER_GROUP_NORMALIZATION_H

//****************************************************************************************************
//\file Групповая нормализация
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdio.h>
#include <fstream>
#include <math.h>

#include "../common/idatastream.h"
#include "inetlayer.cu.h"
#include "../cuda/tensor.cu.h"
#include "neuron.cu.h"

//****************************************************************************************************
//константы
//****************************************************************************************************
static const double GN_EPSILON=1e-4;

//****************************************************************************************************
//!Групповая нормализация
//****************************************************************************************************
template<class type_t>
class CNetLayerGroupNormalization:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  uint32_t Layer;

  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  uint32_t BatchSize;///<размер пакета для обучения
  uint32_t NumGroups;///<количество групп
  uint32_t ChannelsPerGroup;///<количество каналов в одной группе

  CTensor<type_t> cTensor_Beta;///<параметр сдвига (1xZx1x1)
  CTensor<type_t> cTensor_Gamma;///<параметр масштабирования (1xZx1x1)

  CTensor<type_t> cTensor_Beta_EMA;///<EMA параметр сдвига
  CTensor<type_t> cTensor_Gamma_EMA;///<EMA параметр масштабирования

  CTensor<type_t> cTensor_H;///<тензор выхода слоя
  CTensor<type_t> cTensor_Delta;///<тензор ошибки (входящий)
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки для предыдущего слоя

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_XHAT;     ///< нормализованные значения
  CTensor<type_t> cTensor_InvStd;   ///< обратное стандартное отклонение (BatchSize x NumGroups x 1 x 1)

  CTensor<type_t> cTensor_dGamma;///<градиент по гамме
  CTensor<type_t> cTensor_dBeta;///<градиент по бете

  bool TrainingEnabled;///<включено ли обучение

  //для оптимизации Adam
  CTensor<type_t> cTensor_MK;///<тензор фильтра 1
  CTensor<type_t> cTensor_VK;///<тензор фильтра 2
  CTensor<type_t> cTensor_MB;///<коэффициент фильтра 1 сдвигов
  CTensor<type_t> cTensor_VB;///<коэффициент фильтра 2 сдвигов

  using INetLayer<type_t>::Beta1;///<параметры алгоритма Adam
  using INetLayer<type_t>::Beta2;
  using INetLayer<type_t>::Epsilon;
  //режим усреднения
  using INetLayer<type_t>::EMAEnabled;
  using INetLayer<type_t>::UseEMA;
  using INetLayer<type_t>::EMA_K;
  //ограничение нормы
  using INetLayer<type_t>::ClipByNormThresHold;///<ограничение нормы
  using INetLayer<type_t>::InferenceMode;///режим вывода
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerGroupNormalization(uint32_t num_groups,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerGroupNormalization();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(uint32_t num_groups,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
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
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerGroupNormalization<type_t>::CNetLayerGroupNormalization(uint32_t num_groups,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 Create(num_groups,prev_layer_ptr,batch_size);
}

//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerGroupNormalization<type_t>::~CNetLayerGroupNormalization() {}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
/*!создать слой
\param[in] num_groups Количество групп
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::Create(uint32_t num_groups,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 static uint32_t i=0;
 i++;
 Layer=i;

 if (prev_layer_ptr==NULL) throw("Слой GroupNormalization не может быть входным!");

 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;
 BatchSize=batch_size;
 NumGroups=num_groups;

 uint32_t Z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 uint32_t Y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 uint32_t X=PrevLayerPtr->GetOutputTensor().GetSizeX();

 if (Z % NumGroups != 0) throw("Количество каналов Z должно делиться на количество групп!");
 ChannelsPerGroup=Z / NumGroups;

 cTensor_H=CTensor<type_t>(BatchSize,Z,Y,X);
 cTensor_Delta=CTensor<type_t>(BatchSize,Z,Y,X);
 cTensor_XHAT=CTensor<type_t>(BatchSize,Z,Y,X);
 cTensor_PrevLayerError=CTensor<type_t>(BatchSize,Z,Y,X);

 cTensor_Gamma=CTensor<type_t>(1,Z,1,1);
 cTensor_Beta=CTensor<type_t>(1,Z,1,1);
 cTensor_Gamma_EMA=CTensor<type_t>(1,Z,1,1);
 cTensor_Beta_EMA=CTensor<type_t>(1,Z,1,1);

 cTensor_InvStd=CTensor<type_t>(BatchSize,NumGroups,1,1);

 prev_layer_ptr->SetNextLayerPtr(this);
 TrainingEnabled=false;
}

//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию весов и сдвигов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::Reset(type_t scale)
{
 CTensorMath<type_t>::Fill(cTensor_Gamma,1);
 CTensorMath<type_t>::Fill(cTensor_Beta,0);
 CTensorMath<type_t>::Fill(cTensor_Gamma_EMA,1);
 CTensorMath<type_t>::Fill(cTensor_Beta_EMA,0);
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::SetOutput(CTensor<type_t> &output)
{
 cTensor_H.RestoreSize();
 //if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerGroupNormalization<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerGroup<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerGroup<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerGroup<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::GetOutput(CTensor<type_t> &output)
{
 cTensor_H.RestoreSize();
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerGroup<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerGroup<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerGroup<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerGroup<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::Forward(void)
{
 cTensor_H.RestoreSize();
 CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();

 CTensor<type_t> &gamma=(UseEMA)?cTensor_Gamma_EMA:cTensor_Gamma;
 CTensor<type_t> &beta=(UseEMA)?cTensor_Beta_EMA:cTensor_Beta;

 // Вызов функции из CTensorMath с передачей GN_EPSILON
    // используем единую константу
 CTensorMath<type_t>::GroupNormForward(cTensor_H,input,gamma,beta,cTensor_XHAT,cTensor_InvStd,NumGroups,ChannelsPerGroup,(type_t)GN_EPSILON);
}


//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerGroupNormalization<type_t>::GetOutputTensor(void)
{
 cTensor_H.RestoreSize();
 return(cTensor_H);
}
//----------------------------------------------------------------------------------------------------
/*!задать указатель на последующий слой
\param[in] next_layer_ptr Указатель на последующий слой
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerGroupNormalization<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 cTensor_Beta.Save(iDataStream_Ptr);
 cTensor_Gamma.Save(iDataStream_Ptr);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerGroupNormalization<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 cTensor_Beta.Load(iDataStream_Ptr,check_size);
 cTensor_Gamma.Load(iDataStream_Ptr,check_size);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerGroupNormalization<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
{
 cTensor_MK.Save(iDataStream_Ptr);
 cTensor_VK.Save(iDataStream_Ptr);
 cTensor_MB.Save(iDataStream_Ptr);
 cTensor_VB.Save(iDataStream_Ptr);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerGroupNormalization<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 cTensor_MK.Load(iDataStream_Ptr);
 cTensor_VK.Load(iDataStream_Ptr);
 cTensor_MB.Load(iDataStream_Ptr);
 cTensor_VB.Load(iDataStream_Ptr);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::TrainingStart(void)
{
 cTensor_H.RestoreSize();

 uint32_t Z=cTensor_Gamma.GetSizeZ();
 cTensor_dGamma=CTensor<type_t>(1,Z,1,1);
 cTensor_dBeta=CTensor<type_t>(1,Z,1,1);

 cTensor_MK=cTensor_dGamma;
 cTensor_VK=cTensor_dGamma;
 cTensor_MB=cTensor_dBeta;
 cTensor_VB=cTensor_dBeta;

 CTensorMath<type_t>::Fill(cTensor_MK,0);
 CTensorMath<type_t>::Fill(cTensor_VK,0);
 CTensorMath<type_t>::Fill(cTensor_MB,0);
 CTensorMath<type_t>::Fill(cTensor_VB,0);

 TrainingEnabled=true;
}

//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_dGamma=CTensor<type_t>(1,1,1,1);
 cTensor_dBeta=CTensor<type_t>(1,1,1,1);

 cTensor_MK=cTensor_dGamma;
 cTensor_VK=cTensor_dGamma;
 cTensor_MB=cTensor_dBeta;
 cTensor_VB=cTensor_dBeta;

 TrainingEnabled=false;
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::TrainingBackward(bool create_delta_weight)
{
 if (create_delta_weight)
 {
  CTensorMath<type_t>::Fill(cTensor_dGamma,0);
  CTensorMath<type_t>::Fill(cTensor_dBeta,0);
 }

 // Вызов функции из CTensorMath
 CTensorMath<type_t>::GroupNormBackward(cTensor_Delta,cTensor_XHAT,cTensor_Gamma,cTensor_InvStd,cTensor_PrevLayerError,cTensor_dGamma,cTensor_dBeta,NumGroups,ChannelsPerGroup,create_delta_weight);
 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
}

//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::TrainingResetDeltaWeight(void)
{
 CTensorMath<type_t>::Fill(cTensor_dGamma,0);
 CTensorMath<type_t>::Fill(cTensor_dBeta,0);
}

//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
 CTensorMath<type_t>::ClipByNormXY(cTensor_dGamma,cTensor_dGamma,ClipByNormThresHold);
 CTensorMath<type_t>::ClipByNormXY(cTensor_dBeta,cTensor_dBeta,ClipByNormThresHold);

 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_ADAM)
 {
  //применяем алгоритм Adam
  CTensorMath<type_t>::Adam(cTensor_Gamma,cTensor_dGamma,cTensor_MK,cTensor_VK,BatchSize*batch_scale,speed,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_Beta,cTensor_dBeta,cTensor_MB,cTensor_VB,BatchSize*batch_scale,speed,Beta1,Beta2,Epsilon,iteration);
 }
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_GRADIENT)
 {
  double s=speed / static_cast<double>(BatchSize);
  CTensorMath<type_t>::Sub(cTensor_Gamma,cTensor_Gamma,cTensor_dGamma,1,s/batch_scale);
  CTensorMath<type_t>::Sub(cTensor_Beta,cTensor_Beta,cTensor_dBeta,1,s/batch_scale);
 }

 // ДОБАВЛЕННАЯ ЗАЩИТА: Ограничиваем Gamma и Beta в диапазоне [-100,100]
 // Этого с головой хватит для нормализации,но это спасет от взрыва сети.
 CTensorMath<type_t>::Clip(cTensor_Gamma,-100.0,100.0);
 CTensorMath<type_t>::Clip(cTensor_Beta,-100.0,100.0);

 if (EMAEnabled)
 {
  CTensorMath<type_t>::Add(cTensor_Gamma_EMA,cTensor_Gamma_EMA,cTensor_Gamma,EMA_K,1.0-EMA_K);
  CTensorMath<type_t>::Add(cTensor_Beta_EMA,cTensor_Beta_EMA,cTensor_Beta,EMA_K,1.0-EMA_K);
 }
}

//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerGroupNormalization<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::SetOutputError(CTensor<type_t>& error)
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
void CNetLayerGroupNormalization<type_t>::ClipWeight(type_t min,type_t max)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" GroupNormalization: input",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::PrintOutputTensorSize(const std::string &name)
{
 cTensor_H.RestoreSize();
 GetOutputTensor().Print(name+" GroupNormalization: output",false);
}

//----------------------------------------------------------------------------------------------------
/*!<разрешить/запретить использование усреднённых весов
\param[in] state - разрешить запретить использование усреднённых весов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerGroupNormalization<type_t>::EnableEMA(bool state)
{
 EMAEnabled=state;
 if (state)
 {
  cTensor_Gamma_EMA=cTensor_Gamma;
  cTensor_Beta_EMA=cTensor_Beta;
 }
}

//----------------------------------------------------------------------------------------------------
/*!загрузить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerGroupNormalization<type_t>::LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size)
{
 cTensor_Beta_EMA.Load(iDataStream_Ptr,check_size);
 cTensor_Gamma_EMA.Load(iDataStream_Ptr,check_size);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!сохранить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerGroupNormalization<type_t>::SaveEMAWeight(IDataStream *iDataStream_Ptr)
{
 cTensor_Beta_EMA.Save(iDataStream_Ptr);
 cTensor_Gamma_EMA.Save(iDataStream_Ptr);
 return(true);
}

#endif
