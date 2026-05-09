#ifndef C_NET_LAYER_VAE_CODER_OUTPUT_H
#define C_NET_LAYER_VAE_CODER_OUTPUT_H

//****************************************************************************************************
//\file Слой выхода кодера генеративного автоэнкодера
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdio.h>
#include <fstream>
#include <vector>
#include <math.h>
#include <memory>

#include "../common/idatastream.h"
#include "inetlayer.cu.h"
#include "../cuda/tensor.cu.h"
#include "neuron.cu.h"
#include "../common/crandom.h"

#include "cnetlayerlinear.cu.h"
#include "cnetlayersplitter.cu.h"

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
//! Слой выхода кодера генеративного автоэнкодера
//****************************************************************************************************
template<class type_t>
class CNetLayerVAECoderOutput:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  INetLayer<type_t> *MuLayerPtr;///<указатель на слой mu
  INetLayer<type_t> *LogVarLayerPtr;///<указатель на слой logvar

  uint32_t BatchSize;///<размер пакета для обучения

  CTensor<type_t> cTensor_H;///<выходной тензор значений нейронов
  CTensor<type_t> cTensor_Epsilon;///<тензор шума

  uint32_t InputSize_X;///<размер входного тензора по X
  uint32_t InputSize_Y;///<размер входного тензора по Y
  uint32_t InputSize_Z;///<размер входного тензора по Z

  uint32_t OutputSize_X;///<размер выходного тензора по X
  uint32_t OutputSize_Y;///<размер выходного тензора по Y
  uint32_t OutputSize_Z;///<размер выходного тензора по Z

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<тензоры дельты слоя
  CTensor<type_t> cTensor_PrevLayerError_Mu;///<тензоры ошибки слоя mu
  CTensor<type_t> cTensor_PrevLayerError_LogVar;///<тензоры ошибки слоя logvar

  type_t KLSpeed;///<предельная скорость KL-дивергенции
  type_t KLSpeedCurrent;///<текущая скорость KL-дивергенции
  //режим усреднения
  using INetLayer<type_t>::EMAEnabled;
  using INetLayer<type_t>::UseEMA;
  using INetLayer<type_t>::EMA_K;
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerVAECoderOutput(type_t kl_speed=0.1,INetLayer<type_t> *mu_layer_ptr=NULL,INetLayer<type_t> *logvar_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerVAECoderOutput(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerVAECoderOutput();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(type_t kl_speed=0.1,INetLayer<type_t> *mu_layer_ptr=NULL,INetLayer<type_t> *logvar_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
  void Reset(void);///<выполнить инициализацию слоя
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
CNetLayerVAECoderOutput<type_t>::CNetLayerVAECoderOutput(type_t kl_speed,INetLayer<type_t> *mu_layer_ptr,INetLayer<type_t> *logvar_layer_ptr,uint32_t batch_size)
{
 Create(kl_speed,mu_layer_ptr,logvar_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerVAECoderOutput<type_t>::CNetLayerVAECoderOutput(void)
{
 Create();
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerVAECoderOutput<type_t>::~CNetLayerVAECoderOutput()
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
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::Create(type_t kl_speed,INetLayer<type_t> *mu_layer_ptr,INetLayer<type_t> *logvar_layer_ptr,uint32_t batch_size)
{
 MuLayerPtr=mu_layer_ptr;
 LogVarLayerPtr=logvar_layer_ptr;
 NextLayerPtr=NULL;

 KLSpeed=kl_speed;
 KLSpeedCurrent=0;

 BatchSize=batch_size;

 if (mu_layer_ptr==NULL || logvar_layer_ptr==NULL) throw("Слой выхода кодера VAE не может быть входным!");//слой без предшествующего считается входным

 //запомним размеры входного тензора, чтобы потом всегда к ним приводить
 InputSize_X=MuLayerPtr->GetOutputTensor().GetSizeX();
 InputSize_Y=MuLayerPtr->GetOutputTensor().GetSizeY();
 InputSize_Z=MuLayerPtr->GetOutputTensor().GetSizeZ();

 if (InputSize_X!=LogVarLayerPtr->GetOutputTensor().GetSizeX() || InputSize_Y!=LogVarLayerPtr->GetOutputTensor().GetSizeY() || InputSize_Z!=LogVarLayerPtr->GetOutputTensor().GetSizeZ())
 {
  throw("Слои mu и logvar кодера VAE должны иметь одинаковую размерность!");
 }

 //размер выходного тензора
 OutputSize_X=InputSize_X;
 OutputSize_Y=InputSize_Y;
 OutputSize_Z=InputSize_Z;

 //создаём выходные тензоры
 cTensor_H=CTensor<type_t>(BatchSize,OutputSize_Z,OutputSize_Y,OutputSize_X);
 cTensor_Epsilon=CTensor<type_t>(BatchSize,OutputSize_Z,OutputSize_Y,OutputSize_X);
 //задаём предшествующему слою, что мы его последующий слой
 mu_layer_ptr->SetNextLayerPtr(this);
 logvar_layer_ptr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::Reset(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::Forward(void)
{
 //считаем выход слоя
 type_t mean=0;
 type_t stddev=1;

 //генератор случайных чисел
 static std::random_device rd;
 static std::mt19937 gen(rd());

 type_t kl_loss=0;//расстояние Кульбака-Лейбнера

 for(uint32_t w=0;w<BatchSize;w++)
 {
  std::normal_distribution<type_t> dist(mean,stddev);

  for(uint32_t z=0;z<OutputSize_Z;z++)
  {
   for(uint32_t y=0;y<OutputSize_Y;y++)
   {
    for(uint32_t x=0;x<OutputSize_X;x++)
    {
     type_t mu=MuLayerPtr->GetOutputTensor().GetElement(w,z,y,x);
     type_t log_var=LogVarLayerPtr->GetOutputTensor().GetElement(w,z,y,x);
     if (log_var>10) log_var=10;
     if (log_var<-10) log_var=-10;
     type_t e=exp(log_var/2.0);
     type_t epsilon=dist(gen);
     cTensor_Epsilon.SetElement(w,z,y,x,epsilon);
     type_t out=mu+e*epsilon;
     cTensor_H.SetElement(w,z,y,x,out);
     //считаем расстояние Кульбака-Лейбнера
     kl_loss+=1+log_var-mu*mu-exp(log_var);
    }
   }
  }
 }

 static char str[255];
 kl_loss*=-0.5;
 kl_loss/=BatchSize;
 sprintf(str,"KL loss:%f",kl_loss);
 SYSTEM::PutMessageToConsole(str);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerVAECoderOutput<type_t>::GetOutputTensor(void)
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
void CNetLayerVAECoderOutput<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerVAECoderOutput<type_t>::Save(IDataStream *iDataStream_Ptr)
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
bool CNetLayerVAECoderOutput<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
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
bool CNetLayerVAECoderOutput<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
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
bool CNetLayerVAECoderOutput<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 cTensor_Delta=cTensor_H;
 cTensor_PrevLayerError_Mu=CTensor<type_t>(BatchSize,OutputSize_Z,OutputSize_Y,OutputSize_X);
 cTensor_PrevLayerError_LogVar=CTensor<type_t>(BatchSize,OutputSize_Z,OutputSize_Y,OutputSize_X);
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
 cTensor_PrevLayerError_Mu=CTensor<type_t>(1,1,1,1);
 cTensor_PrevLayerError_LogVar=CTensor<type_t>(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingBackward(bool create_delta_weight)
{
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
 if (iteration<=5) KLSpeedCurrent=0;
 else
 {
  KLSpeedCurrent=KLSpeed;//*(iteration-20)/100;
  if (KLSpeedCurrent>KLSpeed) KLSpeedCurrent=KLSpeed;
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerVAECoderOutput<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::SetOutputError(CTensor<type_t>& error)
{
 cTensor_Delta=error;
 type_t k=KLSpeedCurrent;

 for(uint32_t w=0;w<BatchSize;w++)
 {
  for(uint32_t z=0;z<OutputSize_Z;z++)
  {
   for(uint32_t y=0;y<OutputSize_Y;y++)
   {
    for(uint32_t x=0;x<OutputSize_X;x++)
    {
     type_t epsilon=cTensor_Epsilon.GetElement(w,z,y,x);
     //считаем ошибку для слоя логарифма дисперсии
     type_t log_var_delta=cTensor_Delta.GetElement(w,z,y,x);
     type_t log_var=LogVarLayerPtr->GetOutputTensor().GetElement(w,z,y,x);
     type_t log_var_diff=1.0/2.0*epsilon*exp(log_var/2.0);//производная функции построения выхода слоя по log_var
     //по этим данным получаем ошибку на входе слоя
     type_t log_var_delta_out=log_var_delta*log_var_diff;
     log_var_delta_out+=-1.0/2.0*(1-exp(log_var))*k;//добавляем ошибку по расстоянию Кульбака-Лейбнера
     cTensor_PrevLayerError_LogVar.SetElement(w,z,y,x,log_var_delta_out);

     //считаем ошибку для слоя среднего
     type_t mu=MuLayerPtr->GetOutputTensor().GetElement(w,z,y,x);
     type_t mu_delta=cTensor_Delta.GetElement(w,z,y,x);
     type_t mu_diff=1;//производная функции построения выхода слоя по mu
     //по этим данным получаем ошибку на входе слоя
     type_t mu_delta_out=mu_delta*mu_diff;
     mu_delta_out+=-1.0/2.0*(0-2*mu)*k;//добавляем ошибку по расстоянию Кульбака-Лейбнера
	 cTensor_PrevLayerError_Mu.SetElement(w,z,y,x,mu_delta_out);
    }
   }
  }
 }
 LogVarLayerPtr->SetOutputError(cTensor_PrevLayerError_LogVar);
 MuLayerPtr->SetOutputError(cTensor_PrevLayerError_Mu);
}
//----------------------------------------------------------------------------------------------------
/*!ограничить веса в диапазон
\param[in] min Минимальное значение веса
\param[in] max Максимальное значение веса
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::ClipWeight(type_t min,type_t max)
{
}

//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (MuLayerPtr!=NULL) MuLayerPtr->GetOutputTensor().Print(name+" VAECoderOutput: input ",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::PrintOutputTensorSize(const std::string &name)
{
 GetOutputTensor().Print(name+" VAECoderOutput: output",false);
}
//----------------------------------------------------------------------------------------------------
/*!<разрешить/запретить использование усреднённых весов
\param[in] state - разрешить запретить использование усреднённых весов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::EnableEMA(bool state)
{
 EMAEnabled=true;
}
//----------------------------------------------------------------------------------------------------
/*!загрузить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerVAECoderOutput<type_t>::LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size)
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
bool CNetLayerVAECoderOutput<type_t>::SaveEMAWeight(IDataStream *iDataStream_Ptr)
{
 return(true);
}


#endif
