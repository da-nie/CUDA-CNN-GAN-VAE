#ifndef C_NET_LAYER_TIME_EMBEDDING_H
#define C_NET_LAYER_TIME_EMBEDDING_H

//****************************************************************************************************
//\file Слой времени
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
//!Слой времени
//****************************************************************************************************
template<class type_t>
class CNetLayerTimeEmbedding:public INetLayer<type_t>
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

  uint32_t Dimension;///<размерность слоя

  uint32_t InputSize_X;///<размер входного тензора по X
  uint32_t InputSize_Y;///<размер входного тензора по Y
  uint32_t InputSize_Z;///<размер входного тензора по Z

  std::vector< std::vector<type_t> > TimeLine;///<вектор кодирования

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<тензоры дельты слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerTimeEmbedding(uint32_t dimension,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerTimeEmbedding(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerTimeEmbedding();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(uint32_t dimension,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
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
CNetLayerTimeEmbedding<type_t>::CNetLayerTimeEmbedding(uint32_t dimension,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 Create(dimension,prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerTimeEmbedding<type_t>::CNetLayerTimeEmbedding(void)
{
 Create(1);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerTimeEmbedding<type_t>::~CNetLayerTimeEmbedding()
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
\param[in] pooling_y Коэффициент сжатия по Y
\param[in] pooling_x Коэффициент сжатия по X
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::Create(uint32_t dimension,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 Dimension=dimension;

 if (dimension==0) throw("Размерность слоя времени не должна быть нулевой!");

 if (prev_layer_ptr==NULL) throw("Слой времени не может быть входным!");//слой без предшествующего считается входным

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

 //создаём выходные тензоры
 cTensor_H=CTensor<type_t>(BatchSize,output_z,output_y,output_x);
 //задаём предшествующему слою, что мы его последующий слой
 prev_layer_ptr->SetNextLayerPtr(this);

 TimeLine.resize(BatchSize);
 for(uint32_t n=0;n<BatchSize;n++) TimeLine[n].resize(Dimension);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::Reset(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::SetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerTimeEmbedding<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerTimeEmbedding<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerTimeEmbedding<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerTimeEmbedding<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerTimeEmbedding<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerTimeEmbedding<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerTimeEmbedding<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerTimeEmbedding<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::Forward(void)
{
 //приведём входной и выходной тензор к нужному виду
 CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();
 CTensor<type_t> &output=cTensor_H;

 uint32_t input_x=input.GetSizeX();
 uint32_t input_y=input.GetSizeY();
 uint32_t input_z=input.GetSizeZ();

 uint32_t output_x=output.GetSizeX();
 uint32_t output_y=output.GetSizeY();
 uint32_t output_z=output.GetSizeZ();

 uint32_t size=input_x*input_y*input_z;

 input.ReinterpretSize(BatchSize,1,1,size);
 output.ReinterpretSize(BatchSize,1,1,size);
 //проецирование временной линии к нужной размерности
 type_t k=static_cast<type_t>(Dimension)/static_cast<type_t>(size);
 for(uint32_t b=0;b<BatchSize;b++)
 {
  type_t kd=0;
  for(uint32_t n=0;n<size;n++,kd+=k)
  {
   uint32_t i=static_cast<uint32_t>(kd);
   type_t value=input.GetElement(b,0,0,n);
   value+=TimeLine[b][i];
   output.SetElement(b,0,0,n,value);
  }
 }
 input.ReinterpretSize(BatchSize,input_z,input_y,input_x);
 output.ReinterpretSize(BatchSize,output_z,output_y,output_x);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerTimeEmbedding<type_t>::GetOutputTensor(void)
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
void CNetLayerTimeEmbedding<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerTimeEmbedding<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveUInt32(Dimension);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerTimeEmbedding<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 if (check_size==true)
 {
  if (Dimension!=iDataStream_Ptr->LoadUInt32()) throw("Ошибка загрузки слоя времени: неверный размер слоя.");
 }
 else
 {
  Dimension=iDataStream_Ptr->LoadUInt32();
 }
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerTimeEmbedding<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
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
bool CNetLayerTimeEmbedding<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 cTensor_Delta=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::TrainingBackward(bool create_delta_weight)
{
 //задаём ошибку предыдущего слоя
 PrevLayerPtr->SetOutputError(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerTimeEmbedding<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::SetOutputError(CTensor<type_t>& error)
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
void CNetLayerTimeEmbedding<type_t>::ClipWeight(type_t min,type_t max)
{
}

//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerTimeEmbedding<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
 //создаём таблицу синусоидального позиционного кодирования
 for(uint32_t i=0;i<Dimension/2;i++)
 {
  type_t angle=time_step/std::pow(10000.0f,2.0f*i/Dimension);
  TimeLine[index][2*i]=std::sin(angle);
  TimeLine[index][2*i+1]=std::cos(angle);
 }
}

#endif
