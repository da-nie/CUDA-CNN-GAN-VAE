#ifndef C_NET_LAYER_LINEAR_H
#define C_NET_LAYER_LINEAR_H

//****************************************************************************************************
//\file Линейный слой нейронов
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
#include "../common/crandom.h"

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
//!Линейный слой нейронов
//****************************************************************************************************
template<class type_t>
class CNetLayerLinear:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  size_t BatchSize;///<размер пакета для обучения

  CTensor<type_t> cTensor_W;///<тензор весов слоя
  CTensor<type_t> cTensor_B;///<тензор сдвигов слоя

  std::vector<CTensor<type_t>> cTensor_H_Array;///<тензоры значений нейронов до функции активации

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_dW;///<тензор поправок весов слоя
  CTensor<type_t> cTensor_dB;///<тензор поправок сдвигов слоя
  std::vector<CTensor<type_t>> cTensor_Delta_Array;///<тензоры дельты слоя
  CTensor<type_t> cTensor_TmpdW;///<вспомогательный тензор поправок весов слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя

  //для оптимизации Adam
  CTensor<type_t> cTensor_MW;///<тензор фильтра 1
  CTensor<type_t> cTensor_VW;///<тензор фильтра 2
  CTensor<type_t> cTensor_MB;///<тензор фильтра 1 сдвигов
  CTensor<type_t> cTensor_VB;///<тензор фильтра 2 сдвигов

  using INetLayer<type_t>::Beta1;///<параметры алгоритма Adam
  using INetLayer<type_t>::Beta2;
  using INetLayer<type_t>::Epsilon;

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerLinear(size_t neurons,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);
  CNetLayerLinear(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerLinear();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(size_t neurons,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);///<создать слой
  void Reset(void);///<выполнить инициализацию весов и сдвигов
  void SetOutput(size_t unit_index,CTensor<type_t> &output);///<задать выход слоя
  void GetOutput(size_t unit_index,CTensor<type_t> &output);///<получить выход слоя
  void Forward(void);///<выполнить прямой проход по слою
  CTensor<type_t>& GetOutputTensor(size_t unit_index);///<получить ссылку на выходной тензор
  void SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr);///<задать указатель на последующий слой
  bool Save(IDataStream *iDataStream_Ptr);///<сохранить параметры слоя
  bool Load(IDataStream *iDataStream_Ptr);///<загрузить параметры слоя
  bool SaveTrainingParam(IDataStream *iDataStream_Ptr);///<сохранить параметры обучения слоя
  bool LoadTrainingParam(IDataStream *iDataStream_Ptr);///<загрузить параметры обучения слоя

  void TrainingStart(void);///<начать процесс обучения
  void TrainingStop(void);///<завершить процесс обучения
  void TrainingBackward(bool create_delta_weight=true);///<выполнить обратный проход по сети для обучения
  void TrainingResetDeltaWeight(void);///<сбросить поправки к весам
  void TrainingUpdateWeight(double speed,double iteration);///<выполнить обновления весов
  CTensor<type_t>& GetDeltaTensor(size_t unit_index);///<получить ссылку на тензор дельты слоя

  void SetOutputError(size_t unit_index,CTensor<type_t>& error);///<задать ошибку и расчитать дельту

  void ClipWeight(type_t min,type_t max);///<ограничить веса в диапазон
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
CNetLayerLinear<type_t>::CNetLayerLinear(size_t neurons,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 Create(neurons,prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerLinear<type_t>::CNetLayerLinear(void)
{
 Create(1);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerLinear<type_t>::~CNetLayerLinear()
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
\param[in] neurons Количество нейронов слоя
\param[in] neuron_function Функция активации нейронов
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::Create(size_t neurons,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 cTensor_H_Array.resize(BatchSize);

 for(size_t n=0;n<BatchSize;n++) cTensor_H_Array[n]=CTensor<type_t>(1,neurons,1);

 if (prev_layer_ptr==NULL)//слой без предшествующего считается входным
 {
  cTensor_W=CTensor<type_t>(1,1,1);
  cTensor_B=CTensor<type_t>(1,1,1);
 }
 else
 {
  size_t size_x=PrevLayerPtr->GetOutputTensor(0).GetSizeX();
  size_t size_y=PrevLayerPtr->GetOutputTensor(0).GetSizeY();
  size_t size_z=PrevLayerPtr->GetOutputTensor(0).GetSizeZ();

  cTensor_W=CTensor<type_t>(1,neurons,size_x*size_y*size_z);
  cTensor_B=CTensor<type_t>(1,neurons,1);
  //задаём предшествующему слою, что мы его последующий слой
  prev_layer_ptr->SetNextLayerPtr(this);
 }
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию весов и сдвигов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::Reset(void)
{
 if (PrevLayerPtr==NULL) return;

 type_t size=static_cast<type_t>(cTensor_W.GetSizeX());//нормируется только по количеству входов нейрона
 type_t koeff=static_cast<type_t>(sqrt(6.0/size));
 CTensor<type_t> cTensor_Rand(1,1,size);
 //веса
 for(size_t y=0;y<cTensor_W.GetSizeY();y++)
 {
  //CRandom<type_t>::SetRandomNormal(cTensor_Rand,-koeff,koeff);
  //CRandom<type_t>::SetRandomNormal(cTensor_Rand);
  for(size_t x=0;x<cTensor_W.GetSizeX();x++)
  {
   //используем метод инициализации He (Ге)
   type_t rnd=static_cast<type_t>(CRandom<type_t>::GetRandValue(2.0)-1.0);
   type_t init=rnd*koeff;
   //type_t init=cTensor_Rand.GetElement(0,0,x);
   cTensor_W.SetElement(0,y,x,init);
  }
 }
 //сдвиги
 size=static_cast<type_t>(cTensor_B.GetSizeY());
 koeff=static_cast<type_t>(sqrt(2.0/size));
 for(size_t y=0;y<cTensor_B.GetSizeY();y++)
 {
  //используем метод инициализации He (Ге)
  //type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
  type_t init=0.1;//rnd*koeff;
  cTensor_B.SetElement(0,y,0,init);
 }
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output)
{
 //if (output.GetSizeX()!=cTensor_H_Array[unit_index].GetSizeX()) throw("void CNetLayerLinear<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeY()!=cTensor_H_Array[unit_index].GetSizeY()) throw("void CNetLayerLinear<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeZ()!=cTensor_H_Array[unit_index].GetSizeZ()) throw("void CNetLayerLinear<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности матрицы output!");
 cTensor_H_Array[unit_index]=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H_Array[unit_index].GetSizeX()) throw("void CNetLayerLinear<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeY()!=cTensor_H_Array[unit_index].GetSizeY()) throw("void CNetLayerLinear<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeZ()!=cTensor_H_Array[unit_index].GetSizeZ()) throw("void CNetLayerLinear<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности матрицы output!");
 output=cTensor_H_Array[unit_index];
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::Forward(void)
{
 if (PrevLayerPtr==NULL) return;//для входного слоя ничего делать не нужно

 for(size_t n=0;n<BatchSize;n++)
 {
  //Z=WxZprev
  //приводим входной тензор к линии
  size_t size_x=PrevLayerPtr->GetOutputTensor(n).GetSizeX();
  size_t size_y=PrevLayerPtr->GetOutputTensor(n).GetSizeY();
  size_t size_z=PrevLayerPtr->GetOutputTensor(n).GetSizeZ();
  PrevLayerPtr->GetOutputTensor(n).ReinterpretSize(1,size_x*size_y*size_z,1);

  CTensorMath<type_t>::Mul(cTensor_H_Array[n],cTensor_W,PrevLayerPtr->GetOutputTensor(n));

  PrevLayerPtr->GetOutputTensor(n).RestoreSize();

  CTensorMath<type_t>::Add(cTensor_H_Array[n],cTensor_H_Array[n],cTensor_B);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerLinear<type_t>::GetOutputTensor(size_t unit_index)
{
 return(cTensor_H_Array[unit_index]);
}
//----------------------------------------------------------------------------------------------------
/*!задать указатель на последующий слой
\param[in] next_layer_ptr Указатель на последующий слой
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerLinear<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 cTensor_W.Save(iDataStream_Ptr);
 cTensor_B.Save(iDataStream_Ptr);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerLinear<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 cTensor_W.Load(iDataStream_Ptr);
 cTensor_B.Load(iDataStream_Ptr);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerLinear<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
{
 cTensor_MW.Save(iDataStream_Ptr);
 cTensor_VW.Save(iDataStream_Ptr);
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
bool CNetLayerLinear<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 cTensor_MW.Load(iDataStream_Ptr);
 cTensor_VW.Load(iDataStream_Ptr);
 cTensor_MB.Load(iDataStream_Ptr);
 cTensor_VB.Load(iDataStream_Ptr);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 cTensor_dW=CTensor<type_t>(1,cTensor_W.GetSizeY(),cTensor_W.GetSizeX());
 cTensor_TmpdW=CTensor<type_t>(1,cTensor_W.GetSizeY(),cTensor_W.GetSizeX());
 cTensor_dB=CTensor<type_t>(1,cTensor_B.GetSizeY(),cTensor_B.GetSizeX());

 cTensor_Delta_Array.resize(BatchSize);
 for(size_t n=0;n<BatchSize;n++) cTensor_Delta_Array[n]=CTensor<type_t>(1,cTensor_H_Array[n].GetSizeY(),cTensor_H_Array[n].GetSizeX());

 //для алгоритма Adam
 cTensor_MW=CTensor<type_t>(1,cTensor_W.GetSizeY(),cTensor_W.GetSizeX());
 cTensor_VW=CTensor<type_t>(1,cTensor_W.GetSizeY(),cTensor_W.GetSizeX());
 cTensor_MB=CTensor<type_t>(1,cTensor_B.GetSizeY(),cTensor_B.GetSizeX());
 cTensor_VB=CTensor<type_t>(1,cTensor_B.GetSizeY(),cTensor_B.GetSizeX());

 cTensor_MW.Zero();
 cTensor_VW.Zero();
 cTensor_MB.Zero();
 cTensor_VB.Zero();

 if (PrevLayerPtr!=NULL)
 {
  CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor(0);
  cTensor_PrevLayerError=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 }
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_dW=CTensor<type_t>(1,1,1);
 cTensor_TmpdW=CTensor<type_t>(1,1,1);
 cTensor_dB=CTensor<type_t>(1,1,1);
 cTensor_PrevLayerError=CTensor<type_t>(1,1,1);

 cTensor_MW=CTensor<type_t>(1,1,1);
 cTensor_VW=CTensor<type_t>(1,1,1);
 cTensor_MB=CTensor<type_t>(1,1,1);
 cTensor_VB=CTensor<type_t>(1,1,1);

 cTensor_Delta_Array.clear();
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::TrainingBackward(bool create_delta_weight)
{
 //вычисляем ошибку предыдущего слоя (D=tr(W)xDnext)
 if (PrevLayerPtr!=NULL)//это не входной слой
 {
  for(size_t n=0;n<BatchSize;n++)
  {
   //считаем ошибку предыдущего слоя
   size_t size_x=cTensor_PrevLayerError.GetSizeX();
   size_t size_y=cTensor_PrevLayerError.GetSizeY();
   size_t size_z=cTensor_PrevLayerError.GetSizeZ();

   cTensor_PrevLayerError.ReinterpretSize(1,size_x*size_y*size_z,1);
   CTensorMath<type_t>::TransponseMul(cTensor_PrevLayerError,cTensor_W,cTensor_Delta_Array[n]);
   cTensor_PrevLayerError.RestoreSize();
   //задаём ошибку предыдущего слоя
   PrevLayerPtr->SetOutputError(n,cTensor_PrevLayerError);

   if (create_delta_weight==true)
   {
    CTensor<type_t> &h=PrevLayerPtr->GetOutputTensor(n);
    size_x=h.GetSizeX();
    size_y=h.GetSizeY();
    size_z=h.GetSizeZ();
    h.ReinterpretSize(1,1,size_x*size_y*size_z);

    CTensorMath<type_t>::Mul(cTensor_TmpdW,cTensor_Delta_Array[n],h);

    h.RestoreSize();

    CTensorMath<type_t>::Add(cTensor_dW,cTensor_dW,cTensor_TmpdW);
    CTensorMath<type_t>::Add(cTensor_dB,cTensor_dB,cTensor_Delta_Array[n]);
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::TrainingResetDeltaWeight(void)
{
 cTensor_dW.Zero();
 cTensor_dB.Zero();
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::TrainingUpdateWeight(double speed,double iteration)
{
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_ADAM)
 {
  //применяем алгоритм Adam
  CTensorMath<type_t>::Adam(cTensor_W,cTensor_dW,cTensor_MW,cTensor_VW,BatchSize,speed,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_B,cTensor_dB,cTensor_MB,cTensor_VB,BatchSize,speed,Beta1,Beta2,Epsilon,iteration);
 }
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_GRADIENT)
 {
  speed/=static_cast<double>(BatchSize);
  CTensorMath<type_t>::Sub(cTensor_W,cTensor_W,cTensor_dW,1,speed);
  CTensorMath<type_t>::Sub(cTensor_B,cTensor_B,cTensor_dB,1,speed);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerLinear<type_t>::GetDeltaTensor(size_t unit_index)
{
 return(cTensor_Delta_Array[unit_index]);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::SetOutputError(size_t unit_index,CTensor<type_t>& error)
{
 cTensor_Delta_Array[unit_index]=error;
}

//----------------------------------------------------------------------------------------------------
/*!ограничить веса в диапазон
\param[in] min Минимальное значение веса
\param[in] max Максимальное значение веса
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::ClipWeight(type_t min,type_t max)
{
 CTensorMath<type_t>::Clip(cTensor_W,min,max);
 CTensorMath<type_t>::Clip(cTensor_B,min,max);
}
#endif
