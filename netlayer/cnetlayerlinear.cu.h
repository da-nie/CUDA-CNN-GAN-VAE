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

  uint32_t BatchSize;///<размер пакета для обучения

  CTensor<type_t> cTensor_W;///<тензор весов слоя
  CTensor<type_t> cTensor_B;///<тензор сдвигов слоя

  CTensor<type_t> cTensor_H;///<тензор значений нейронов до функции активации

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_dW;///<тензор поправок весов слоя
  CTensor<type_t> cTensor_dB;///<тензор поправок сдвигов слоя
  CTensor<type_t> cTensor_Delta;///<тензор дельты слоя
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
  CNetLayerLinear(uint32_t neurons,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerLinear(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerLinear();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(uint32_t neurons,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
  void Reset(void);///<выполнить инициализацию весов и сдвигов
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
CNetLayerLinear<type_t>::CNetLayerLinear(uint32_t neurons,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
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
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::Create(uint32_t neurons,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 cTensor_H=CTensor<type_t>(BatchSize,1,neurons,1);

 if (prev_layer_ptr==NULL)//слой без предшествующего считается входным
 {
  cTensor_W=CTensor<type_t>(1,1,1,1);
  cTensor_B=CTensor<type_t>(1,1,1,1);
 }
 else
 {
  uint32_t size_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
  uint32_t size_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
  uint32_t size_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();

  cTensor_W=CTensor<type_t>(1,1,neurons,size_x*size_y*size_z);
  cTensor_B=CTensor<type_t>(1,1,neurons,1);
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
 for(uint32_t y=0;y<cTensor_W.GetSizeY();y++)
 {
  //CRandom<type_t>::SetRandomNormal(cTensor_Rand,-koeff,koeff);
  //CRandom<type_t>::SetRandomNormal(cTensor_Rand);
  for(uint32_t x=0;x<cTensor_W.GetSizeX();x++)
  {
   //используем метод инициализации He (Ге)
   type_t rnd=static_cast<type_t>(CRandom<type_t>::GetRandValue(2.0)-1.0);
   type_t init=rnd*koeff;
   //type_t init=cTensor_Rand.GetElement(0,0,x);
   cTensor_W.SetElement(0,0,y,x,init);
  }
 }
 //сдвиги
 size=static_cast<type_t>(cTensor_B.GetSizeY());
 koeff=static_cast<type_t>(sqrt(2.0/size));
 for(uint32_t y=0;y<cTensor_B.GetSizeY();y++)
 {
  //используем метод инициализации He (Ге)
  //type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
  type_t init=0.1;//rnd*koeff;
  cTensor_B.SetElement(0,0,y,0,init);
 }
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::SetOutput(CTensor<type_t> &output)
{
 //if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerLinear<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerLinear<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerLinear<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerLinear<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerLinear<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerLinear<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerLinear<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerLinear<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::Forward(void)
{
 if (PrevLayerPtr==NULL) return;//для входного слоя ничего делать не нужно

 //Z=WxZprev
 //приводим входной тензор к линии по X*Y*Z
 uint32_t size_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 uint32_t size_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 uint32_t size_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(BatchSize,1,size_x*size_y*size_z,1);

 CTensorMath<type_t>::Mul(cTensor_H,cTensor_W,PrevLayerPtr->GetOutputTensor());

 PrevLayerPtr->GetOutputTensor().RestoreSize();

 CTensorMath<type_t>::Add(cTensor_H,cTensor_H,cTensor_B);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerLinear<type_t>::GetOutputTensor(void)
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
bool CNetLayerLinear<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 cTensor_W.Load(iDataStream_Ptr,check_size);
 cTensor_B.Load(iDataStream_Ptr,check_size);
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
 cTensor_dW=CTensor<type_t>(1,1,cTensor_W.GetSizeY(),cTensor_W.GetSizeX());
 cTensor_TmpdW=CTensor<type_t>(BatchSize,1,cTensor_W.GetSizeY(),cTensor_W.GetSizeX());
 cTensor_dB=CTensor<type_t>(1,1,cTensor_B.GetSizeY(),cTensor_B.GetSizeX());

 cTensor_Delta=CTensor<type_t>(BatchSize,1,cTensor_H.GetSizeY(),cTensor_H.GetSizeX());

 //для алгоритма Adam
 cTensor_MW=CTensor<type_t>(1,1,cTensor_W.GetSizeY(),cTensor_W.GetSizeX());
 cTensor_VW=CTensor<type_t>(1,1,cTensor_W.GetSizeY(),cTensor_W.GetSizeX());
 cTensor_MB=CTensor<type_t>(1,1,cTensor_B.GetSizeY(),cTensor_B.GetSizeX());
 cTensor_VB=CTensor<type_t>(1,1,cTensor_B.GetSizeY(),cTensor_B.GetSizeX());

 CTensorMath<type_t>::Fill(cTensor_MW,0);
 CTensorMath<type_t>::Fill(cTensor_VW,0);
 CTensorMath<type_t>::Fill(cTensor_MB,0);
 CTensorMath<type_t>::Fill(cTensor_VB,0);

 if (PrevLayerPtr!=NULL)
 {
  CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor();
  cTensor_PrevLayerError=CTensor<type_t>(BatchSize,prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
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
 cTensor_dW=CTensor<type_t>(1,1,1,1);
 cTensor_TmpdW=CTensor<type_t>(1,1,1,1);
 cTensor_dB=CTensor<type_t>(1,1,1,1);
 cTensor_PrevLayerError=CTensor<type_t>(1,1,1,1);

 cTensor_MW=CTensor<type_t>(1,1,1,1);
 cTensor_VW=CTensor<type_t>(1,1,1,1);
 cTensor_MB=CTensor<type_t>(1,1,1,1);
 cTensor_VB=CTensor<type_t>(1,1,1,1);

 cTensor_Delta=CTensor<type_t>(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::TrainingBackward(bool create_delta_weight)
{
 if (PrevLayerPtr==NULL) return;//это входной слой

 //вычисляем ошибку предыдущего слоя (D=tr(W)xDnext)
 //считаем ошибку предыдущего слоя
 uint32_t size_x=cTensor_PrevLayerError.GetSizeX();
 uint32_t size_y=cTensor_PrevLayerError.GetSizeY();
 uint32_t size_z=cTensor_PrevLayerError.GetSizeZ();

 cTensor_PrevLayerError.ReinterpretSize(BatchSize,1,size_x*size_y*size_z,1);
 CTensorMath<type_t>::TransponseMul(cTensor_PrevLayerError,cTensor_W,cTensor_Delta);
 cTensor_PrevLayerError.RestoreSize();
 //задаём ошибку предыдущего слоя

 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);

 if (create_delta_weight==true)
 {
  CTensor<type_t> &h=PrevLayerPtr->GetOutputTensor();
  size_x=h.GetSizeX();
  size_y=h.GetSizeY();
  size_z=h.GetSizeZ();
  h.ReinterpretSize(BatchSize,1,1,size_x*size_y*size_z);

  CTensorMath<type_t>::Mul(cTensor_TmpdW,cTensor_Delta,h);

  h.RestoreSize();

  CTensorMath<type_t>::AddSumW(cTensor_dW,cTensor_dW,cTensor_TmpdW);
  CTensorMath<type_t>::AddSumW(cTensor_dB,cTensor_dB,cTensor_Delta);
 }
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::TrainingResetDeltaWeight(void)
{
 CTensorMath<type_t>::Fill(cTensor_dW,0);
 CTensorMath<type_t>::Fill(cTensor_dB,0);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_ADAM)
 {
  //применяем алгоритм Adam
  CTensorMath<type_t>::Adam(cTensor_W,cTensor_dW,cTensor_MW,cTensor_VW,BatchSize*batch_scale,speed,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_B,cTensor_dB,cTensor_MB,cTensor_VB,BatchSize*batch_scale,speed,Beta1,Beta2,Epsilon,iteration);
 }
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_GRADIENT)
 {
  speed/=static_cast<double>(BatchSize);
  CTensorMath<type_t>::Sub(cTensor_W,cTensor_W,cTensor_dW,1,speed/batch_scale);
  CTensorMath<type_t>::Sub(cTensor_B,cTensor_B,cTensor_dB,1,speed/batch_scale);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerLinear<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::SetOutputError(CTensor<type_t>& error)
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
void CNetLayerLinear<type_t>::ClipWeight(type_t min,type_t max)
{
 CTensorMath<type_t>::Clip(cTensor_W,min,max);
 CTensorMath<type_t>::Clip(cTensor_B,min,max);
}

//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" Linear: input",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::PrintOutputTensorSize(const std::string &name)
{
 GetOutputTensor().Print(name+" Linear: output",false);
}

#endif
