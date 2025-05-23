#ifndef C_NET_LAYER_DROP_OUT_H
#define C_NET_LAYER_DROP_OUT_H

//****************************************************************************************************
//\file Вероятностное исключение нейронов линейного слоя из обучения
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdio.h>
#include <fstream>

#include "../common/idatastream.h"
#include "../common/crandom.h"
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
//!Вероятностное исключение нейронов линейного слоя из обучения
//****************************************************************************************************
template<class type_t>
class CNetLayerDropOut:public INetLayer<type_t>
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

  double DropOut;///<вероятность исключения нейронов
  bool Training;///<включён ли режим обучения
  //тензоры, используемые при обучении
  std::vector<CTensor<type_t>> cTensor_H_Array;///<тензоры выхода слоя
  std::vector<CTensor<type_t>> cTensor_H_DropOut_Array;///<тензоры исключения выхода слоя
  std::vector<CTensor<type_t>> cTensor_Delta_Array;///<тензоры ошибки предыдущего слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerDropOut(double drop_out,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);
  CNetLayerDropOut(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerDropOut();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(double drop_out,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);///<создать слой
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
CNetLayerDropOut<type_t>::CNetLayerDropOut(double drop_out,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 Create(drop_out,prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerDropOut<type_t>::CNetLayerDropOut(void)
{
 Create(1);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerDropOut<type_t>::~CNetLayerDropOut()
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
void CNetLayerDropOut<type_t>::Create(double drop_out,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 DropOut=drop_out;
 Training=false;

 if (prev_layer_ptr==NULL)//слой без предшествующего считается входным
 {
  throw("Слой DropOut не может быть входным!");
 }
 //задаём предшествующему слою, что мы его последующий слой
 prev_layer_ptr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию весов и сдвигов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::Reset(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output)
{
 throw("Слою DropOut нельзя задать выходное значение!");
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output)
{
 throw("От слоя DropOut нельзя получить выходное значение!");
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::Forward(void)
{
 if (Training==false) return;
 //создаём матрицу исключения

 for(size_t n=0;n<BatchSize;n++)
 {
  type_t mult=static_cast<type_t>(1.0/(1.0-DropOut));

  size_t size_x=cTensor_H_DropOut_Array[n].GetSizeX();
  size_t size_y=cTensor_H_DropOut_Array[n].GetSizeY();
  size_t size_z=cTensor_H_DropOut_Array[n].GetSizeZ();
  cTensor_H_DropOut_Array[n].Zero();
  for(size_t z=0;z<size_z;z++)
  {
   for(size_t y=0;y<size_y;y++)
   {
    for(size_t x=0;x<size_x;x++)
    {
     if (CRandom<type_t>::GetRandValue(1)>=DropOut) cTensor_H_DropOut_Array[n].SetElement(z,y,x,mult);
    }
   }
  }
  //умножаем входной тензор на тензор исключения поэлементно
  CTensorMath<type_t>::TensorItemProduction(cTensor_H_Array[n],PrevLayerPtr->GetOutputTensor(n),cTensor_H_DropOut_Array[n]);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerDropOut<type_t>::GetOutputTensor(size_t unit_index)
{
 if (Training==true) return(cTensor_H_Array[unit_index]);
 return(PrevLayerPtr->GetOutputTensor(unit_index));
}
//----------------------------------------------------------------------------------------------------
/*!задать указатель на последующий слой
\param[in] next_layer_ptr Указатель на последующий слой
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerDropOut<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveDouble(DropOut);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerDropOut<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 DropOut=iDataStream_Ptr->LoadDouble();
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerDropOut<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
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
bool CNetLayerDropOut<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::TrainingStart(void)
{
 Training=true;
 //создаём все вспомогательные тензоры
 cTensor_H_Array.resize(BatchSize);
 cTensor_H_DropOut_Array.resize(BatchSize);
 cTensor_Delta_Array.resize(BatchSize);

 CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor(0);
 for(size_t n=0;n<BatchSize;n++)
 {
  cTensor_H_Array[n]=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
  cTensor_H_DropOut_Array[n]=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
  cTensor_Delta_Array[n]=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 }
 cTensor_PrevLayerError=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::TrainingStop(void)
{
 Training=false;
 //удаляем все вспомогательные тензоры
 cTensor_H_Array.clear();
 cTensor_H_DropOut_Array.clear();
 cTensor_Delta_Array.clear();
 cTensor_PrevLayerError=CTensor<type_t>(1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::TrainingBackward(bool create_delta_weight)
{
 for(size_t n=0;n<BatchSize;n++)
 {
  //умножаем тензор ошибки на тензор исключения поэлементно
  CTensorMath<type_t>::TensorItemProduction(cTensor_PrevLayerError,cTensor_Delta_Array[n],cTensor_H_DropOut_Array[n]);
  //задаём ошибку предыдущего слоя
  PrevLayerPtr->SetOutputError(n,cTensor_PrevLayerError);
 }
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::TrainingUpdateWeight(double speed,double iteration)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerDropOut<type_t>::GetDeltaTensor(size_t unit_index)
{
 return(cTensor_Delta_Array[unit_index]);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::SetOutputError(size_t unit_index,CTensor<type_t>& error)
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
void CNetLayerDropOut<type_t>::ClipWeight(type_t min,type_t max)
{
}

#endif
