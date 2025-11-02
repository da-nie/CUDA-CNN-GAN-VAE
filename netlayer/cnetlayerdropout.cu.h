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

  uint32_t BatchSize;///<размер пакета для обучения

  double DropOut;///<вероятность исключения нейронов
  bool Training;///<включён ли режим обучения
  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_H;///<тензоры выхода слоя
  CTensor<type_t> cTensor_H_DropOut;///<тензоры исключения выхода слоя
  CTensor<type_t> cTensor_Delta;///<тензоры ошибки предыдущего слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerDropOut(double drop_out,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerDropOut(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerDropOut();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(double drop_out,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
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
CNetLayerDropOut<type_t>::CNetLayerDropOut(double drop_out,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
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
\param[in] drop_out Коэффициент "выбрасывания" нейронов
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::Create(double drop_out,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
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
void CNetLayerDropOut<type_t>::SetOutput(CTensor<type_t> &output)
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
void CNetLayerDropOut<type_t>::GetOutput(CTensor<type_t> &output)
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
 CTensorMath<type_t>::Fill(cTensor_H_DropOut,0);
 type_t mult=static_cast<type_t>(1.0/(1.0-DropOut));
 uint32_t size_x=cTensor_H_DropOut.GetSizeX();
 uint32_t size_y=cTensor_H_DropOut.GetSizeY();
 uint32_t size_z=cTensor_H_DropOut.GetSizeZ();
 for(uint32_t w=0;w<BatchSize;w++)
 {
  for(uint32_t z=0;z<size_z;z++)
  {
   for(uint32_t y=0;y<size_y;y++)
   {
    for(uint32_t x=0;x<size_x;x++)
    {
     if (CRandom<type_t>::GetRandValue(1)>=DropOut) cTensor_H_DropOut.SetElement(w,z,y,x,mult);
    }
   }
  }
  //умножаем входной тензор на тензор исключения поэлементно
  CTensorMath<type_t>::TensorItemProduction(cTensor_H,PrevLayerPtr->GetOutputTensor(),cTensor_H_DropOut);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerDropOut<type_t>::GetOutputTensor(void)
{
 if (Training==true) return(cTensor_H);
 return(PrevLayerPtr->GetOutputTensor());
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
bool CNetLayerDropOut<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
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
 CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor();
 cTensor_H=CTensor<type_t>(BatchSize,prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_H_DropOut=CTensor<type_t>(BatchSize,prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_Delta=CTensor<type_t>(BatchSize,prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_PrevLayerError=CTensor<type_t>(BatchSize,prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
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
 cTensor_H=CTensor<type_t>(1,1,1,1);
 cTensor_H_DropOut=CTensor<type_t>(1,1,1,1);
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
 cTensor_PrevLayerError=CTensor<type_t>(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::TrainingBackward(bool create_delta_weight)
{
 //умножаем тензор ошибки на тензор исключения поэлементно
 CTensorMath<type_t>::TensorItemProduction(cTensor_PrevLayerError,cTensor_Delta,cTensor_H_DropOut);
 //задаём ошибку предыдущего слоя
 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
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
void CNetLayerDropOut<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerDropOut<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::SetOutputError(CTensor<type_t>& error)
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
void CNetLayerDropOut<type_t>::ClipWeight(type_t min,type_t max)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" DropOut: input",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerDropOut<type_t>::PrintOutputTensorSize(const std::string &name)
{
 GetOutputTensor().Print(name+" DropOut: output",false);
}

#endif
