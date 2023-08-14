#ifndef C_NET_LAYER_BATCH_NORMALIZATION_H
#define C_NET_LAYER_BATCH_NORMALIZATION_H

//****************************************************************************************************
//\file Пакетная нормализация
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
//!Пакетная нормализация
//****************************************************************************************************
template<class type_t>
class CNetLayerBatchNormalization:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  type_t Beta;///<параметр сдвига
  type_t Gamma;///<параметр масштабирования

  CTensor<type_t> cTensor_H;///<тензор выхода слоя

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<тензор ошибки предыдущего слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerBatchNormalization(INetLayer<type_t> *prev_layer_ptr=NULL);
  CNetLayerBatchNormalization(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerBatchNormalization();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(INetLayer<type_t> *prev_layer_ptr=NULL);///<создать слой
  void Reset(void);///<выполнить инициализацию весов и сдвигов
  void SetOutput(CTensor<type_t> &output);///<задать выход слоя
  void GetOutput(CTensor<type_t> &output);///<получить выход слоя
  void Forward(void);///<выполнить прямой проход по слою
  CTensor<type_t>& GetOutputTensor(void);///<получить ссылку на выходной тензор
  void SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr);///<задать указатель на последующий слой
  bool Save(IDataStream *iDataStream_Ptr);///<сохранить параметры слоя
  bool Load(IDataStream *iDataStream_Ptr);///<загрузить параметры слоя

  void TrainingStart(void);///<начать процесс обучения
  void TrainingStop(void);///<завершить процесс обучения
  void TrainingBackward(void);///<выполнить обратный проход по сети для обучения
  void TrainingResetDeltaWeight(void);///<сбросить поправки к весам
  void TrainingUpdateWeight(double speed);///<выполнить обновления весов
  CTensor<type_t>& GetDeltaTensor(void);///<получить ссылку на тензор дельты слоя

  void SetOutputError(CTensor<type_t>& error);///<задать ошибку и расчитать дельту

  void ClipWeight(type_t min,type_t max);///<ограничить веса в диапазон
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  type_t GetRandValue(type_t max_value);///<получить случайное число
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerBatchNormalization<type_t>::CNetLayerBatchNormalization(INetLayer<type_t> *prev_layer_ptr)
{
 Create(prev_layer_ptr);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerBatchNormalization<type_t>::CNetLayerBatchNormalization(void)
{
 Create();
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerBatchNormalization<type_t>::~CNetLayerBatchNormalization()
{
}

//****************************************************************************************************
//закрытые функции
//****************************************************************************************************


//----------------------------------------------------------------------------------------------------
/*!получить случайное число
\param[in] max_value Максимальное значение случайного числа
\return Случайное число в диапазоне [0...max_value]
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CNetLayerBatchNormalization<type_t>::GetRandValue(type_t max_value)
{
 return((static_cast<type_t>(rand())*max_value)/static_cast<type_t>(RAND_MAX));
}

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
void CNetLayerBatchNormalization<type_t>::Create(INetLayer<type_t> *prev_layer_ptr)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 cTensor_H=CTensor<type_t>(PrevLayerPtr->GetOutputTensor().GetSizeZ(),PrevLayerPtr->GetOutputTensor().GetSizeY(),PrevLayerPtr->GetOutputTensor().GetSizeX());
 if (prev_layer_ptr==NULL)//слой без предшествующего считается входным
 {
  throw("Слой BatchNormalization не может быть входным!");
 }
 //задаём предшествующему слою, что мы его последующий слой
 prev_layer_ptr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию весов и сдвигов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::Reset(void)
{
 Beta=0;
 Gamma=1;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::SetOutput(CTensor<type_t> &output)
{
 throw("Слою BatchNormalization нельзя задать выходное значение!");
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::GetOutput(CTensor<type_t> &output)
{
 throw("От слоя BatchNormalization нельзя получить выходное значение!");
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::Forward(void)
{
 static const double EPSILON=0.001;

 CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();

 size_t input_x=input.GetSizeX();
 size_t input_y=input.GetSizeY();
 size_t input_z=input.GetSizeZ();
 //считаем среднее
 double middle=0;
 for(size_t z=0;z<input_z;z++)
 {
  for(size_t y=0;y<input_y;y++)
  {
   for(size_t x=0;x<input_x;x++)
   {
	middle+=input.GetElement(z,y,x);
   }
  }
 }
 middle/=static_cast<double>(input_x*input_y*input_z);
 //считаем стандартное отклонение по пакету (дисперсию)
 double sigma=0;
 for(size_t z=0;z<input_z;z++)
 {
  for(size_t y=0;y<input_y;y++)
  {
   for(size_t x=0;x<input_x;x++)
   {
    type_t e=input.GetElement(z,y,x);
	sigma+=(e-middle)*(e-middle);
   }
  }
 }
 sigma/=static_cast<double>(input_x*input_y*input_z);
 //выполняем нормализацию
 double k=1.0/sqrt(sigma+EPSILON);
 for(size_t z=0;z<input_z;z++)
 {
  for(size_t y=0;y<input_y;y++)
  {
   for(size_t x=0;x<input_x;x++)
   {
	type_t e=input.GetElement(z,y,x);
	e=(e-middle)/k;//нормировка
	e=e*Gamma+Beta;//сдвиг и масштабирование
	cTensor_H.SetElement(z,y,x,e);
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerBatchNormalization<type_t>::GetOutputTensor(void)
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
void CNetLayerBatchNormalization<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerBatchNormalization<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveDouble(Beta);
 iDataStream_Ptr->SaveDouble(Gamma);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerBatchNormalization<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 Beta=iDataStream_Ptr->LoadDouble();
 Gamma=iDataStream_Ptr->LoadDouble();
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor();
 cTensor_Delta=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_PrevLayerError=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1);
 cTensor_PrevLayerError=CTensor<type_t>(1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingBackward(void)
{
 //умножаем тензор ошибки на тензор исключения поэлементно
 //CTensorMath<type_t>::TensorItemProduction(cTensor_PrevLayerError,cTensor_Delta,cTensor_H_BatchNormalization);
 //задаём ошибку предыдущего слоя
 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingUpdateWeight(double speed)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerBatchNormalization<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::SetOutputError(CTensor<type_t>& error)
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
void CNetLayerBatchNormalization<type_t>::ClipWeight(type_t min,type_t max)
{
}

#endif
