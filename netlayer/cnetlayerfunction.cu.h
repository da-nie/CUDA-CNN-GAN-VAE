#ifndef C_NET_LAYER_FUNCTION_H
#define C_NET_LAYER_FUNCTION_H

//****************************************************************************************************
//\file Функции активации
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
//!Функции активации
//****************************************************************************************************
template<class type_t>
class CNetLayerFunction:public INetLayer<type_t>
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

  NNeuron::NEURON_FUNCTION NeuronFunction;///<функция активации нейронов слоя

  std::vector<CTensor<type_t>> cTensor_H_Array;///<тензоры значений нейронов после функции активации
  //тензоры, используемые при обучении
  std::vector<CTensor<type_t>> cTensor_Delta_Array;///<тензоры дельты слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerFunction(NNeuron::NEURON_FUNCTION neuron_function=NNeuron::NEURON_FUNCTION_SIGMOID,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerFunction(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerFunction();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(NNeuron::NEURON_FUNCTION neuron_function=NNeuron::NEURON_FUNCTION_SIGMOID,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
  void Reset(void);///<выполнить инициализацию весов и сдвигов
  void SetOutput(uint32_t unit_index,CTensor<type_t> &output);///<задать выход слоя
  void GetOutput(uint32_t unit_index,CTensor<type_t> &output);///<получить выход слоя
  void Forward(void);///<выполнить прямой проход по слою
  CTensor<type_t>& GetOutputTensor(uint32_t unit_index);///<получить ссылку на выходной тензор
  void SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr);///<задать указатель на последующий слой
  bool Save(IDataStream *iDataStream_Ptr);///<сохранить параметры слоя
  bool Load(IDataStream *iDataStream_Ptr,bool check_size);///<загрузить параметры слоя
  bool SaveTrainingParam(IDataStream *iDataStream_Ptr);///<сохранить параметры обучения слоя
  bool LoadTrainingParam(IDataStream *iDataStream_Ptr);///<загрузить параметры обучения слоя

  void TrainingStart(void);///<начать процесс обучения
  void TrainingStop(void);///<завершить процесс обучения
  void TrainingBackward(bool create_delta_weight=true);///<выполнить обратный проход по сети для обучения
  void TrainingResetDeltaWeight(void);///<сбросить поправки к весам
  void TrainingUpdateWeight(double speed,double iteration,double batch_scale=1);///<выполнить обновления весов
  CTensor<type_t>& GetDeltaTensor(uint32_t unit_index);///<получить ссылку на тензор дельты слоя

  void SetOutputError(uint32_t unit_index,CTensor<type_t>& error);///<задать ошибку и расчитать дельту

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
CNetLayerFunction<type_t>::CNetLayerFunction(NNeuron::NEURON_FUNCTION neuron_function,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 Create(neuron_function,prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerFunction<type_t>::CNetLayerFunction(void)
{
 Create(1);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerFunction<type_t>::~CNetLayerFunction()
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
\param[in] neuron_function Функция активации нейронов
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::Create(NNeuron::NEURON_FUNCTION neuron_function,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;
 NeuronFunction=neuron_function;

 BatchSize=batch_size;

 if (prev_layer_ptr==NULL)//слой без предшествующего считается входным
 {
  throw("Слой функций активации не может быть входным");
 }
 else
 {
  cTensor_H_Array.resize(BatchSize);
  for(uint32_t n=0;n<BatchSize;n++) cTensor_H_Array[n]=PrevLayerPtr->GetOutputTensor(n);
  //задаём предшествующему слою, что мы его последующий слой
  prev_layer_ptr->SetNextLayerPtr(this);
 }
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию весов и сдвигов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::Reset(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::SetOutput(uint32_t unit_index,CTensor<type_t> &output)
{
 cTensor_H_Array[unit_index].CopyItem(output);
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::GetOutput(uint32_t unit_index,CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H_Array[unit_index].GetSizeX()) throw("void CNetLayerFunction<type_t>::GetOutput(uint32_t unit_index,CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeY()!=cTensor_H_Array[unit_index].GetSizeY()) throw("void CNetLayerFunction<type_t>::GetOutput(uint32_t unit_index,CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeZ()!=cTensor_H_Array[unit_index].GetSizeZ()) throw("void CNetLayerFunction<type_t>::GetOutput(uint32_t unit_index,CTensor<type_t> &output) - ошибка размерности матрицы output!");
 output=cTensor_H_Array[unit_index];
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::Forward(void)
{
 for(uint32_t n=0;n<BatchSize;n++)
 {
  //применим функцию активации
  if (NeuronFunction==NNeuron::NEURON_FUNCTION_SIGMOID) CTensorApplyFunc<type_t>::ApplySigmoid(cTensor_H_Array[n],PrevLayerPtr->GetOutputTensor(n));
  if (NeuronFunction==NNeuron::NEURON_FUNCTION_RELU) CTensorApplyFunc<type_t>::ApplyReLU(cTensor_H_Array[n],PrevLayerPtr->GetOutputTensor(n));
  if (NeuronFunction==NNeuron::NEURON_FUNCTION_GELU) CTensorApplyFunc<type_t>::ApplyGeLU(cTensor_H_Array[n],PrevLayerPtr->GetOutputTensor(n));
  if (NeuronFunction==NNeuron::NEURON_FUNCTION_LEAKY_RELU) CTensorApplyFunc<type_t>::ApplyLeakyReLU(cTensor_H_Array[n],PrevLayerPtr->GetOutputTensor(n));
  if (NeuronFunction==NNeuron::NEURON_FUNCTION_LINEAR) CTensorApplyFunc<type_t>::ApplyLinear(cTensor_H_Array[n],PrevLayerPtr->GetOutputTensor(n));
  if (NeuronFunction==NNeuron::NEURON_FUNCTION_TANGENCE) CTensorApplyFunc<type_t>::ApplyTangence(cTensor_H_Array[n],PrevLayerPtr->GetOutputTensor(n));
  //if (NeuronFunction==NNeuron::NEURON_FUNCTION_SOFTMAX) CTensorApplyFunc<type_t>::ApplySoftMax(cTensor_H_Array[n],PrevLayerPtr->GetOutputTensor(n));
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerFunction<type_t>::GetOutputTensor(uint32_t unit_index)
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
void CNetLayerFunction<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerFunction<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveUInt32(NeuronFunction);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerFunction<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 NeuronFunction=iDataStream_Ptr->LoadUInt32();
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerFunction<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
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
bool CNetLayerFunction<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::TrainingStart(void)
{
 cTensor_Delta_Array.resize(BatchSize);
 for(uint32_t n=0;n<BatchSize;n++) cTensor_Delta_Array[n]=cTensor_H_Array[n];
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta_Array.clear();
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::TrainingBackward(bool create_delta_weight)
{
 //вычисляем ошибку предыдущего слоя (D=tr(W)xDnext)
 if (PrevLayerPtr!=NULL)//это не входной слой
 {
  //задаём ошибку предыдущего слоя
  for(uint32_t n=0;n<BatchSize;n++) PrevLayerPtr->SetOutputError(n,cTensor_Delta_Array[n]);
 }
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerFunction<type_t>::GetDeltaTensor(uint32_t unit_index)
{
 return(cTensor_Delta_Array[unit_index]);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::SetOutputError(uint32_t unit_index,CTensor<type_t>& error)
{
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_SIGMOID) CTensorApplyFunc<type_t>::ApplyDifferentialSigmoid(cTensor_Delta_Array[unit_index],PrevLayerPtr->GetOutputTensor(unit_index));
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_RELU) CTensorApplyFunc<type_t>::ApplyDifferentialReLU(cTensor_Delta_Array[unit_index],PrevLayerPtr->GetOutputTensor(unit_index));
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_GELU) CTensorApplyFunc<type_t>::ApplyDifferentialGeLU(cTensor_Delta_Array[unit_index],PrevLayerPtr->GetOutputTensor(unit_index));
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LEAKY_RELU) CTensorApplyFunc<type_t>::ApplyDifferentialLeakyReLU(cTensor_Delta_Array[unit_index],PrevLayerPtr->GetOutputTensor(unit_index));
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LINEAR) CTensorApplyFunc<type_t>::ApplyDifferentialLinear(cTensor_Delta_Array[unit_index],PrevLayerPtr->GetOutputTensor(unit_index));
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_TANGENCE) CTensorApplyFunc<type_t>::ApplyDifferentialTangence(cTensor_Delta_Array[unit_index],PrevLayerPtr->GetOutputTensor(unit_index));
 //if (NeuronFunction==NNeuron::NEURON_FUNCTION_SOFTMAX) CTensorApplyFunc<type_t>::ApplyDifferentialSoftMax(cTensor_Delta_Array[unit_index],PrevLayerPtr->GetOutputTensor(unit_index));

 CTensorMath<type_t>::TensorItemProduction(cTensor_Delta_Array[unit_index],error,cTensor_Delta_Array[unit_index]);
}

//----------------------------------------------------------------------------------------------------
/*!ограничить веса в диапазон
\param[in] min Минимальное значение веса
\param[in] max Максимальное значение веса
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::ClipWeight(type_t min,type_t max)
{
}
#endif
