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

  CTensor<type_t> cTensor_H;///<тензоры значений нейронов после функции активации
  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<тензоры дельты слоя
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
  void SetOutput(CTensor<type_t> &output);///<задать выход слоя
  void GetOutput(CTensor<type_t> &output);///<получить выход слоя
  void Forward(void);///<выполнить прямой проход по слою
  CTensor<type_t>& GetOutputTensor(void);///<получить ссылку на выходной тензор
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
  cTensor_H=PrevLayerPtr->GetOutputTensor();
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
void CNetLayerFunction<type_t>::SetOutput(CTensor<type_t> &output)
{
 cTensor_H.CopyItem(output);
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerFunction<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerFunction<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerFunction<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerFunction<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::Forward(void)
{
 //применим функцию активации
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_SIGMOID) CTensorApplyFunc<type_t>::ApplySigmoid(cTensor_H,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_RELU) CTensorApplyFunc<type_t>::ApplyReLU(cTensor_H,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_GELU) CTensorApplyFunc<type_t>::ApplyGeLU(cTensor_H,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LEAKY_RELU) CTensorApplyFunc<type_t>::ApplyLeakyReLU(cTensor_H,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LINEAR) CTensorApplyFunc<type_t>::ApplyLinear(cTensor_H,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_TANGENCE) CTensorApplyFunc<type_t>::ApplyTangence(cTensor_H,PrevLayerPtr->GetOutputTensor());
 //if (NeuronFunction==NNeuron::NEURON_FUNCTION_SOFTMAX) CTensorApplyFunc<type_t>::ApplySoftMax(cTensor_H,PrevLayerPtr->GetOutputTensor());
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerFunction<type_t>::GetOutputTensor(void)
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
 cTensor_Delta=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
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
  PrevLayerPtr->SetOutputError(cTensor_Delta);
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
CTensor<type_t>& CNetLayerFunction<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::SetOutputError(CTensor<type_t>& error)
{
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_SIGMOID) CTensorApplyFunc<type_t>::ApplyDifferentialSigmoid(cTensor_Delta,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_RELU) CTensorApplyFunc<type_t>::ApplyDifferentialReLU(cTensor_Delta,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_GELU) CTensorApplyFunc<type_t>::ApplyDifferentialGeLU(cTensor_Delta,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LEAKY_RELU) CTensorApplyFunc<type_t>::ApplyDifferentialLeakyReLU(cTensor_Delta,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LINEAR) CTensorApplyFunc<type_t>::ApplyDifferentialLinear(cTensor_Delta,PrevLayerPtr->GetOutputTensor());
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_TANGENCE) CTensorApplyFunc<type_t>::ApplyDifferentialTangence(cTensor_Delta,PrevLayerPtr->GetOutputTensor());
 //if (NeuronFunction==NNeuron::NEURON_FUNCTION_SOFTMAX) CTensorApplyFunc<type_t>::ApplyDifferentialSoftMax(cTensor_Delta,PrevLayerPtr->GetOutputTensor());

 CTensorMath<type_t>::TensorItemProduction(cTensor_Delta,error,cTensor_Delta);
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
//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" Function: input",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerFunction<type_t>::PrintOutputTensorSize(const std::string &name)
{
 GetOutputTensor().Print(name+" Function: output",false);
}

#endif
