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
  CTensor<type_t> cTensor_W;///<тензор весов слоя
  CTensor<type_t> cTensor_B;///<тензор сдвигов слоя
  CTensor<type_t> cTensor_Z;///<тензор значений нейронов до функции активации
  CTensor<type_t> cTensor_H;///<тензор значений нейронов после функции активации
  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)
  NNeuron::NEURON_FUNCTION NeuronFunction;///<функция активации нейронов слоя

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_dW;///<тензор поправок весов слоя
  CTensor<type_t> cTensor_dB;///<тензор поправок сдвигов слоя
  CTensor<type_t> cTensor_Delta;///<тензор дельты слоя
  CTensor<type_t> cTensor_TmpdW;///<вспомогательный тензор поправок весов слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerLinear(size_t neurons,NNeuron::NEURON_FUNCTION neuron_function=NNeuron::NEURON_FUNCTION_SIGMOID,INetLayer<type_t> *prev_layer_ptr=NULL);
  CNetLayerLinear(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerLinear();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(size_t neurons,NNeuron::NEURON_FUNCTION neuron_function=NNeuron::NEURON_FUNCTION_SIGMOID,INetLayer<type_t> *prev_layer_ptr=NULL);///<создать слой
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
CNetLayerLinear<type_t>::CNetLayerLinear(size_t neurons,NNeuron::NEURON_FUNCTION neuron_function,INetLayer<type_t> *prev_layer_ptr)
{
 Create(neurons,neuron_function,prev_layer_ptr);
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


//----------------------------------------------------------------------------------------------------
/*!получить случайное число
\param[in] max_value Максимальное значение случайного числа
\return Случайное число в диапазоне [0...max_value]
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CNetLayerLinear<type_t>::GetRandValue(type_t max_value)
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
void CNetLayerLinear<type_t>::Create(size_t neurons,NNeuron::NEURON_FUNCTION neuron_function,INetLayer<type_t> *prev_layer_ptr)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;
 NeuronFunction=neuron_function;
 cTensor_H=CTensor<type_t>(1,neurons,1);
 cTensor_Z=CTensor<type_t>(1,neurons,1);

 if (prev_layer_ptr==NULL)//слой без предшествующего считается входным
 {
  cTensor_W=CTensor<type_t>(1,1,1);
  cTensor_B=CTensor<type_t>(1,1,1);
 }
 else
 {
  size_t size_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
  size_t size_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
  size_t size_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();

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

 type_t size=static_cast<type_t>(cTensor_W.GetSizeX());
 type_t koeff=static_cast<type_t>(sqrt(2.0/size));
 //веса
 for(size_t y=0;y<cTensor_W.GetSizeY();y++)
 {
  for(size_t x=0;x<cTensor_W.GetSizeX();x++)
  {
   //используем метод инициализации He (Ге)
   type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
   type_t init=rnd*koeff;
   cTensor_W.SetElement(0,y,x,init);
  }
 }
 //сдвиги
 size=static_cast<type_t>(cTensor_B.GetSizeY());
 koeff=static_cast<type_t>(sqrt(2.0/size));
 for(size_t y=0;y<cTensor_B.GetSizeY();y++)
 {
  //используем метод инициализации He (Ге)
  type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
  type_t init=rnd*koeff;
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
void CNetLayerLinear<type_t>::SetOutput(CTensor<type_t> &output)
{
 //if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerLinear<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerLinear<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 //if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerLinear<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 cTensor_H.CopyItem(output);
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
 //приводим входной тензор к линии
 size_t size_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t size_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t size_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(1,size_x*size_y*size_z,1);

 CTensorMath<type_t>::Mul(cTensor_Z,cTensor_W,PrevLayerPtr->GetOutputTensor());

 PrevLayerPtr->GetOutputTensor().RestoreSize();

 CTensorMath<type_t>::Add(cTensor_Z,cTensor_Z,cTensor_B);
 //применим функцию активации

 if (NeuronFunction==NNeuron::NEURON_FUNCTION_SIGMOID) CTensorApplyFunc<type_t>::ApplySigmoid(cTensor_H,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_RELU) CTensorApplyFunc<type_t>::ApplyReLU(cTensor_H,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LEAKY_RELU) CTensorApplyFunc<type_t>::ApplyLeakyReLU(cTensor_H,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LINEAR) CTensorApplyFunc<type_t>::ApplyLinear(cTensor_H,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_TANGENCE) CTensorApplyFunc<type_t>::ApplyTangence(cTensor_H,cTensor_Z);
 //if (NeuronFunction==NNeuron::NEURON_FUNCTION_SOFTMAX) CTensorApplyFunc<type_t>::ApplySoftMax(cTensor_H,cTensor_Z);

/*
 for(size_t y=0;y<cTensor_Z.GetSizeY();y++)
 {
  type_t v=cTensor_Z.GetElement(0,y,0);
  cTensor_H.SetElement(0,y,0,NNeuron::GetNeuronFunctionPtr(NeuronFunction)(v));
 }
*/
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
 iDataStream_Ptr->SaveUInt32(NeuronFunction);
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
 NeuronFunction=iDataStream_Ptr->LoadUInt32();
 cTensor_W.Load(iDataStream_Ptr);
 cTensor_B.Load(iDataStream_Ptr);
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
 cTensor_Delta=CTensor<type_t>(1,cTensor_H.GetSizeY(),cTensor_H.GetSizeX());
 if (PrevLayerPtr!=NULL)
 {
  CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor();
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
 cTensor_Delta=CTensor<type_t>(1,1,1);
 cTensor_PrevLayerError=CTensor<type_t>(1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLinear<type_t>::TrainingBackward(void)
{
 //вычисляем ошибку предыдущего слоя (D=tr(W)xDnext)
 if (PrevLayerPtr!=NULL)//это не входной слой
 {
  //считаем ошибку предыдущего слоя
  size_t size_x=cTensor_PrevLayerError.GetSizeX();
  size_t size_y=cTensor_PrevLayerError.GetSizeY();
  size_t size_z=cTensor_PrevLayerError.GetSizeZ();
  cTensor_PrevLayerError.ReinterpretSize(1,size_x*size_y*size_z,1);

  CTensorMath<type_t>::TransponseMul(cTensor_PrevLayerError,cTensor_W,cTensor_Delta);
  cTensor_PrevLayerError.RestoreSize();
  //задаём ошибку предыдущего слоя
  PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
  CTensor<type_t> &h=PrevLayerPtr->GetOutputTensor();
  size_x=h.GetSizeX();
  size_y=h.GetSizeY();
  size_z=h.GetSizeZ();
  h.ReinterpretSize(1,1,size_x*size_y*size_z);

  CTensorMath<type_t>::Mul(cTensor_TmpdW,cTensor_Delta,h);

  h.RestoreSize();

  CTensorMath<type_t>::Add(cTensor_dW,cTensor_dW,cTensor_TmpdW);
  CTensorMath<type_t>::Add(cTensor_dB,cTensor_dB,cTensor_Delta);
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
void CNetLayerLinear<type_t>::TrainingUpdateWeight(double speed)
{
 //speed/=static_cast<double>(cTensor_H.GetSizeX()*cTensor_H.GetSizeY()*cTensor_H.GetSizeZ());
 //CTensorMath<type_t>::Mul(cTensor_dW,speed,cTensor_dW);
 CTensorMath<type_t>::Sub(cTensor_W,cTensor_W,cTensor_dW,1,speed);
 //CTensorMath<type_t>::Mul(cTensor_dB,speed,cTensor_dB);
 CTensorMath<type_t>::Sub(cTensor_B,cTensor_B,cTensor_dB,1,speed);
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
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_SIGMOID) CTensorApplyFunc<type_t>::ApplyDifferentialSigmoid(cTensor_Delta,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_RELU) CTensorApplyFunc<type_t>::ApplyDifferentialReLU(cTensor_Delta,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LEAKY_RELU) CTensorApplyFunc<type_t>::ApplyDifferentialLeakyReLU(cTensor_Delta,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LINEAR) CTensorApplyFunc<type_t>::ApplyDifferentialLinear(cTensor_Delta,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_TANGENCE) CTensorApplyFunc<type_t>::ApplyDifferentialTangence(cTensor_Delta,cTensor_Z);
 //if (NeuronFunction==NNeuron::NEURON_FUNCTION_SOFTMAX) CTensorApplyFunc<type_t>::ApplyDifferentialSoftMax(cTensor_Delta,cTensor_Z);

 CTensorMath<type_t>::TensorItemProduction(cTensor_Delta,error,cTensor_Delta);

/*
 for(size_t y=0;y<error.GetSizeY();y++)
 {
  type_t e=error.GetElement(0,y,0);
  type_t v=cTensor_Z.GetElement(0,y,0);
  cTensor_Delta.SetElement(0,y,0,e*NNeuron::GetNeuronFunctionDifferencialPtr(NeuronFunction)(v));
 }
 */
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
 cTensor_W.Clip(min,max);
 cTensor_B.Clip(min,max);
}

#endif
