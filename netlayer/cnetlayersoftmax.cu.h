#ifndef C_NET_LAYER_SOFT_MAX_H
#define C_NET_LAYER_SOFT_MAX_H

//****************************************************************************************************
//\file Слой SoftMax
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
//!Слой SoftMax
//****************************************************************************************************
template<class type_t>
class CNetLayerSoftMax:public INetLayer<type_t>
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

  uint32_t InputSize_X;///<размер входного тензора по X
  uint32_t InputSize_Y;///<размер входного тензора по Y
  uint32_t InputSize_Z;///<размер входного тензора по Z

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<тензоры дельты слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerSoftMax(INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerSoftMax(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerSoftMax();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
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
CNetLayerSoftMax<type_t>::CNetLayerSoftMax(INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 Create(prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerSoftMax<type_t>::CNetLayerSoftMax(void)
{
 Create();
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerSoftMax<type_t>::~CNetLayerSoftMax()
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
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::Create(INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 if (prev_layer_ptr==NULL) throw("Слой SoftMax не может быть входным!");//слой без предшествующего считается входным

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
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::Reset(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::SetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerSoftMax<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerSoftMax<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerSoftMax<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerSoftMax<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerSoftMax<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerSoftMax<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerSoftMax<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerSoftMax<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::Forward(void)
{
 CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();

 uint32_t input_x=input.GetSizeX();
 uint32_t input_y=input.GetSizeY();
 uint32_t input_z=input.GetSizeZ();
 uint32_t input_w=input.GetSizeW();

 for(size_t w=0;w<input_w;w++)
 {
  //считаем сумму экспонент
  double summ=0;
  for(size_t z=0;z<input_z;z++)
  {
   for(size_t y=0;y<input_y;y++)
   {
    for(size_t x=0;x<input_x;x++)
    {
     type_t e=input.GetElement(w,z,y,x);
     summ+=exp(e);
    }
   }
  }
  //делим каждый элемент на сумму
  for(size_t z=0;z<input_z;z++)
  {
   for(size_t y=0;y<input_y;y++)
   {
    for(size_t x=0;x<input_x;x++)
    {
     type_t e=input.GetElement(w,z,y,x);
     e=exp(e)/summ;
     cTensor_H.SetElement(w,z,y,x,e);
    }
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
CTensor<type_t>& CNetLayerSoftMax<type_t>::GetOutputTensor(void)
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
void CNetLayerSoftMax<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerSoftMax<type_t>::Save(IDataStream *iDataStream_Ptr)
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
bool CNetLayerSoftMax<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
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
bool CNetLayerSoftMax<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
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
bool CNetLayerSoftMax<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 cTensor_Delta=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::TrainingBackward(bool create_delta_weight)
{
 //задаём ошибку предыдущего слоя
 PrevLayerPtr->SetOutputError(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerSoftMax<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::SetOutputError(CTensor<type_t>& error)
{
CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();

 uint32_t input_x=input.GetSizeX();
 uint32_t input_y=input.GetSizeY();
 uint32_t input_z=input.GetSizeZ();
 uint32_t input_w=input.GetSizeW();

 for(size_t w=0;w<input_w;w++)
 {
  type_t S=0;

  for(size_t z=0;z<input_z;z++)
  {
   for(size_t y=0;y<input_y;y++)
   {
    for(size_t x=0;x<input_x;x++)
    {
     type_t s=cTensor_H.GetElement(w,z,y,x);
     type_t a=error.GetElement(w,z,y,x);
     S+=s*a;
    }
   }
  }

  for(size_t z=0;z<input_z;z++)
  {
   for(size_t y=0;y<input_y;y++)
   {
    for(size_t x=0;x<input_x;x++)
    {
     type_t s=cTensor_H.GetElement(w,z,y,x);
     type_t c=error.GetElement(w,z,y,x)-S;
     c*=s;
     cTensor_Delta.SetElement(w,z,y,x,c);
    }
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
/*!ограничить веса в диапазон
\param[in] min Минимальное значение веса
\param[in] max Максимальное значение веса
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::ClipWeight(type_t min,type_t max)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" SoftMax: input",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSoftMax<type_t>::PrintOutputTensorSize(const std::string &name)
{
 GetOutputTensor().Print(name+" SoftMax: output",false);
}

#endif
