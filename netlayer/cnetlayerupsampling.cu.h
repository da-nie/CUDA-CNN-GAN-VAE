#ifndef C_NET_LAYER_UP_SAMPLING_H
#define C_NET_LAYER_UP_SAMPLING_H

//****************************************************************************************************
//\file Слой повышения разрешения
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdio.h>
#include <fstream>
#include <vector>

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
//!Слой повышения разрешения
//****************************************************************************************************
template<class type_t>
class CNetLayerUpSampling:public INetLayer<type_t>
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

  std::vector<CTensor<type_t>> cTensor_H_Array;///<тензоры значений нейронов после функции активации

  size_t UpSampling_X;///<коэффициент расширения по X
  size_t UpSampling_Y;///<коэффициент расширения по Y

  size_t InputSize_X;///<размер входного тензора по X
  size_t InputSize_Y;///<размер входного тензора по Y
  size_t InputSize_Z;///<размер входного тензора по Z

  //тензоры, используемые при обучении
  std::vector<CTensor<type_t>> cTensor_Delta_Array;///<тензоры дельты слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerUpSampling(size_t upsampling_y,size_t upsampling_x,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);
  CNetLayerUpSampling(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerUpSampling();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(size_t upsampling_y,size_t upsampling_x,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);///<создать слой
  void Reset(void);///<выполнить инициализацию слоя
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
CNetLayerUpSampling<type_t>::CNetLayerUpSampling(size_t upsampling_y,size_t upsampling_x,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 UpSampling_X=0;///<коэффициент расширения по X
 UpSampling_Y=0;///<коэффициент расширения по Y
 Create(upsampling_y,upsampling_x,prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerUpSampling<type_t>::CNetLayerUpSampling(void)
{
 Create(1,1);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerUpSampling<type_t>::~CNetLayerUpSampling()
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
\param[in] upsampling_y Коэффициент сжатия по Y
\param[in] upsampling_x Коэффициент сжатия по X
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::Create(size_t upsampling_y,size_t upsampling_x,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 cTensor_H_Array.resize(BatchSize);

 UpSampling_X=upsampling_x;
 UpSampling_Y=upsampling_y;

 if (upsampling_x==0 || upsampling_y==0) throw("Коэффициенты слоя обратной субдискретизации не должны быть нулями!");

 if (prev_layer_ptr==NULL) throw("Слой обратной субдискретизации не может быть входным!");//слой без предшествующего считается входным

 //размер входного тензора
 size_t input_x=PrevLayerPtr->GetOutputTensor(0).GetSizeX();
 size_t input_y=PrevLayerPtr->GetOutputTensor(0).GetSizeY();
 size_t input_z=PrevLayerPtr->GetOutputTensor(0).GetSizeZ();
 //размер выходного тензора
 size_t output_x=input_x*upsampling_x;
 size_t output_y=input_y*upsampling_y;
 size_t output_z=input_z;

 //запомним размеры входного тензора, чтобы потом всегда к ним приводить
 InputSize_X=input_x;
 InputSize_Y=input_y;
 InputSize_Z=input_z;

 if (output_x==0 || output_y==0) throw("Выходной тензор слоя обратной субдискретизации нулевого размера!");

 for(size_t n=0;n<BatchSize;n++)
 {
  //создаём выходные тензоры
  cTensor_H_Array[n]=CTensor<type_t>(output_z,output_y,output_x);
 }
//задаём предшествующему слою, что мы его последующий слой
 prev_layer_ptr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::Reset(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H_Array[unit_index].GetSizeX()) throw("void CNetLayerUpSampling<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H_Array[unit_index].GetSizeY()) throw("void CNetLayerUpSampling<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H_Array[unit_index].GetSizeZ()) throw("void CNetLayerUpSampling<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности тензора output!");
 cTensor_H_Array[unit_index]=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H_Array[unit_index].GetSizeX()) throw("void CNetLayerUpSampling<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H_Array[unit_index].GetSizeY()) throw("void CNetLayerUpSampling<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H_Array[unit_index].GetSizeZ()) throw("void CNetLayerUpSampling<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H_Array[unit_index];
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::Forward(void)
{
 for(size_t n=0;n<BatchSize;n++)
 {
  //размер выходного тензора
  size_t output_x=cTensor_H_Array[n].GetSizeX();
  size_t output_y=cTensor_H_Array[n].GetSizeY();
  size_t output_z=cTensor_H_Array[n].GetSizeZ();

  //приведём входной тензор к нужному виду

  size_t basic_input_x=PrevLayerPtr->GetOutputTensor(n).GetSizeX();
  size_t basic_input_y=PrevLayerPtr->GetOutputTensor(n).GetSizeY();
  size_t basic_input_z=PrevLayerPtr->GetOutputTensor(n).GetSizeZ();

  PrevLayerPtr->GetOutputTensor(n).ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);
  CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor(n);

  CTensorMath<type_t>::UpSampling(cTensor_H_Array[n],input,UpSampling_X,UpSampling_Y);

  PrevLayerPtr->GetOutputTensor(n).ReinterpretSize(basic_input_z,basic_input_y,basic_input_x);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerUpSampling<type_t>::GetOutputTensor(size_t unit_index)
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
void CNetLayerUpSampling<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerUpSampling<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveUInt32(UpSampling_X);
 iDataStream_Ptr->SaveUInt32(UpSampling_Y);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerUpSampling<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 UpSampling_X=iDataStream_Ptr->LoadUInt32();
 UpSampling_Y=iDataStream_Ptr->LoadUInt32();
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerUpSampling<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
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
bool CNetLayerUpSampling<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 cTensor_Delta_Array.resize(BatchSize);
 for(size_t n=0;n<BatchSize;n++)
 {
  cTensor_Delta_Array[n]=cTensor_H_Array[n];
 }
 cTensor_PrevLayerError=PrevLayerPtr->GetOutputTensor(0);
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta_Array.clear();
 cTensor_PrevLayerError=CTensor<type_t>(1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::TrainingBackward(bool create_delta_weight)
{
 for(size_t n=0;n<BatchSize;n++)
 {
  size_t basic_input_x=PrevLayerPtr->GetOutputTensor(n).GetSizeX();
  size_t basic_input_y=PrevLayerPtr->GetOutputTensor(n).GetSizeY();
  size_t basic_input_z=PrevLayerPtr->GetOutputTensor(n).GetSizeZ();
  //приведём входной тензор к нужному виду
  PrevLayerPtr->GetOutputTensor(n).ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);
  cTensor_PrevLayerError.ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);

  CTensorMath<type_t>::DownSampling(cTensor_PrevLayerError,cTensor_Delta_Array[n],UpSampling_X,UpSampling_Y);

  //задаём ошибку предыдущего слоя
  PrevLayerPtr->GetOutputTensor(n).ReinterpretSize(basic_input_z,basic_input_y,basic_input_x);
  cTensor_PrevLayerError.ReinterpretSize(basic_input_z,basic_input_y,basic_input_x);

  PrevLayerPtr->SetOutputError(n,cTensor_PrevLayerError);
 }
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::TrainingUpdateWeight(double speed,double iteration)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerUpSampling<type_t>::GetDeltaTensor(size_t unit_index)
{
 return(cTensor_Delta_Array[unit_index]);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerUpSampling<type_t>::SetOutputError(size_t unit_index,CTensor<type_t>& error)
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
void CNetLayerUpSampling<type_t>::ClipWeight(type_t min,type_t max)
{
}

#endif
