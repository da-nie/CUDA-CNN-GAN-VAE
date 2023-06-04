#ifndef C_NET_LAYER_MAX_POOLING_H
#define C_NET_LAYER_MAX_POOLING_H

//****************************************************************************************************
//\file Слой субдискретизации
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdio.h>
#include <fstream>
#include <vector>

#include "../common/idatastream.h"
#include "inetlayer.cu.h"
#include "../cuda/ctensor.cu.h"
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
//!Слой субдискретизации
//****************************************************************************************************
template<class type_t>
class CNetLayerMaxPooling:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  struct SPos
  {
   size_t X;
   size_t Y;
  };

  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  CTensor<type_t> cTensor_H;///<тензор значений нейронов после функции активации
  CTensor<SPos> cTensor_P;///<тензор выбранной позиции субдискретизации (номер точки в блоке)

  size_t Pooling_X;///<коэффициент сжатия по X
  size_t Pooling_Y;///<коэффициент сжатия по Y

  size_t InputSize_X;///<размер входного тензора по X
  size_t InputSize_Y;///<размер входного тензора по Y
  size_t InputSize_Z;///<размер входного тензора по Z

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<тензор дельты слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerMaxPooling(size_t pooling_y,size_t pooling_x,INetLayer<type_t> *prev_layer_ptr=NULL);
  CNetLayerMaxPooling(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerMaxPooling();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(size_t pooling_y,size_t pooling_x,INetLayer<type_t> *prev_layer_ptr=NULL);///<создать слой
  void Reset(void);///<выполнить инициализацию слоя
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
CNetLayerMaxPooling<type_t>::CNetLayerMaxPooling(size_t pooling_y,size_t pooling_x,INetLayer<type_t> *prev_layer_ptr)
{
 Pooling_X=1;///<коэффициент сжатия по X
 Pooling_Y=1;///<коэффициент сжатия по Y
 Create(pooling_y,pooling_x,prev_layer_ptr);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerMaxPooling<type_t>::CNetLayerMaxPooling(void)
{
 Create(1,1);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerMaxPooling<type_t>::~CNetLayerMaxPooling()
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
\param[in] pooling_y Коэффициент сжатия по Y
\param[in] pooling_x Коэффициент сжатия по X
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::Create(size_t pooling_y,size_t pooling_x,INetLayer<type_t> *prev_layer_ptr)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 Pooling_X=pooling_x;
 Pooling_Y=pooling_y;

 if (pooling_x==0 || pooling_y==0) throw("Коэффициенты слоя субдискретизации не должны быть нулями!");

 if (prev_layer_ptr==NULL) throw("Слой субдискретизации не может быть входным!");//слой без предшествующего считается входным

 //размер входного тензора
 size_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 //размер выходного тензора
 size_t output_x=input_x/pooling_x;
 size_t output_y=input_y/pooling_y;
 size_t output_z=input_z;

 //запомним размеры входного тензора, чтобы потом всегда к ним приводить
 InputSize_X=input_x;
 InputSize_Y=input_y;
 InputSize_Z=input_z;

 if (output_x==0 || output_y==0) throw("Выходной тензор слоя субдискретизации нулевого размера!");
 //создаём выходные тензоры
 cTensor_H=CTensor<type_t>(output_z,output_y,output_x);
 cTensor_P=CTensor<SPos>(output_z,output_y,output_x);
 //задаём предшествующему слою, что мы его последующий слой
 prev_layer_ptr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::Reset(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::SetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerMaxPooling<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerMaxPooling<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerMaxPooling<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerMaxPooling<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerMaxPooling<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerMaxPooling<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::Forward(void)
{
 //размер выходного тензора
 size_t output_x=cTensor_H.GetSizeX();
 size_t output_y=cTensor_H.GetSizeY();
 size_t output_z=cTensor_H.GetSizeZ();

 //приведём входной тензор к нужному виду
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);

 CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();
 size_t input_x=input.GetSizeX();
 size_t input_y=input.GetSizeY();
 size_t input_z=input.GetSizeZ();

 for(size_t z=0;z<output_z;z++)
 {
  for(size_t y=0;y<output_y;y++)
  {
   for(size_t x=0;x<output_x;x++)
   {
    size_t ix=x*Pooling_X;
    size_t iy=y*Pooling_Y;
    type_t max=input.GetElement(z,iy,ix);
    size_t max_x=ix;
    size_t max_y=iy;
    for(size_t py=0;py<Pooling_Y;py++)
    {
     for(size_t px=0;px<Pooling_X;px++)
     {
      type_t e=input.GetElement(z,iy+py,ix+px);
      if (e>max)
      {
       max=e;
       max_x=ix+px;
       max_y=iy+py;
      }
     }
    }
    cTensor_H.SetElement(z,y,x,max);
    SPos sPos;
    sPos.X=max_x;
    sPos.Y=max_y;
    cTensor_P.SetElement(z,y,x,sPos);
   }
  }
 }

 PrevLayerPtr->GetOutputTensor().ReinterpretSize(input_z,input_y,input_x);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerMaxPooling<type_t>::GetOutputTensor(void)
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
void CNetLayerMaxPooling<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerMaxPooling<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveUInt32(Pooling_X);
 iDataStream_Ptr->SaveUInt32(Pooling_Y);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerMaxPooling<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 Pooling_X=iDataStream_Ptr->LoadUInt32();
 Pooling_Y=iDataStream_Ptr->LoadUInt32();
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 cTensor_Delta=cTensor_H;
 cTensor_PrevLayerError=PrevLayerPtr->GetOutputTensor();
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::TrainingBackward(void)
{
 size_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 //приведём входной тензор к нужному виду
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);
 cTensor_PrevLayerError.ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);

 //размер выходного тензора
 size_t output_x=cTensor_Delta.GetSizeX();
 size_t output_y=cTensor_Delta.GetSizeY();
 size_t output_z=cTensor_Delta.GetSizeZ();

 cTensor_PrevLayerError.Zero();

 for(size_t z=0;z<output_z;z++)
 {
  for(size_t y=0;y<output_y;y++)
  {
   for(size_t x=0;x<output_x;x++)
   {
    type_t delta=cTensor_Delta.GetElement(z,y,x);
    SPos sPos=cTensor_P.GetElement(z,y,x);
    cTensor_PrevLayerError.SetElement(z,sPos.Y,sPos.X,delta);
   }
  }
 }

 //задаём ошибку предыдущего слоя
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(input_z,input_y,input_x);
 cTensor_PrevLayerError.ReinterpretSize(input_z,input_y,input_x);

 //задаём ошибку предыдущего слоя
 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::TrainingUpdateWeight(double speed)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerMaxPooling<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxPooling<type_t>::SetOutputError(CTensor<type_t>& error)
{
 cTensor_Delta=error;
}

#endif
