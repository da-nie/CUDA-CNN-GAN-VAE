#ifndef C_NET_LAYER_MAX_DE_POOLING_H
#define C_NET_LAYER_MAX_DE_POOLING_H

//****************************************************************************************************
//\file Слой обратной субдискретизации
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
//!Слой обратной субдискретизации
//****************************************************************************************************
template<class type_t>
class CNetLayerMaxDePooling:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  CTensor<type_t> cTensor_H;///<тензор значений нейронов после функции активации

  size_t DePooling_X;///<коэффициент расширения по X
  size_t DePooling_Y;///<коэффициент расширения по Y

  size_t InputSize_X;///<размер входного тензора по X
  size_t InputSize_Y;///<размер входного тензора по Y
  size_t InputSize_Z;///<размер входного тензора по Z

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<тензор дельты слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerMaxDePooling(size_t depooling_y,size_t depooling_x,INetLayer<type_t> *prev_layer_ptr=NULL);
  CNetLayerMaxDePooling(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerMaxDePooling();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(size_t depooling_y,size_t depooling_x,INetLayer<type_t> *prev_layer_ptr=NULL);///<создать слой
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
CNetLayerMaxDePooling<type_t>::CNetLayerMaxDePooling(size_t depooling_y,size_t depooling_x,INetLayer<type_t> *prev_layer_ptr)
{
 DePooling_X=1;///<коэффициент расширения по X
 DePooling_Y=1;///<коэффициент расширения по Y
 Create(depooling_y,depooling_x,prev_layer_ptr);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerMaxDePooling<type_t>::CNetLayerMaxDePooling(void)
{
 Create(1,1);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerMaxDePooling<type_t>::~CNetLayerMaxDePooling()
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
void CNetLayerMaxDePooling<type_t>::Create(size_t depooling_y,size_t depooling_x,INetLayer<type_t> *prev_layer_ptr)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 DePooling_X=depooling_x;
 DePooling_Y=depooling_y;

 if (depooling_x==0 || depooling_y==0) throw("Коэффициенты слоя обратной субдискретизации не должны быть нулями!");

 if (prev_layer_ptr==NULL) throw("Слой обратной субдискретизации не может быть входным!");//слой без предшествующего считается входным

 //размер входного тензора
 size_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 //размер выходного тензора
 size_t output_x=input_x*depooling_x;
 size_t output_y=input_y*depooling_y;
 size_t output_z=input_z;

 //запомним размеры входного тензора, чтобы потом всегда к ним приводить
 InputSize_X=input_x;
 InputSize_Y=input_y;
 InputSize_Z=input_z;

 if (output_x==0 || output_y==0) throw("Выходной тензор слоя обратной субдискретизации нулевого размера!");
 //создаём выходные тензоры
 cTensor_H=CTensor<type_t>(output_z,output_y,output_x);
 //задаём предшествующему слою, что мы его последующий слой
 prev_layer_ptr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxDePooling<type_t>::Reset(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxDePooling<type_t>::SetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerMaxDePooling<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerMaxDePooling<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerMaxDePooling<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxDePooling<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerMaxDePooling<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerMaxDePooling<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerMaxDePooling<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxDePooling<type_t>::Forward(void)
{
 //размер выходного тензора
 size_t output_x=cTensor_H.GetSizeX();
 size_t output_y=cTensor_H.GetSizeY();
 size_t output_z=cTensor_H.GetSizeZ();

 //приведём входной тензор к нужному виду
 size_t basic_input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t basic_input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t basic_input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();

 PrevLayerPtr->GetOutputTensor().ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);

 CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();
 size_t input_x=input.GetSizeX();
 size_t input_y=input.GetSizeY();
 size_t input_z=input.GetSizeZ();

 type_t divider=static_cast<type_t>(DePooling_X*DePooling_Y);

 for(size_t z=0;z<input_z;z++)
 {
  for(size_t y=0;y<input_y;y++)
  {
   for(size_t x=0;x<input_x;x++)
   {
    type_t input_v=input.GetElement(z,y,x);
    //выставляем всем точкам блока обратной субдискретизации значение входа, делённое на размерность
    type_t value=input_v;///divider;
    size_t hx=x*DePooling_X;
    size_t hy=y*DePooling_Y;
    for(size_t py=0;py<DePooling_Y;py++)
    {
     for(size_t px=0;px<DePooling_X;px++)
     {
      cTensor_H.SetElement(z,hy+py,hx+px,value);
     }
    }
   }
  }
 }
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(basic_input_z,basic_input_y,basic_input_x);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerMaxDePooling<type_t>::GetOutputTensor(void)
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
void CNetLayerMaxDePooling<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerMaxDePooling<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveUInt32(DePooling_X);
 iDataStream_Ptr->SaveUInt32(DePooling_Y);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerMaxDePooling<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 DePooling_X=iDataStream_Ptr->LoadUInt32();
 DePooling_Y=iDataStream_Ptr->LoadUInt32();
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxDePooling<type_t>::TrainingStart(void)
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
void CNetLayerMaxDePooling<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxDePooling<type_t>::TrainingBackward(void)
{
 size_t basic_input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t basic_input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t basic_input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 //приведём входной тензор к нужному виду
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);
 cTensor_PrevLayerError.ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);

 size_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();

 //размер выходного тензора
 size_t output_x=cTensor_Delta.GetSizeX();
 size_t output_y=cTensor_Delta.GetSizeY();
 size_t output_z=cTensor_Delta.GetSizeZ();

 type_t divider=static_cast<type_t>(DePooling_X*DePooling_Y);

 for(size_t z=0;z<input_z;z++)
 {
  for(size_t y=0;y<input_y;y++)
  {
   for(size_t x=0;x<input_x;x++)
   {
    size_t ix=x*DePooling_X;
    size_t iy=y*DePooling_Y;
    type_t res=0;
    for(size_t py=0;py<DePooling_Y;py++)
    {
     for(size_t px=0;px<DePooling_X;px++)
     {
      type_t e=cTensor_Delta.GetElement(z,iy+py,ix+px);
      res+=e;
     }
    }
    res/=divider;
    cTensor_PrevLayerError.SetElement(z,y,x,res);
   }
  }
 }
 //задаём ошибку предыдущего слоя
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(basic_input_z,basic_input_y,basic_input_x);
 cTensor_PrevLayerError.ReinterpretSize(basic_input_z,basic_input_y,basic_input_x);

 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxDePooling<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxDePooling<type_t>::TrainingUpdateWeight(double speed)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerMaxDePooling<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerMaxDePooling<type_t>::SetOutputError(CTensor<type_t>& error)
{
 cTensor_Delta=error;
}

#endif
