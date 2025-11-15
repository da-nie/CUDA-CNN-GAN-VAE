#ifndef C_NET_LAYER_VAE_CODER_OUTPUT_H
#define C_NET_LAYER_VAE_CODER_OUTPUT_H

//****************************************************************************************************
//\file Слой выхода кодера генеративного автоэнкодера
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
#include "../common/crandom.h"

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
//! Слой выхода кодера генеративного автоэнкодера
//****************************************************************************************************
template<class type_t>
class CNetLayerVAECoderOutput:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  INetLayer<type_t> *PrevLayerMuPtr;///<указатель на предшествующий слой Мю (либо NULL)
  INetLayer<type_t> *PrevLayerLogVarPtr;///<указатель на предшествующий слой логарифм дисперсии (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  uint32_t BatchSize;///<размер пакета для обучения

  CTensor<type_t> cTensor_H;///<выходной тензор значений нейронов
  CTensor<type_t> cTensor_Epsilon;///<тензор шума

  uint32_t InputSize_X;///<размер входного тензора по X
  uint32_t InputSize_Y;///<размер входного тензора по Y
  uint32_t InputSize_Z;///<размер входного тензора по Z

  uint32_t OutputSize_X;///<размер выходного тензора по X
  uint32_t OutputSize_Y;///<размер выходного тензора по Y
  uint32_t OutputSize_Z;///<размер выходного тензора по Z

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<тензоры дельты слоя
  CTensor<type_t> cTensor_PrevLayerMuError;///<тензоры ошибки слоя Мю
  CTensor<type_t> cTensor_PrevLayerLogVarError;///<тензоры ошибки слоя логарифм дисперсии
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerVAECoderOutput(INetLayer<type_t> *prev_layer_mu_ptr=NULL,INetLayer<type_t> *prev_layer_log_var_ptr=NULL,uint32_t batch_size=1);
  CNetLayerVAECoderOutput(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerVAECoderOutput();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(INetLayer<type_t> *prev_layer_mu_ptr=NULL,INetLayer<type_t> *prev_layer_log_var_ptr=NULL,uint32_t batch_size=1);///<создать слой
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
CNetLayerVAECoderOutput<type_t>::CNetLayerVAECoderOutput(INetLayer<type_t> *prev_layer_mu_ptr,INetLayer<type_t> *prev_layer_log_var_ptr,uint32_t batch_size)
{
 Create(prev_layer_mu_ptr,prev_layer_log_var_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerVAECoderOutput<type_t>::CNetLayerVAECoderOutput(void)
{
 Create();
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerVAECoderOutput<type_t>::~CNetLayerVAECoderOutput()
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
\param[in] prev_layer_mu_ptr Указатель на класс предшествующего слоя Мю (NULL-слой входной)
\param[in] prev_layer_log_var_ptr Указатель на класс предшествующего слоя логарифм дисперсии (NULL-слой входной)
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::Create(INetLayer<type_t> *prev_layer_mu_ptr,INetLayer<type_t> *prev_layer_log_var_ptr,uint32_t batch_size)
{
 PrevLayerMuPtr=prev_layer_mu_ptr;
 PrevLayerLogVarPtr=prev_layer_log_var_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 if (prev_layer_mu_ptr==NULL) throw("Слой выхода кодера VAE не может быть входным!");//слой без предшествующего считается входным
 if (prev_layer_log_var_ptr==NULL) throw("Слой выхода кодера VAE не может быть входным!");//слой без предшествующего считается входным

 //запомним размеры входного тензора, чтобы потом всегда к ним приводить
 InputSize_X=PrevLayerMuPtr->GetOutputTensor().GetSizeX();
 InputSize_Y=PrevLayerMuPtr->GetOutputTensor().GetSizeY();
 InputSize_Z=PrevLayerMuPtr->GetOutputTensor().GetSizeZ();

 if (InputSize_X!=PrevLayerLogVarPtr->GetOutputTensor().GetSizeX() || InputSize_Y!=PrevLayerLogVarPtr->GetOutputTensor().GetSizeY() || InputSize_Z!=PrevLayerLogVarPtr->GetOutputTensor().GetSizeZ()) throw("Слои Мю и логарифм дисперсии должны иметь одинаковую размерность по X, Y и Z!");

 //размер выходного тензора
 OutputSize_X=InputSize_X;
 OutputSize_Y=InputSize_Y;
 OutputSize_Z=InputSize_Z;

 //создаём выходные тензоры
 cTensor_H=CTensor<type_t>(BatchSize,OutputSize_Z,OutputSize_Y,OutputSize_X);
 cTensor_Epsilon=CTensor<type_t>(BatchSize,InputSize_Z,InputSize_Y,InputSize_X);
 //задаём предшествующему слою, что мы его последующий слой
 prev_layer_mu_ptr->SetNextLayerPtr(this);
 prev_layer_log_var_ptr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::Reset(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerVAECoderOutput<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerVAECoderOutput<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::Forward(void)
{
 type_t mean=0;
 type_t stddev=1;

 //генератор случайных чисел
 static std::random_device rd;

 type_t kl_loss=0;//расстояние Кульбака-Лейбнера

 for(uint32_t w=0;w<PrevLayerMuPtr->GetOutputTensor().GetSizeW();w++)
 {
  std::mt19937 gen(rd());
  std::normal_distribution<type_t> dist(mean,stddev);

  for(uint32_t z=0;z<PrevLayerMuPtr->GetOutputTensor().GetSizeZ();z++)
  {
   for(uint32_t y=0;y<PrevLayerMuPtr->GetOutputTensor().GetSizeY();y++)
   {
    for(uint32_t x=0;x<PrevLayerMuPtr->GetOutputTensor().GetSizeX();x++)
    {
     type_t mu=PrevLayerMuPtr->GetOutputTensor().GetElement(w,z,y,x);
     type_t log_var=PrevLayerLogVarPtr->GetOutputTensor().GetElement(w,z,y,x);
     type_t e=exp(log_var/2.0);
     type_t epsilon=dist(gen);
     cTensor_Epsilon.SetElement(w,z,y,x,epsilon);
     type_t out=mu+e*epsilon;
     cTensor_H.SetElement(w,z,y,x,out);
     //считаем расстояние Кульбака-Лейбнера
     kl_loss+=1+log_var-mu*mu-exp(log_var);
    }
   }
  }
 }

 kl_loss*=-0.5;
 kl_loss/=PrevLayerMuPtr->GetOutputTensor().GetSizeW();
 printf("KL loss:%f\r\n",kl_loss);

}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerVAECoderOutput<type_t>::GetOutputTensor(void)
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
void CNetLayerVAECoderOutput<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerVAECoderOutput<type_t>::Save(IDataStream *iDataStream_Ptr)
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
bool CNetLayerVAECoderOutput<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
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
bool CNetLayerVAECoderOutput<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
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
bool CNetLayerVAECoderOutput<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 cTensor_Delta=cTensor_H;
 cTensor_PrevLayerMuError=CTensor<type_t>(BatchSize,OutputSize_Z,OutputSize_Y,OutputSize_X);
 cTensor_PrevLayerLogVarError=CTensor<type_t>(BatchSize,OutputSize_Z,OutputSize_Y,OutputSize_X);
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
 cTensor_PrevLayerMuError=CTensor<type_t>(1,1,1,1);
 cTensor_PrevLayerLogVarError=CTensor<type_t>(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingBackward(bool create_delta_weight)
{
 //задаём ошибки предыдущих слоёв
 PrevLayerMuPtr->SetOutputError(cTensor_PrevLayerMuError);
 PrevLayerLogVarPtr->SetOutputError(cTensor_PrevLayerLogVarError);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingResetDeltaWeight(void)
{
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerVAECoderOutput<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::SetOutputError(CTensor<type_t>& error)
{
 cTensor_Delta=error;
 type_t k=0.001;//*1.0/(cTensor_PrevLayerLogVarError.GetSizeX()*cTensor_PrevLayerLogVarError.GetSizeY()*cTensor_PrevLayerLogVarError.GetSizeZ());

 for(uint32_t w=0;w<cTensor_PrevLayerLogVarError.GetSizeW();w++)
 {
  for(uint32_t z=0;z<cTensor_PrevLayerLogVarError.GetSizeZ();z++)
  {
   for(uint32_t y=0;y<cTensor_PrevLayerLogVarError.GetSizeY();y++)
   {
    for(uint32_t x=0;x<cTensor_PrevLayerLogVarError.GetSizeX();x++)
    {
     type_t epsilon=cTensor_Epsilon.GetElement(w,z,y,x);
     //считаем ошибку для слоя логарифма дисперсии
     type_t log_var_delta=cTensor_Delta.GetElement(w,z,y,x);
     type_t log_var=PrevLayerLogVarPtr->GetOutputTensor().GetElement(w,z,y,x);
     type_t log_var_diff=1.0/2.0*epsilon*exp(log_var/2.0);//производная функции построения выхода слоя по log_var
     //по этим данным получаем ошибку на входе слоя
     type_t log_var_delta_out=log_var_delta*log_var_diff;
     log_var_delta_out+=-1.0/2.0*(1-exp(log_var))*k;//добавляем ошибку по расстоянию Кульбака-Лейбнера
     cTensor_PrevLayerLogVarError.SetElement(w,z,y,x,log_var_delta_out);

     //считаем ошибку для слоя среднего
     type_t mu=PrevLayerMuPtr->GetOutputTensor().GetElement(w,z,y,x);
     type_t mu_delta=cTensor_Delta.GetElement(w,z,y,x);
     type_t mu_diff=1;//производная функции построения выхода слоя по mu
     //по этим данным получаем ошибку на входе слоя
     type_t mu_delta_out=mu_delta*mu_diff;
     mu_delta_out+=-1.0/2.0*(0-2*mu)*k;//добавляем ошибку по расстоянию Кульбака-Лейбнера
     cTensor_PrevLayerMuError.SetElement(w,z,y,x,mu_delta_out);
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
void CNetLayerVAECoderOutput<type_t>::ClipWeight(type_t min,type_t max)
{
}

//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerMuPtr!=NULL) PrevLayerMuPtr->GetOutputTensor().Print(name+" VAECoderOutput: input Mu",false);
 if (PrevLayerLogVarPtr!=NULL) PrevLayerLogVarPtr->GetOutputTensor().Print(name+" VAECoderOutput: input LogVar",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerVAECoderOutput<type_t>::PrintOutputTensorSize(const std::string &name)
{
 GetOutputTensor().Print(name+" VAECoderOutput: output",false);
}

#endif
