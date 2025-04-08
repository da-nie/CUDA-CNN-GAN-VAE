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
static const double EPSILON=0.0000001;

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

  type_t dGamma;///<поправка
  type_t dBeta;///<поправка

  CTensor<type_t> cTensor_XMU;///<отклонение от среднего
  CTensor<type_t> cTensor_XHAT;
  CTensor<type_t> cTensor_DXHAT;
  CTensor<type_t> cTensor_DXMU1;
  CTensor<type_t> cTensor_DXMU2;
  double IVAR;///<обратная дисперсия
  double SQRTVAR;///<сигма
  double VAR;///<дисперсия

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
  bool SaveTrainingParam(IDataStream *iDataStream_Ptr);///<сохранить параметры обучения слоя
  bool LoadTrainingParam(IDataStream *iDataStream_Ptr);///<загрузить параметры обучения слоя

  void TrainingStart(void);///<начать процесс обучения
  void TrainingStop(void);///<завершить процесс обучения
  void TrainingBackward(bool create_delta_weight=true);///<выполнить обратный проход по сети для обучения
  void TrainingResetDeltaWeight(void);///<сбросить поправки к весам
  void TrainingUpdateWeight(double speed,double iteration);///<выполнить обновления весов
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
/*
 #step1: calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
*/
 CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();

 size_t input_x=input.GetSizeX();
 size_t input_y=input.GetSizeY();
 size_t input_z=input.GetSizeZ();

 type_t N=static_cast<type_t>(input_x*input_y*input_z);
 //type_t D=static_cast<type_t>(1);

 //считаем среднее
 double mu=0;//mu = 1./N * np.sum(x, axis = 0)
 for(size_t z=0;z<input_z;z++)
 {
  for(size_t y=0;y<input_y;y++)
  {
   for(size_t x=0;x<input_x;x++)
   {
	mu+=input.GetElement(z,y,x);
   }
  }
 }
 mu/=N;
 //считаем отклонение элемента от среднего и дисперсию
 VAR=0;
 for(size_t z=0;z<input_z;z++)
 {
  for(size_t y=0;y<input_y;y++)
  {
   for(size_t x=0;x<input_x;x++)
   {
	type_t e=input.GetElement(z,y,x);
	type_t xmu=e-mu;//xmu = x - mu
	cTensor_XMU.SetElement(z,y,x,xmu);
	//sq = xmu ** 2
	VAR+=xmu*xmu;//var = 1./N * np.sum(sq, axis = 0)
   }
  }
 }
 VAR/=N;
 SQRTVAR=sqrt(VAR+EPSILON);//sqrtvar = np.sqrt(var + eps)
 IVAR=1.0/SQRTVAR;//ivar = 1./sqrtvar
 //делаем нормализацию
 for(size_t z=0;z<input_z;z++)
 {
  for(size_t y=0;y<input_y;y++)
  {
   for(size_t x=0;x<input_x;x++)
   {
	type_t xmu=cTensor_XMU.GetElement(z,y,x);
	type_t xhat=xmu*IVAR;//xhat = xmu * ivar
	cTensor_XHAT.SetElement(z,y,x,xhat);
	type_t gammax=Gamma*xhat;//gammax = gamma * xhat
	type_t out=gammax+Beta;//out = gammax + beta
	cTensor_H.SetElement(z,y,x,out);
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
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerBatchNormalization<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
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
bool CNetLayerBatchNormalization<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
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

 cTensor_XMU=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_XHAT=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_DXHAT=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());

 cTensor_DXMU1=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());;
 cTensor_DXMU2=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());;
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
 cTensor_XMU=CTensor<type_t>(1,1,1);
 cTensor_XHAT=CTensor<type_t>(1,1,1);
 cTensor_DXHAT=CTensor<type_t>(1,1,1);

 cTensor_DXMU1=CTensor<type_t>(1,1,1);
 cTensor_DXMU2=CTensor<type_t>(1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingBackward(bool create_delta_weight)
{
/*
 #unfold the variables stored in cache
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  #get the dimensions of the input/output
  N,D = dout.shape

  #step9
  dbeta = np.sum(dout, axis=0)
  dgammax = dout #not necessary, but more understandable

  #step8
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step7
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar

  #step6
  dsqrtvar = -1. /(sqrtvar**2) * divar

  #step5
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step4
  dsq = 1. /N * np.ones((N,D)) * dvar

  #step3
  dxmu2 = 2 * xmu * dsq

  #step2
  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

  #step1
  dx2 = 1. /N * np.ones((N,D)) * dmu

  #step0
  dx = dx1 + dx2
*/
 CTensor<type_t> &dout=cTensor_Delta;

 size_t dout_x=dout.GetSizeX();
 size_t dout_y=dout.GetSizeY();
 size_t dout_z=dout.GetSizeZ();

 type_t N=static_cast<type_t>(dout_x*dout_y*dout_z);
 //type_t D=static_cast<type_t>(1);

 //считаем dbeta,dxhat
 double dbeta=0;
 double dgamma=0;
 for(size_t z=0;z<dout_z;z++)
 {
  for(size_t y=0;y<dout_y;y++)
  {
   for(size_t x=0;x<dout_x;x++)
   {
	type_t e=dout.GetElement(z,y,x);
	type_t xhat=cTensor_XHAT.GetElement(z,y,x);
	dbeta+=e;//dbeta = np.sum(dout, axis=0)
	dgamma+=e*xhat;//dgamma = np.sum(dout*xhat, axis=0)
	type_t dxhat=e*Gamma;//dxhat = dout * gamma
	cTensor_DXHAT.SetElement(z,y,x,dxhat);
	type_t dxmul1=dxhat*IVAR;//dxmu1 = dxhat * ivar
	cTensor_DXMU1.SetElement(z,y,x,dxmul1);
   }
  }
 }
 //считаем divar
 double divar=0;
 for(size_t z=0;z<dout_z;z++)
 {
  for(size_t y=0;y<dout_y;y++)
  {
   for(size_t x=0;x<dout_x;x++)
   {
	type_t dxhat=cTensor_DXHAT.GetElement(z,y,x);
	type_t xmu=cTensor_XMU.GetElement(z,y,x);
	divar+=dxhat*xmu;//divar = np.sum(dxhat*xmu, axis=0)
   }
  }
 }
 double dsqrtvar=-1.0/(SQRTVAR*SQRTVAR)*divar;//dsqrtvar = -1. /(sqrtvar**2) * divar
 double dvar=0.5*1.0/sqrt(VAR+EPSILON)*dsqrtvar;//dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

 //считаем dxmu2 и dmu
 double dmu=0;
 for(size_t z=0;z<dout_z;z++)
 {
  for(size_t y=0;y<dout_y;y++)
  {
   for(size_t x=0;x<dout_x;x++)
   {
	type_t xmu=cTensor_XMU.GetElement(z,y,x);
	type_t dsq=dvar*1.0/N;//dsq = 1. /N * np.ones((N,D)) * dvar
    type_t dxmu2=2.0*xmu*dsq;//dxmu2 = 2 * xmu * dsq
    cTensor_DXMU2.SetElement(z,y,x,dxmu2);
    type_t dxmu1=cTensor_DXMU1.GetElement(z,y,x);
    dmu+=-1.0*(dxmu2+dxmu1);//dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
   }
  }
 }
 //формируем тензор ошибки
 for(size_t z=0;z<dout_z;z++)
 {
  for(size_t y=0;y<dout_y;y++)
  {
   for(size_t x=0;x<dout_x;x++)
   {
    type_t dxmu1=cTensor_DXMU1.GetElement(z,y,x);
    type_t dxmu2=cTensor_DXMU2.GetElement(z,y,x);
    type_t dx1=dxmu1+dxmu2;//dx1 = (dxmu1 + dxmu2)
    type_t dx2=dmu/N;
    cTensor_PrevLayerError.SetElement(z,y,x,dx1+dx2);
   }
  }
 }
 if (create_delta_weight==true)
 {
  dGamma+=dgamma;
  dBeta+=dbeta;
 }
 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingResetDeltaWeight(void)
{
 dGamma=0;
 dBeta=0;
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingUpdateWeight(double speed,double iteration)
{
 Gamma-=dGamma*speed;
 Beta-=dBeta*speed;

 printf("dGamma[%f]:%f dBeta[%f]:%f\r\n",Gamma,dGamma,Beta,dBeta);
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
