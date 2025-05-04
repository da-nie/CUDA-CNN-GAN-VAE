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
static const double EPSILON=1E-6;

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
  size_t Layer;


  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  size_t BatchSize;///<размер пакета для обучения

  CTensor<type_t> cTensor_Beta;///<параметр сдвига
  CTensor<type_t> cTensor_Gamma;///<параметр масштабирования
  CTensor<type_t> cTensor_Mean;///<среднее
  CTensor<type_t> cTensor_Variable;///<дисперсия

  CTensor<type_t> cTensor_NewMean;///<новое среднее
  CTensor<type_t> cTensor_NewVariable;///<новая дисперсия

  std::vector<CTensor<type_t>> cTensor_H_Array;///<тензор выхода слоя
  CTensor<type_t> cTensor_TmpA;///<промежуточный тензор, размерности H

  //тензоры, используемые при обучении
  std::vector<CTensor<type_t>> cTensor_Delta_Array;///<тензор ошибки предыдущего слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя
  std::vector<CTensor<type_t>> cTensor_XHAT_Array;
  std::vector<CTensor<type_t>> cTensor_DXHAT_Array;
  CTensor<type_t> cTensor_TmpB;
  CTensor<type_t> cTensor_TmpC;
  CTensor<type_t> cTensor_IVAR;///<обратная дисперсия
  CTensor<type_t> cTensor_dGamma;///<поправка
  CTensor<type_t> cTensor_dBeta;///<поправка
  type_t Momentum;///<фильтрация дисперсии и среднего
  type_t NewMean;///<новое значение среднего
  type_t NewVariable;///<новое значение дисперсии

  //для оптимизации Adam
  CTensor<type_t> cTensor_MK;///<тензор фильтра 1
  CTensor<type_t> cTensor_VK;///<тензор фильтра 2
  CTensor<type_t> cTensor_MB;///<коэффициент фильтра 1 сдвигов
  CTensor<type_t> cTensor_VB;///<коэффициент фильтра 2 сдвигов
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerBatchNormalization(type_t momentum,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);
  CNetLayerBatchNormalization(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerBatchNormalization();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(type_t momentum,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);///<создать слой
  void Reset(void);///<выполнить инициализацию весов и сдвигов
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
CNetLayerBatchNormalization<type_t>::CNetLayerBatchNormalization(type_t momentum,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 Create(momentum,prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerBatchNormalization<type_t>::CNetLayerBatchNormalization(void)
{
 Create(0.9);
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
void CNetLayerBatchNormalization<type_t>::Create(type_t momentum,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 static size_t i=0;
 i++;
 Layer=i;

 if (prev_layer_ptr==NULL)//слой без предшествующего считается входным
 {
  throw("Слой BatchNormalization не может быть входным!");
 }

 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 Momentum=momentum;

 cTensor_H_Array.resize(BatchSize);
 cTensor_Delta_Array.resize(BatchSize);
 cTensor_XHAT_Array.resize(BatchSize);
 for(size_t n=0;n<BatchSize;n++)
 {
  cTensor_H_Array[n]=CTensor<type_t>(PrevLayerPtr->GetOutputTensor(n).GetSizeZ(),PrevLayerPtr->GetOutputTensor(n).GetSizeY(),PrevLayerPtr->GetOutputTensor(n).GetSizeX());
  cTensor_Delta_Array[n]=CTensor<type_t>(PrevLayerPtr->GetOutputTensor(n).GetSizeZ(),PrevLayerPtr->GetOutputTensor(n).GetSizeY(),PrevLayerPtr->GetOutputTensor(n).GetSizeX());
  cTensor_XHAT_Array[n]=CTensor<type_t>(PrevLayerPtr->GetOutputTensor(n).GetSizeZ(),PrevLayerPtr->GetOutputTensor(n).GetSizeY(),PrevLayerPtr->GetOutputTensor(n).GetSizeX());
 }

 cTensor_Mean=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_Variable=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 cTensor_NewMean=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_NewVariable=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 cTensor_IVAR=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 cTensor_Gamma=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_Beta=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 cTensor_TmpA=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

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
 cTensor_Gamma.Fill(1);
 cTensor_Beta.Zero();

 cTensor_Mean.Zero();
 cTensor_Variable.Fill(1);
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::SetOutput(size_t unit_index,CTensor<type_t> &output)
{
 cTensor_H_Array[unit_index]=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::GetOutput(size_t unit_index,CTensor<type_t> &output)
{
 output=cTensor_H_Array[unit_index];
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

  #step2: subtract mean vector of every Deltatrainings example
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
  cache = (xhat,gamma,xmu,ivar,sqDeltartvar,var,eps)
*/
 type_t N=static_cast<type_t>(BatchSize);
 //используем Tensor_XHAT_Array вместо Tensor_XMU_Array - всё равно Tensor_XMU_Array нужно только здесь, а Tensor_XHAT_Array до заполнения не используется

 //считаем среднее для каждого элемента пакета по всем пакетам
 //mu = 1./N * np.sum(x, axis = 0)
 cTensor_TmpA.Zero();
 for(size_t n=0;n<BatchSize;n++)
 {
  CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor(n);
  CTensorMath<type_t>::Add(cTensor_TmpA,cTensor_TmpA,input,1.0,1.0/N);
 }

 //считаем разность от среднего для каждого пакета
 //xmu = x - mu
 CTensorMath<type_t>::Add(cTensor_NewMean,cTensor_NewMean,cTensor_TmpA,Momentum,1.0-Momentum);

 for(size_t n=0;n<BatchSize;n++)
 {
  CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor(n);
  CTensorMath<type_t>::Sub(cTensor_XHAT_Array[n],input,cTensor_TmpA,1.0,1.0);
 }

 //возводим в квадрат xmu и вычисляем дисперсию
 //sq = xmu ** 2
 //var = 1./N * np.sum(sq, axis = 0)
 CTensor<type_t> &cTensor_VAR=cTensor_TmpB;
 cTensor_VAR.Zero();
 for(size_t n=0;n<BatchSize;n++)
 {
  CTensorMath<type_t>::Pow2(cTensor_TmpA,cTensor_XHAT_Array[n],1);
  CTensorMath<type_t>::Add(cTensor_VAR,cTensor_VAR,cTensor_TmpA,1,1.0/N);
 }

 CTensorMath<type_t>::Add(cTensor_NewVariable,cTensor_NewVariable,cTensor_VAR,Momentum,1.0-Momentum);

 //printf("VAR:%f Middle:%f\r\n",cTensor_VAR.GetElement(0,0,0),cTensor_TmpA.GetElement(0,0,0));

 /*cTensor_VAR=cTensor_Variable;
 for(size_t n=0;n<BatchSize;n++)
 {
  CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor(n);
  CTensorMath<type_t>::Sub(cTensor_XMU_Array[n],input,cTensor_NewMean,1.0,1.0);
 }
 */

 //считаем среднеквадратичное отклонение
 //sqrtvar = np.sqrt(var + eps)
 CTensor<type_t> &cTensor_SQRTVAR=cTensor_TmpC;
 CTensorMath<type_t>::SQRT(cTensor_SQRTVAR,cTensor_VAR,1,EPSILON);
 //считаем обратный тензор к среднеквадратичному отклонению
 //ivar = 1./sqrtvar
 CTensorMath<type_t>::Inv(cTensor_IVAR,cTensor_SQRTVAR);

 //считаем xhat и выход слоя
 for(size_t n=0;n<BatchSize;n++)
 {
  //xhat = xmu * ivar
  CTensorMath<type_t>::TensorItemProduction(cTensor_XHAT_Array[n],cTensor_XHAT_Array[n],cTensor_IVAR);
  //считаем выход
  //gammax = gamma * xhat
  CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,cTensor_Gamma,cTensor_XHAT_Array[n]);
  //out = gammax + beta
  CTensorMath<type_t>::Add(cTensor_H_Array[n],cTensor_TmpA,cTensor_Beta);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerBatchNormalization<type_t>::GetOutputTensor(size_t unit_index)
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
 cTensor_Beta.Save(iDataStream_Ptr);
 cTensor_Gamma.Save(iDataStream_Ptr);
 cTensor_Mean.Save(iDataStream_Ptr);
 cTensor_Variable.Save(iDataStream_Ptr);
 iDataStream_Ptr->SaveDouble(Momentum);
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
 cTensor_Beta.Load(iDataStream_Ptr);
 cTensor_Gamma.Load(iDataStream_Ptr);
 cTensor_Mean.Load(iDataStream_Ptr);
 cTensor_Variable.Load(iDataStream_Ptr);
 Momentum=iDataStream_Ptr->LoadDouble();
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
 cTensor_MK.Save(iDataStream_Ptr);
 cTensor_VK.Save(iDataStream_Ptr);
 cTensor_MB.Save(iDataStream_Ptr);
 cTensor_VB.Save(iDataStream_Ptr);
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
 cTensor_MK.Load(iDataStream_Ptr);
 cTensor_VK.Load(iDataStream_Ptr);
 cTensor_MB.Load(iDataStream_Ptr);
 cTensor_VB.Load(iDataStream_Ptr);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingStart(void)
{
 CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor(0);

 cTensor_dGamma=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_dBeta=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());

 cTensor_DXHAT_Array.resize(BatchSize);
 for(size_t n=0;n<BatchSize;n++)
 {
  cTensor_DXHAT_Array[n]=CTensor<type_t>(PrevLayerPtr->GetOutputTensor(n).GetSizeZ(),PrevLayerPtr->GetOutputTensor(n).GetSizeY(),PrevLayerPtr->GetOutputTensor(n).GetSizeX());
 }
 cTensor_PrevLayerError=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_TmpB=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_TmpC=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 //для оптимизации Adam
 cTensor_MK=cTensor_dGamma;
 cTensor_VK=cTensor_dGamma;
 cTensor_MK.Zero();
 cTensor_VK.Zero();

 cTensor_MB=cTensor_dBeta;
 cTensor_VB=cTensor_dBeta;
 cTensor_MB.Zero();
 cTensor_VB.Zero();
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingStop(void)
{
 cTensor_DXHAT_Array.clear();
 cTensor_PrevLayerError=CTensor<type_t>(1,1,1);
 cTensor_TmpB=CTensor<type_t>(1,1,1);
 cTensor_TmpC=CTensor<type_t>(1,1,1);

 cTensor_MK=CTensor<type_t>(1,1,1);
 cTensor_VK=CTensor<type_t>(1,1,1);
 cTensor_MB=CTensor<type_t>(1,1,1);
 cTensor_VB=CTensor<type_t>(1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingBackward(bool create_delta_weight)
{
//Вариант 1
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

  #step0Variable
  dx = dx1 + dx2
*/

 //Вариант 2
 /*
	N, D = dout.shape
	x_mu, inv_var, x_hat, gamma = cache

	# intermediate partial derivatives
	dxhat = dout * gamma

	# final partial derivatives
	dx = (1. / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=0)
		- x_hat*np.sum(dxhat*x_hat, axis=0))
	dbeta = np.sum(dout, axis=0)
	dgamma = np.sum(x_hat*dout, axis=0)

	return dx, dgamma, dbeta
	*/
 //В варианте 2 есть недостающая часть. По экспериментам она нулевая и потому убрана из программы (осталась в git).

 type_t N=static_cast<type_t>(BatchSize);
 cTensor_TmpB.Zero();//будет np.sum(dxhat, axis=0)
 cTensor_TmpC.Zero();//будет np.sum(dxhat*x_hat, axis=0)
 for(size_t n=0;n<BatchSize;n++)
 {
  CTensor<type_t> &dout=cTensor_Delta_Array[n];
  CTensorMath<type_t>::TensorItemProduction(cTensor_DXHAT_Array[n],dout,cTensor_Gamma);//dxhat = dout * gamma
  CTensorMath<type_t>::Add(cTensor_TmpB,cTensor_TmpB,cTensor_DXHAT_Array[n]);//np.sum(dxhat, axis=0)

  CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,cTensor_DXHAT_Array[n],cTensor_XHAT_Array[n]);//dxhat*x_hat
  CTensorMath<type_t>::Add(cTensor_TmpC,cTensor_TmpC,cTensor_TmpA);//np.sum(dxhat*x_hat, axis=0)
  if (create_delta_weight==true)
  {
   //dgamma = np.sum(dout*xhat, axis=0)
   CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,dout,cTensor_XHAT_Array[n]);
   CTensorMath<type_t>::Add(cTensor_dGamma,cTensor_dGamma,cTensor_TmpA);
   //dbeta = np.sum(dout, axis=0)
   CTensorMath<type_t>::Add(cTensor_dBeta,cTensor_dBeta,dout);
  }
 }
 for(size_t n=0;n<BatchSize;n++)
 {
  CTensorMath<type_t>::Mul(cTensor_PrevLayerError,cTensor_DXHAT_Array[n],N);//N*dxhat
  CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,cTensor_XHAT_Array[n],cTensor_TmpC);//x_hat*np.sum(dxhat*x_hat, axis=0)
  CTensorMath<type_t>::Sub(cTensor_PrevLayerError,cTensor_PrevLayerError,cTensor_TmpA);
  CTensorMath<type_t>::Sub(cTensor_PrevLayerError,cTensor_PrevLayerError,cTensor_TmpB);
  CTensorMath<type_t>::TensorItemProduction(cTensor_PrevLayerError,cTensor_PrevLayerError,cTensor_IVAR);
  CTensorMath<type_t>::Mul(cTensor_PrevLayerError,cTensor_PrevLayerError,1.0/N);
  PrevLayerPtr->SetOutputError(n,cTensor_PrevLayerError);
 }

 printf("Layer:%i Gamma:%f Beta:%f -> ",Layer,cTensor_Gamma.GetElement(0,0,0),cTensor_Beta.GetElement(0,0,0));
 printf("dGamma:%f dBeta:%f\r\n",cTensor_dGamma.GetElement(0,0,0),cTensor_dBeta.GetElement(0,0,0));

}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingResetDeltaWeight(void)
{
 cTensor_dGamma.Zero();
 cTensor_dBeta.Zero();

 cTensor_NewMean=cTensor_Mean;
 cTensor_NewVariable=cTensor_Variable;
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingUpdateWeight(double speed,double iteration)
{
/*
 printf("Layer:%i Gamma:%f Beta:%f -> ",Layer,cTensor_Gamma.GetElement(0,0,0),cTensor_Beta.GetElement(0,0,0));
 printf("dGamma:%f dBeta:%f\r\n",cTensor_dGamma.GetElement(0,0,0),cTensor_dBeta.GetElement(0,0,0));
*/
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_ADAM)
 {
  double beta1=0.9;
  double beta2=0.999;
  static const double epsilon=1E-8;
  //применяем алгоритм Adam
  CTensorMath<type_t>::Adam(cTensor_Gamma,cTensor_dGamma,cTensor_MK,cTensor_VK,speed,beta1,beta2,epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_Beta,cTensor_dBeta,cTensor_MB,cTensor_VB,speed,beta1,beta2,epsilon,iteration);
 }
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_GRADIENT)
 {
  CTensorMath<type_t>::Sub(cTensor_Gamma,cTensor_Gamma,cTensor_dGamma,1.0,speed);
  CTensorMath<type_t>::Sub(cTensor_Beta,cTensor_Beta,cTensor_dBeta,1.0,speed);
 }

 cTensor_Mean=cTensor_NewMean;
 cTensor_Variable=cTensor_NewVariable;
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerBatchNormalization<type_t>::GetDeltaTensor(size_t unit_index)
{
 return(cTensor_Delta_Array[unit_index]);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::SetOutputError(size_t unit_index,CTensor<type_t>& error)
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
void CNetLayerBatchNormalization<type_t>::ClipWeight(type_t min,type_t max)
{
}


#endif
