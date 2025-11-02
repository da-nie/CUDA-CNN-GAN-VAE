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
  uint32_t Layer;


  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  uint32_t BatchSize;///<размер пакета для обучения

  CTensor<type_t> cTensor_Beta;///<параметр сдвига
  CTensor<type_t> cTensor_Gamma;///<параметр масштабирования
  CTensor<type_t> cTensor_Mean;///<среднее
  CTensor<type_t> cTensor_Variable;///<дисперсия

  CTensor<type_t> cTensor_NewMean;///<новое среднее
  CTensor<type_t> cTensor_NewVariable;///<новая дисперсия

  CTensor<type_t> cTensor_H_Array;///<тензор выхода слоя
  CTensor<type_t> cTensor_TmpA;///<промежуточный тензор, размерности H по (x,y,z), но не по w
  CTensor<type_t> cTensor_TmpA_H;///<промежуточный тензор, размерности H

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta_Array;///<тензор ошибки предыдущего слоя
  CTensor<type_t> cTensor_PrevLayerError_Array;///<тензор ошибки предыдущего слоя
  CTensor<type_t> cTensor_XHAT_Array;
  CTensor<type_t> cTensor_DXHAT_Array;
  CTensor<type_t> cTensor_TmpB;
  CTensor<type_t> cTensor_TmpC;
  CTensor<type_t> cTensor_IVAR;///<обратная дисперсия
  CTensor<type_t> cTensor_dGamma;///<поправка
  CTensor<type_t> cTensor_dBeta;///<поправка
  type_t Momentum;///<фильтрация дисперсии и среднего
  type_t NewMean;///<новое значение среднего
  type_t NewVariable;///<новое значение дисперсии
  bool TrainingEnabled;///<включено ли обучение

  //для оптимизации Adam
  CTensor<type_t> cTensor_MK;///<тензор фильтра 1
  CTensor<type_t> cTensor_VK;///<тензор фильтра 2
  CTensor<type_t> cTensor_MB;///<коэффициент фильтра 1 сдвигов
  CTensor<type_t> cTensor_VB;///<коэффициент фильтра 2 сдвигов

  using INetLayer<type_t>::Beta1;///<параметры алгоритма Adam
  using INetLayer<type_t>::Beta2;
  using INetLayer<type_t>::Epsilon;
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerBatchNormalization(type_t momentum,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerBatchNormalization(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerBatchNormalization();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(type_t momentum,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
  void Reset(void);///<выполнить инициализацию весов и сдвигов
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
CNetLayerBatchNormalization<type_t>::CNetLayerBatchNormalization(type_t momentum,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
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
\param[in] momentum Коэффициент фильтрации при обработке слоя
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::Create(type_t momentum,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 static uint32_t i=0;
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

 cTensor_H_Array=CTensor<type_t>(BatchSize,PrevLayerPtr->GetOutputTensor().GetSizeZ(),PrevLayerPtr->GetOutputTensor().GetSizeY(),PrevLayerPtr->GetOutputTensor().GetSizeX());
 cTensor_Delta_Array=CTensor<type_t>(BatchSize,PrevLayerPtr->GetOutputTensor().GetSizeZ(),PrevLayerPtr->GetOutputTensor().GetSizeY(),PrevLayerPtr->GetOutputTensor().GetSizeX());
 cTensor_XHAT_Array=CTensor<type_t>(BatchSize,PrevLayerPtr->GetOutputTensor().GetSizeZ(),PrevLayerPtr->GetOutputTensor().GetSizeY(),PrevLayerPtr->GetOutputTensor().GetSizeX());

 cTensor_Mean=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());
 cTensor_Variable=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());

 cTensor_NewMean=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());
 cTensor_NewVariable=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());

 cTensor_IVAR=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());

 cTensor_Gamma=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());
 cTensor_Beta=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());

 cTensor_TmpA=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());
 cTensor_TmpB=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());
 cTensor_TmpC=CTensor<type_t>(1,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());

 cTensor_TmpA_H=CTensor<type_t>(BatchSize,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());

 //задаём предшествующему слою, что мы его последующий слой
 prev_layer_ptr->SetNextLayerPtr(this);

 TrainingEnabled=false;
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию весов и сдвигов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::Reset(void)
{
 CTensorMath<type_t>::Fill(cTensor_Gamma,1);
 CTensorMath<type_t>::Fill(cTensor_Beta,0);

 CTensorMath<type_t>::Fill(cTensor_Mean,0);
 CTensorMath<type_t>::Fill(cTensor_Variable,1);
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
 cTensor_H_Array=output;
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
 output=cTensor_H_Array;
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
bool
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
 CTensor<type_t> &cTensor_VAR=cTensor_TmpB;
 if (TrainingEnabled==true)//режим обучения
 {
  //считаем среднее для каждого элемента пакета по всем пакетам
  //mu = 1./N * np.sum(x, axis = 0)
  CTensorMath<type_t>::Fill(cTensor_TmpA,0);
  CTensorMath<type_t>::AddSumW(cTensor_TmpA,cTensor_TmpA,PrevLayerPtr->GetOutputTensor(),1,1.0/N);

  CTensorMath<type_t>::Add(cTensor_NewMean,cTensor_NewMean,cTensor_TmpA,Momentum,1.0-Momentum);

  //считаем разность от среднего для каждого пакета
  //xmu = x - mu
  CTensorMath<type_t>::Sub(cTensor_XHAT_Array,PrevLayerPtr->GetOutputTensor(),cTensor_TmpA,1.0,1.0);

  //возводим в квадрат xmu и вычисляем дисперсию
  //sq = xmu ** 2
  //var = 1./N * np.sum(sq, axis = 0)
  CTensorMath<type_t>::Fill(cTensor_VAR,0);
  CTensorMath<type_t>::Pow2(cTensor_TmpA_H,cTensor_XHAT_Array,1);
  CTensorMath<type_t>::AddSumW(cTensor_VAR,cTensor_VAR,cTensor_TmpA_H,1,1.0/N);

  CTensorMath<type_t>::Add(cTensor_NewVariable,cTensor_NewVariable,cTensor_VAR,Momentum,1.0-Momentum);
 }
 else//режим работы
 {
  cTensor_VAR=cTensor_Variable;
  //считаем разность от среднего
  CTensorMath<type_t>::Sub(cTensor_XHAT_Array,PrevLayerPtr->GetOutputTensor(),cTensor_Mean,1.0,1.0);
 }


 //printf("VAR:%f Middle:%f\r\n",cTensor_VAR.GetElement(0,0,0),cTensor_TmpA.GetElement(0,0,0));

 /*cTensor_VAR=cTensor_Variable;
 for(uint32_t n=0;n<BatchSize;n++)
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
 //xhat = xmu * ivar
 CTensorMath<type_t>::TensorItemProduction(cTensor_XHAT_Array,cTensor_XHAT_Array,cTensor_IVAR);
 //считаем выход
 //gammax = gamma * xhat
 CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA_H,cTensor_Gamma,cTensor_XHAT_Array);
 //out = gammax + beta
 CTensorMath<type_t>::Add(cTensor_H_Array,cTensor_TmpA_H,cTensor_Beta);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerBatchNormalization<type_t>::GetOutputTensor(void)
{
 return(cTensor_H_Array);
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
bool CNetLayerBatchNormalization<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 cTensor_Beta.Load(iDataStream_Ptr,check_size);
 cTensor_Gamma.Load(iDataStream_Ptr,check_size);
 cTensor_Mean.Load(iDataStream_Ptr,check_size);
 cTensor_Variable.Load(iDataStream_Ptr,check_size);
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
 CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor();

 cTensor_dGamma=CTensor<type_t>(1,prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_dBeta=CTensor<type_t>(1,prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());

 cTensor_DXHAT_Array=CTensor<type_t>(BatchSize,PrevLayerPtr->GetOutputTensor().GetSizeZ(),PrevLayerPtr->GetOutputTensor().GetSizeY(),PrevLayerPtr->GetOutputTensor().GetSizeX());

 cTensor_PrevLayerError_Array=CTensor<type_t>(BatchSize,cTensor_H_Array.GetSizeZ(),cTensor_H_Array.GetSizeY(),cTensor_H_Array.GetSizeX());

 //для оптимизации Adam
 cTensor_MK=cTensor_dGamma;
 cTensor_VK=cTensor_dGamma;
 CTensorMath<type_t>::Fill(cTensor_MK,0);
 CTensorMath<type_t>::Fill(cTensor_VK,0);

 cTensor_MB=cTensor_dBeta;
 cTensor_VB=cTensor_dBeta;
 CTensorMath<type_t>::Fill(cTensor_MB,0);
 CTensorMath<type_t>::Fill(cTensor_VB,0);

 TrainingEnabled=true;
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingStop(void)
{
 cTensor_DXHAT_Array=CTensor<type_t>(1,1,1,1);
 cTensor_PrevLayerError_Array=CTensor<type_t>(1,1,1,1);

 cTensor_MK=CTensor<type_t>(1,1,1,1);
 cTensor_VK=CTensor<type_t>(1,1,1,1);
 cTensor_MB=CTensor<type_t>(1,1,1,1);
 cTensor_VB=CTensor<type_t>(1,1,1,1);

 TrainingEnabled=false;
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
 CTensorMath<type_t>::Fill(cTensor_TmpB,0);//будет np.sum(dxhat, axis=0)
 CTensorMath<type_t>::Fill(cTensor_TmpC,0);//будет np.sum(dxhat*x_hat, axis=0)

 CTensorMath<type_t>::TensorItemProduction(cTensor_DXHAT_Array,cTensor_Delta_Array,cTensor_Gamma);//dxhat = dout * gamma
 CTensorMath<type_t>::AddSumW(cTensor_TmpB,cTensor_TmpB,cTensor_DXHAT_Array);//np.sum(dxhat, axis=0)

 CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA_H,cTensor_DXHAT_Array,cTensor_XHAT_Array);//dxhat*x_hat
 CTensorMath<type_t>::AddSumW(cTensor_TmpC,cTensor_TmpC,cTensor_TmpA_H);//np.sum(dxhat*x_hat, axis=0)
 if (create_delta_weight==true)
 {
  //dgamma = np.sum(dout*xhat, axis=0)
  CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA_H,cTensor_Delta_Array,cTensor_XHAT_Array);
  CTensorMath<type_t>::AddSumW(cTensor_dGamma,cTensor_dGamma,cTensor_TmpA_H);
  //dbeta = np.sum(dout, axis=0)
  CTensorMath<type_t>::AddSumW(cTensor_dBeta,cTensor_dBeta,cTensor_Delta_Array);
 }

 CTensorMath<type_t>::Mul(cTensor_PrevLayerError_Array,cTensor_DXHAT_Array,N);//N*dxhat
 CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA_H,cTensor_XHAT_Array,cTensor_TmpC);//x_hat*np.sum(dxhat*x_hat, axis=0)

 CTensorMath<type_t>::Sub(cTensor_PrevLayerError_Array,cTensor_PrevLayerError_Array,cTensor_TmpA_H);
 CTensorMath<type_t>::Sub(cTensor_PrevLayerError_Array,cTensor_PrevLayerError_Array,cTensor_TmpB);
 CTensorMath<type_t>::TensorItemProduction(cTensor_PrevLayerError_Array,cTensor_PrevLayerError_Array,cTensor_IVAR);
 CTensorMath<type_t>::Mul(cTensor_PrevLayerError_Array,cTensor_PrevLayerError_Array,1.0/N);

 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError_Array);

 //printf("Layer:%i Gamma:%f Beta:%f -> ",static_cast<int>(Layer),cTensor_Gamma.GetElement(0,0,0),cTensor_Beta.GetElement(0,0,0));
 //printf("dGamma:%f dBeta:%f\r\n",cTensor_dGamma.GetElement(0,0,0),cTensor_dBeta.GetElement(0,0,0));

}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingResetDeltaWeight(void)
{
 CTensorMath<type_t>::Fill(cTensor_dGamma,0);
 CTensorMath<type_t>::Fill(cTensor_dBeta,0);

 cTensor_NewMean=cTensor_Mean;
 cTensor_NewVariable=cTensor_Variable;
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
/*
 printf("Layer:%i Gamma:%f Beta:%f -> ",Layer,cTensor_Gamma.GetElement(0,0,0,0),cTensor_Beta.GetElement(0,0,0,0));
 printf("dGamma:%f dBeta:%f\r\n",cTensor_dGamma.GetElement(0,0,0,0),cTensor_dBeta.GetElement(0,0,0,0));
 */

 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_ADAM)
 {
  //применяем алгоритм Adam
  CTensorMath<type_t>::Adam(cTensor_Gamma,cTensor_dGamma,cTensor_MK,cTensor_VK,BatchSize*batch_scale,speed,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_Beta,cTensor_dBeta,cTensor_MB,cTensor_VB,BatchSize*batch_scale,speed,Beta1,Beta2,Epsilon,iteration);
 }
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_GRADIENT)
 {
  speed/=static_cast<double>(BatchSize);
  CTensorMath<type_t>::Sub(cTensor_Gamma,cTensor_Gamma,cTensor_dGamma,1,speed/batch_scale);
  CTensorMath<type_t>::Sub(cTensor_Beta,cTensor_Beta,cTensor_dBeta,1,speed/batch_scale);
 }

 //printf("Layer:%i NewGamma:%f NewBeta:%f\r\n",Layer,cTensor_Gamma.GetElement(0,0,0),cTensor_Beta.GetElement(0,0,0));


 cTensor_Mean=cTensor_NewMean;
 cTensor_Variable=cTensor_NewVariable;
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerBatchNormalization<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta_Array);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::SetOutputError(CTensor<type_t>& error)
{
 cTensor_Delta_Array=error;
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
//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" BatchNormalization: input",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::PrintOutputTensorSize(const std::string &name)
{
 GetOutputTensor().Print(name+" BatchNormalization: output",false);
}

#endif
