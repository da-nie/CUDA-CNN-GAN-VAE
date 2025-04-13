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
static const double EPSILON=0.0000000001;

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
  std::vector<CTensor<type_t>> cTensor_XMU_Array;///<отклонение от среднего
  std::vector<CTensor<type_t>> cTensor_XHAT_Array;
  std::vector<CTensor<type_t>> cTensor_DXHAT_Array;
  CTensor<type_t> cTensor_DXMU1;
  CTensor<type_t> cTensor_DXMU2;
  CTensor<type_t> cTensor_IVAR;///<обратная дисперсия
  CTensor<type_t> cTensor_DIVAR;///<приращение обратной дисперсии
  CTensor<type_t> cTensor_SQRTVAR;///<сигма
  CTensor<type_t> cTensor_VAR;///<тензор дисперсии
  CTensor<type_t> cTensor_DVAR;///<тензор приращения дисперсии
  CTensor<type_t> cTensor_dGamma;///<поправка
  CTensor<type_t> cTensor_dBeta;///<поправка
  type_t Momentum;///<фильтрация дисперсии и среднего
  type_t NewMean;///<новое значение среднего
  type_t NewVariable;///<новое значение дисперсии

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
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 Momentum=momentum;

 cTensor_H_Array.resize(BatchSize);
 cTensor_Delta_Array.resize(BatchSize);
 cTensor_XMU_Array.resize(BatchSize);
 cTensor_XHAT_Array.resize(BatchSize);
 cTensor_DXHAT_Array.resize(BatchSize);
 for(size_t n=0;n<BatchSize;n++)
 {
  cTensor_H_Array[n]=CTensor<type_t>(PrevLayerPtr->GetOutputTensor(n).GetSizeZ(),PrevLayerPtr->GetOutputTensor(n).GetSizeY(),PrevLayerPtr->GetOutputTensor(n).GetSizeX());
  cTensor_Delta_Array[n]=CTensor<type_t>(PrevLayerPtr->GetOutputTensor(n).GetSizeZ(),PrevLayerPtr->GetOutputTensor(n).GetSizeY(),PrevLayerPtr->GetOutputTensor(n).GetSizeX());
  cTensor_XMU_Array[n]=CTensor<type_t>(PrevLayerPtr->GetOutputTensor(n).GetSizeZ(),PrevLayerPtr->GetOutputTensor(n).GetSizeY(),PrevLayerPtr->GetOutputTensor(n).GetSizeX());
  cTensor_XHAT_Array[n]=CTensor<type_t>(PrevLayerPtr->GetOutputTensor(n).GetSizeZ(),PrevLayerPtr->GetOutputTensor(n).GetSizeY(),PrevLayerPtr->GetOutputTensor(n).GetSizeX());
  cTensor_DXHAT_Array[n]=CTensor<type_t>(PrevLayerPtr->GetOutputTensor(n).GetSizeZ(),PrevLayerPtr->GetOutputTensor(n).GetSizeY(),PrevLayerPtr->GetOutputTensor(n).GetSizeX());
 }

 cTensor_Mean=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_Variable=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 cTensor_NewMean=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_NewVariable=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 cTensor_VAR=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_SQRTVAR=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_IVAR=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 cTensor_Gamma=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_Beta=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 cTensor_TmpA=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

 //нужны только для обучения
 cTensor_DIVAR=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_DVAR=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_DXMU1=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());
 cTensor_DXMU2=CTensor<type_t>(cTensor_H_Array[0].GetSizeZ(),cTensor_H_Array[0].GetSizeY(),cTensor_H_Array[0].GetSizeX());

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
 throw("Слою BatchNormalization нельзя задать выходное значение!");
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
  CTensorMath<type_t>::Sub(cTensor_XMU_Array[n],input,cTensor_TmpA,1.0,1.0);
 }
 //возводим в квадрат xmu и вычисляем дисперсию
 //sq = xmu ** 2
 //var = 1./N * np.sum(sq, axis = 0)
 cTensor_VAR.Zero();
 for(size_t n=0;n<BatchSize;n++)
 {
  CTensorMath<type_t>::Pow2(cTensor_TmpA,cTensor_XMU_Array[n],1);
  CTensorMath<type_t>::Add(cTensor_VAR,cTensor_VAR,cTensor_TmpA,1,1.0/N);
 }
 cTensor_VAR.Fill(1);//временно - хм, дисперсия уходит к нулю...

 CTensorMath<type_t>::Add(cTensor_NewVariable,cTensor_NewVariable,cTensor_VAR,Momentum,1.0-Momentum);
 cTensor_VAR=cTensor_Variable;

 for(size_t n=0;n<BatchSize;n++)
 {
  CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor(n);
  CTensorMath<type_t>::Sub(cTensor_XMU_Array[n],input,cTensor_Mean,1.0,1.0);
 }

 //считаем среднеквадратичное отклонение
 //sqrtvar = np.sqrt(var + eps)
 CTensorMath<type_t>::SQRT(cTensor_SQRTVAR,cTensor_VAR,1,EPSILON);
 //считаем обратный тензор к среднеквадратичному отклонению
 //ivar = 1./sqrtvar
 CTensorMath<type_t>::Inv(cTensor_IVAR,cTensor_SQRTVAR);
 //считаем xhat и выход слоя
 for(size_t n=0;n<BatchSize;n++)
 {
  //xhat = xmu * ivar
  CTensorMath<type_t>::TensorItemProduction(cTensor_XHAT_Array[n],cTensor_XMU_Array[n],cTensor_IVAR);
  //считаем выход
  //gammax = gamma * xhat
  CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,cTensor_Gamma,cTensor_XHAT_Array[n]);
  //out = gammax + beta
  CTensorMath<type_t>::Add(cTensor_H_Array[n],cTensor_TmpA,cTensor_Beta);
 }


/*
 CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();
 size_t input_x=input.GetSizeX();
 size_t input_y=input.GetSizeY();
 size_t input_z=input.GetSizeZ();

 type_t N=static_cast<type_t>(input_x*input_y*input_z);
 //type_t D=static_cast<type_t>(1);

 CTensor<type_t> cTensor_Output(1,1,1);
 //считаем среднее
 //mu = 1./N * np.sum(x, axis = 0)
 CTensorMath<type_t>::Set(cTensor_TmpA,input,1);
 cTensor_TmpA.ReinterpretSize(1,1,input_z*input_y*input_x);
 cTensor_Output.SetElement(0,0,0,0);
 CTensorMath<type_t>::SummXY(cTensor_Output,cTensor_TmpA);
 cTensor_TmpA.RestoreSize();
 double mu=cTensor_Output.GetElement(0,0,0);
 mu/=N;

 //вычисляем xmu
 //xmu = x - mu
 CTensorMath<type_t>::SubValue(cTensor_XMU,input,1,mu);
 //возводим в квадрат
 CTensorMath<type_t>::Pow2(cTensor_TmpA,cTensor_XMU,1);
 //считаем отклонение элемента от среднего и дисперсию
 cTensor_TmpA.ReinterpretSize(1,1,input_z*input_y*input_x);
 cTensor_Output.SetElement(0,0,0,0);
 CTensorMath<type_t>::SummXY(cTensor_Output,cTensor_TmpA);
 cTensor_TmpA.RestoreSize();
 //var = 1./N * np.sum(sq, axis = 0)
 VAR=cTensor_Output.GetElement(0,0,0);
 VAR/=N;

 NewMean=mu*(1.0-Momentum)+NewMean*Momentum;
 NewVariable=VAR*(1.0-Momentum)+NewVariable*Momentum;
 //mu=Mean;
 //VAR=Variable;

 //строим заново тензоры с учётом текущего среднего и дисперсии
 CTensorMath<type_t>::SubValue(cTensor_XMU,input,1,mu);
 //возводим в квадрат
 CTensorMath<type_t>::Pow2(cTensor_TmpA,cTensor_XMU,1);


 SQRTVAR=sqrt(VAR+EPSILON);//sqrtvar = np.sqrt(var + eps)
 IVAR=1.0/SQRTVAR;//ivar = 1./sqrtvar
 //делаем нормировку
 //xhat = xmu * ivar
 CTensorMath<type_t>::Set(cTensor_XHAT,cTensor_XMU,IVAR);
 //вычисляем выход слоя
 //gammax = gamma * xhat
 //out = gammax + beta
 CTensorMath<type_t>::Set(cTensor_H,cTensor_XHAT,Gamma,Beta);
*/
/*
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
	cTensor_H_Array[unit_index].SetElement(z,y,x,out);
   }
  }
 }*/

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
 CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor(0);

 cTensor_dGamma=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_dBeta=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());

/*
 //создаём все вспомогательные тензоры
 CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor();


 cTensor_PrevLayerError=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());

 cTensor_XMU_Array=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_XHAT=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_DXHAT=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());

 cTensor_DXMU1=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 cTensor_DXMU2=CTensor<type_t>(prev_output.GetSizeZ(),prev_output.GetSizeY(),prev_output.GetSizeX());
 */
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBatchNormalization<type_t>::TrainingStop(void)
{
 /*
 //удаляем все вспомогательные тензоры
 cTensor_Delta_Array.clear();

 cTensor_PrevLayerError=CTensor<type_t>(1,1,1);
 cTensor_XMU=CTensor<type_t>(1,1,1);
 cTensor_XHAT=CTensor<type_t>(1,1,1);
 cTensor_DXHAT=CTensor<type_t>(1,1,1);

 cTensor_DXMU1=CTensor<type_t>(1,1,1);
 cTensor_DXMU2=CTensor<type_t>(1,1,1);
 */
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

  #step0Variable
  dx = dx1 + dx2
*/
 type_t N=static_cast<type_t>(BatchSize);
 cTensor_DIVAR.Zero();
 for(size_t n=0;n<BatchSize;n++)
 {
  CTensor<type_t> &dout=cTensor_Delta_Array[n];

  size_t dout_x=dout.GetSizeX();
  size_t dout_y=dout.GetSizeY();
  size_t dout_z=dout.GetSizeZ();
  //dbeta = np.sum(dout, axis=0)
  CTensorMath<type_t>::Add(cTensor_dBeta,cTensor_dBeta,dout);
  //dgamma = np.sum(dout*xhat, axis=0)
  CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,dout,cTensor_XHAT_Array[n]);
  CTensorMath<type_t>::Add(cTensor_dGamma,cTensor_dGamma,cTensor_TmpA);
  //dxhat = dout * gamma
  CTensorMath<type_t>::TensorItemProduction(cTensor_DXHAT_Array[n],dout,cTensor_Gamma);
  //divar = np.sum(dxhat*xmu, axis=0)
  CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,cTensor_DXHAT_Array[n],cTensor_XMU_Array[n]);
  CTensorMath<type_t>::Add(cTensor_DIVAR,cTensor_DIVAR,cTensor_TmpA);
 }

  //dsqrtvar = -1. /(sqrtvar**2) * divar
  CTensorMath<type_t>::Pow2(cTensor_TmpA,cTensor_SQRTVAR,-1.0);
  CTensorMath<type_t>::Inv(cTensor_TmpA,cTensor_TmpA);
  CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,cTensor_TmpA,cTensor_DIVAR);
  //сейчас в TmpA находится dsqrtvar
  //dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
  //dsq = 1. /N * np.ones((N,D)) * dvar
  CTensorMath<type_t>::TensorItemProduction(cTensor_DVAR,cTensor_IVAR,cTensor_TmpA);
  CTensorMath<type_t>::Mul(cTensor_DVAR,cTensor_DVAR,0.5/N);
  //фактически, сейчас DVAR это dsq

 for(size_t n=0;n<BatchSize;n++)
 {
  //dxmu1 = dxhat * ivar
  CTensorMath<type_t>::TensorItemProduction(cTensor_DXMU1,cTensor_DXHAT_Array[n],cTensor_IVAR);
  //dxmu2 = 2 * xmu * dsq
  CTensorMath<type_t>::TensorItemProduction(cTensor_DXMU2,cTensor_XMU_Array[n],cTensor_DVAR);
  CTensorMath<type_t>::Mul(cTensor_DXMU2,cTensor_DXMU2,2.0);
  //dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
  //dx2 = 1. /N * np.ones((N,D)) * dmu
  CTensorMath<type_t>::Add(cTensor_TmpA,cTensor_DXMU1,cTensor_DXMU2,-1/N,-1/N);
  //фактически, сейчас TmpA это dx2
  //dx1 = (dxmu1 + dxmu2)
  //dx = dx1 + dx2
  CTensorMath<type_t>::Add(cTensor_TmpA,cTensor_TmpA,cTensor_DXMU1);
  CTensorMath<type_t>::Add(cTensor_TmpA,cTensor_TmpA,cTensor_DXMU2);
  PrevLayerPtr->SetOutputError(n,cTensor_TmpA);
 }

/*
 CTensor<type_t> &dout=cTensor_Delta;

 size_t dout_x=dout.GetSizeX();
 size_t dout_y=dout.GetSizeY();
 size_t dout_z=dout.GetSizeZ();

 type_t N=static_cast<type_t>(dout_x*dout_y*dout_z);
 //type_t D=static_cast<type_t>(1);

 CTensor<type_t> cTensor_Output(1,1,1);

 //считаем dbeta
 //dbeta = np.sum(dout, axis=0)
 CTensorMath<type_t>::Set(cTensor_TmpA,dout,1);
 cTensor_TmpA.ReinterpretSize(1,1,dout_z*dout_y*dout_x);
 cTensor_Output.SetElement(0,0,0,0);
 CTensorMath<type_t>::SummXY(cTensor_Output,cTensor_TmpA);
 cTensor_TmpA.RestoreSize();
 double dbeta=cTensor_Output.GetElement(0,0,0);
 //считаем dgamma
 //dgamma = np.sum(dgammax*xhat, axis=0)
 CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,dout,cTensor_XHAT);
 cTensor_TmpA.ReinterpretSize(1,1,dout_z*dout_y*dout_x);
 cTensor_Output.SetElement(0,0,0,0);
 CTensorMath<type_t>::SummXY(cTensor_Output,cTensor_TmpA);
 cTensor_TmpA.RestoreSize();
 double dgamma=cTensor_Output.GetElement(0,0,0);
 //считаем dhxat
 //dxhat = dgammax * gamma
 CTensorMath<type_t>::Set(cTensor_DXHAT,dout,Gamma);
 //считаем dxmu1
 //dxmu1 = dxhat * ivar
 CTensorMath<type_t>::Set(cTensor_DXMU1,cTensor_DXHAT,IVAR);
 //считаем divar
 //divar = np.sum(dxhat*xmu, axis=0)
 CTensorMath<type_t>::TensorItemProduction(cTensor_TmpA,cTensor_DXHAT,cTensor_XMU);
 cTensor_TmpA.ReinterpretSize(1,1,dout_z*dout_y*dout_x);
 cTensor_Output.SetElement(0,0,0,0);
 CTensorMath<type_t>::SummXY(cTensor_Output,cTensor_TmpA);
 cTensor_TmpA.RestoreSize();
 double divar=cTensor_Output.GetElement(0,0,0);
 double dsqrtvar=-1.0/(SQRTVAR*SQRTVAR)*divar;//dsqrtvar = -1. /(sqrtvar**2) * divar
 double dvar=0.5*1.0/sqrt(VAR+EPSILON)*dsqrtvar;//dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
 //считаем dxmu2 и dmu
 //dxmu2 = 2 * xmu * dvar / N
 CTensorMath<type_t>::Set(cTensor_DXMU2,cTensor_XMU,dvar*2.0/N);
 //dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
 CTensorMath<type_t>::Add(cTensor_TmpA,cTensor_DXMU1,cTensor_DXMU2);
 cTensor_TmpA.ReinterpretSize(1,1,dout_z*dout_y*dout_x);
 cTensor_Output.SetElement(0,0,0,0);
 CTensorMath<type_t>::SummXY(cTensor_Output,cTensor_TmpA);
 cTensor_TmpA.RestoreSize();
 double dmu=-1.0*cTensor_Output.GetElement(0,0,0);
 //формируем тензор ошибки слоя
 //dx1 = (dxmu1 + dxmu2)
 CTensorMath<type_t>::Add(cTensor_TmpA,cTensor_DXMU1,cTensor_DXMU2);
 //dx2 = 1. /N * dmu
 //dx = dx1 + dx2
 CTensorMath<type_t>::Set(cTensor_PrevLayerError,cTensor_TmpA,1,dmu/N);
*/
/*
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
 //формируем тензор ошибки слоя
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
*/

/*
 if (create_delta_weight==true)
 {
  dGamma+=dgamma;
  dBeta+=dbeta;
 }
 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
 */

 for(size_t n=0;n<BatchSize;n++)
 {
  //задаём ошибку предыдущего слоя
  PrevLayerPtr->SetOutputError(n,cTensor_Delta_Array[n]);
 }

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
 CTensorMath<type_t>::Sub(cTensor_Gamma,cTensor_Gamma,cTensor_dGamma,1.0,speed);
 CTensorMath<type_t>::Sub(cTensor_Beta,cTensor_Beta,cTensor_dBeta,1.0,speed);

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
