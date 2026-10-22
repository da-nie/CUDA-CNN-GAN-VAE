#ifndef C_NET_LAYER_LAYER_NORMALIZATION_GPT2_H
#define C_NET_LAYER_LAYER_NORMALIZATION_GPT2_H

//****************************************************************************************************
//\file Нормализация по слою (LayerNormalization), вариант для GPT-2
//----------------------------------------------------------------------------------------------------
//Слой рассчитан на работу в цепочке с раскладкой (Batch,1,T,D):
//  W - размер пакета, Y - позиции последовательности, X - признаки (эмбеддинг), Z==1.
//
//Нормализация выполняется ПО ОТДЕЛЬНОМУ ТОКЕНУ: для каждой позиции (w,t) статистики
//(среднее и дисперсия) вычисляются по всем D признакам, затем применяются масштабирование
//Gamma и сдвиг Beta - по одному параметру на признак. Это соответствует ln_1/ln_2/ln_f
//из GPT-2 (LayerNorm по последней оси, layer_norm_epsilon=1e-5).
//
//Ядра CTensorMath::LayerNormForwardGPT2/LayerNormBackwardGPT2 вычисляют статистики по Z
//для каждой позиции (w,y,x) и работают с раскладкой (Batch,Z,Y,X)=(B,D,T,1). Мост между
//раскладками выполняется внутри слоя без перемещения данных:
//    (B,1,T,D) --Transpose(Y<->X)--> (B,1,D,T) --ReinterpretSize--> (B,D,T,1)
//(B,1,D,T) и (B,D,T,1) занимают одну и ту же память, поэтому ReinterpretSize бесплатен.
//
//Прямой проход:  Transpose -> LayerNormForwardGPT2 -> Transpose
//Обратный проход: Transpose(дельта) -> LayerNormBackwardGPT2 (dGamma/dBeta НАКАПЛИВАЮТСЯ)
//                 -> Transpose(ошибка предыдущему слою)
//
//Размерности по X (Features) фиксируются в Create; при изменении W или T (длины
//последовательности) рабочие тензоры пересоздаются автоматически в Forward.
//****************************************************************************************************

#include <stdio.h>
#include <fstream>
#include <math.h>
#include <string>

#include "../common/idatastream.h"
#include "inetlayer.cu.h"
#include "../cuda/tensor.cu.h"
#include "neuron.cu.h"

//****************************************************************************************************
//константы
//****************************************************************************************************
//в GPT-2 используется layer_norm_epsilon=1e-5
static const double GPT2_LN_EPSILON=1e-5;

//****************************************************************************************************
//!Нормализация по слою (вариант GPT-2, раскладка сети (Batch,1,T,D))
//****************************************************************************************************
template<class type_t>
class CNetLayerLayerNormalizationGPT2:public INetLayer<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  uint32_t Layer;///<порядковый номер экземпляра слоя (диагностика)

  INetLayer<type_t> *PrevLayerPtr;///<указатель на предшествующий слой (либо NULL)
  INetLayer<type_t> *NextLayerPtr;///<указатель на последующий слой (либо NULL)

  uint32_t BatchSize;///<размер пакета для обучения

  uint32_t Features;///<D - размер эмбеддинга (размер входа по X, менять нельзя)
  uint32_t Positions;///<T - длина последовательности (размер входа по Y)

  CTensor<type_t> cTensor_Gamma;///<параметр масштабирования по признакам (1,Features,1,1)
  CTensor<type_t> cTensor_Beta;///<параметр сдвига по признакам (1,Features,1,1)

  CTensor<type_t> cTensor_Gamma_EMA;///<EMA параметра масштабирования
  CTensor<type_t> cTensor_Beta_EMA;///<EMA параметра сдвига

  //рабочие тензоры; "сетевая" раскладка (Batch,1,T,D), внутренняя LN (Batch,Features,Positions,1)
  CTensor<type_t> cTensor_H;///<выход слоя в раскладке сети (Batch,1,Positions,Features)
  CTensor<type_t> cTensor_Tmp;///<буфер транспонирования (Batch,1,Features,Positions)<=>(Batch,Features,Positions,1)
  CTensor<type_t> cTensor_H_Z;///<выход во внутреннем представлении (Batch,Features,Positions,1)
  CTensor<type_t> cTensor_XHAT;///<нормализованные значения (Batch,Features,Positions,1)
  CTensor<type_t> cTensor_InvStd;///<обратное СКО по позициям (Batch,1,Positions,1)

  CTensor<type_t> cTensor_Delta;///<входящая ошибка в раскладке сети (Batch,1,Positions,Features)
  CTensor<type_t> cTensor_PrevLayerError;///<ошибка предыдущему слою (Batch,1,Positions,Features)
  CTensor<type_t> cTensor_PrevLayerError_Z;///<ошибка во внутреннем представлении (Batch,Features,Positions,1)

  CTensor<type_t> cTensor_dGamma;///<градиент по гамме (1,Features,1,1)
  CTensor<type_t> cTensor_dBeta;///<градиент по бете (1,Features,1,1)

  bool TrainingEnabled;///<включено ли обучение

  //для оптимизации Adam
  CTensor<type_t> cTensor_MK;///<тензор фильтра 1
  CTensor<type_t> cTensor_VK;///<тензор фильтра 2
  CTensor<type_t> cTensor_MB;///<коэффициент фильтра 1 сдвигов
  CTensor<type_t> cTensor_VB;///<коэффициент фильтра 2 сдвигов

  using INetLayer<type_t>::Beta1;///<параметры алгоритма Adam
  using INetLayer<type_t>::Beta2;
  using INetLayer<type_t>::Epsilon;
  //режим усреднения
  using INetLayer<type_t>::EMAEnabled;
  using INetLayer<type_t>::UseEMA;
  using INetLayer<type_t>::EMA_K;
  //ограничение нормы
  using INetLayer<type_t>::ClipByNormThresHold;///<ограничение нормы
  using INetLayer<type_t>::InferenceMode;///режим вывода
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerLayerNormalizationGPT2(INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerLayerNormalizationGPT2();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
  void Reset(type_t scale=1);///<выполнить инициализацию весов и сдвигов
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

  void EnableEMA(bool state);///<разрешить/запретить использование усреднённых весов
  bool LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size=false);///<загрузить усреднённые веса
  bool SaveEMAWeight(IDataStream *iDataStream_Ptr);///<сохранить усреднённые веса

 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  void ReallocateTensors(uint32_t batch_size,uint32_t positions);///<пересоздать рабочие тензоры при изменении формы входа
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerLayerNormalizationGPT2<type_t>::CNetLayerLayerNormalizationGPT2(INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 Create(prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerLayerNormalizationGPT2<type_t>::~CNetLayerLayerNormalizationGPT2()
{
}

//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
/*!пересоздать рабочие тензоры при изменении формы входа (размер пакета или длины последовательности)
\param[in] batch_size Новый размер пакета
\param[in] positions Новая длина последовательности
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::ReallocateTensors(uint32_t batch_size,uint32_t positions)
{
 if (cTensor_H.GetSizeW()==batch_size && cTensor_H.GetSizeZ()==1 &&
     cTensor_H.GetSizeY()==positions && cTensor_H.GetSizeX()==Features) return;//форма не изменилась

 BatchSize=batch_size;
 Positions=positions;

 cTensor_H=CTensor<type_t>(BatchSize,1,Positions,Features);
 cTensor_Tmp=CTensor<type_t>(BatchSize,1,Features,Positions);
 cTensor_H_Z=CTensor<type_t>(BatchSize,Features,Positions,1);
 cTensor_XHAT=CTensor<type_t>(BatchSize,Features,Positions,1);
 cTensor_InvStd=CTensor<type_t>(BatchSize,1,Positions,1);
 cTensor_Delta=CTensor<type_t>(BatchSize,1,Positions,Features);
 cTensor_PrevLayerError=CTensor<type_t>(BatchSize,1,Positions,Features);

 if (TrainingEnabled==true) cTensor_PrevLayerError_Z=CTensor<type_t>(BatchSize,Features,Positions,1);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
/*!создать слой
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-запрещено)
\param[in] batch_size Количество элементов минипакета
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::Create(INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 static uint32_t i=0;
 i++;
 Layer=i;

 if (prev_layer_ptr==NULL) throw("Слой LayerNormalizationGPT2 не может быть входным!");

 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;
 BatchSize=batch_size;

 CTensor<type_t> &prev_output=PrevLayerPtr->GetOutputTensor();
 if (prev_output.GetSizeZ()!=1)
  throw("CNetLayerLayerNormalizationGPT2::Create - ожидается раскладка (Batch,1,Positions,Features): Z должен быть равен 1!");

 Features=prev_output.GetSizeX();//D - размер эмбеддинга (менять нельзя)
 Positions=prev_output.GetSizeY();//T - длина последовательности

 //выход слоя - в раскладке сети
 cTensor_H=CTensor<type_t>(BatchSize,1,Positions,Features);
 //буфер транспонирования: (B,1,D,T) <=> (B,D,T,1) - одна и та же память
 cTensor_Tmp=CTensor<type_t>(BatchSize,1,Features,Positions);
 //внутренние тензоры LN: статистики по Z=Features для каждой позиции
 cTensor_H_Z=CTensor<type_t>(BatchSize,Features,Positions,1);
 cTensor_XHAT=CTensor<type_t>(BatchSize,Features,Positions,1);
 cTensor_InvStd=CTensor<type_t>(BatchSize,1,Positions,1);

 //параметры - по одному на признак: как ln.weight/ln.bias в GPT-2
 cTensor_Gamma=CTensor<type_t>(1,Features,1,1);
 cTensor_Beta=CTensor<type_t>(1,Features,1,1);
 cTensor_Gamma_EMA=CTensor<type_t>(1,Features,1,1);
 cTensor_Beta_EMA=CTensor<type_t>(1,Features,1,1);

 //ошибки в раскладке сети
 cTensor_Delta=CTensor<type_t>(BatchSize,1,Positions,Features);
 cTensor_PrevLayerError=CTensor<type_t>(BatchSize,1,Positions,Features);

 prev_layer_ptr->SetNextLayerPtr(this);
 TrainingEnabled=false;
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию весов и сдвигов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::Reset(type_t scale)
{
 CTensorMath<type_t>::Fill(cTensor_Gamma,1);
 CTensorMath<type_t>::Fill(cTensor_Beta,0);
 CTensorMath<type_t>::Fill(cTensor_Gamma_EMA,1);
 CTensorMath<type_t>::Fill(cTensor_Beta_EMA,0);
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::SetOutput(CTensor<type_t> &output)
{
 cTensor_H.RestoreSize();
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerLayerNormalizationGPT2<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerLayerNormalizationGPT2<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerLayerNormalizationGPT2<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerLayerNormalizationGPT2<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!получить выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::GetOutput(CTensor<type_t> &output)
{
 cTensor_H.RestoreSize();
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerLayerNormalizationGPT2<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerLayerNormalizationGPT2<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerLayerNormalizationGPT2<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerLayerNormalizationGPT2<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности матрицы output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
//!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::Forward(void)
{
 cTensor_H.RestoreSize();
 CTensor<type_t> &input=PrevLayerPtr->GetOutputTensor();

 if (input.GetSizeZ()!=1) throw("CNetLayerLayerNormalizationGPT2::Forward - ожидается раскладка (Batch,1,Positions,Features)!");
 if (input.GetSizeX()!=Features) throw("CNetLayerLayerNormalizationGPT2::Forward - размер эмбеддинга входа не совпадает с созданным в Create!");

 ReallocateTensors(input.GetSizeW(),input.GetSizeY());

 CTensor<type_t> &gamma=(UseEMA)?cTensor_Gamma_EMA:cTensor_Gamma;
 CTensor<type_t> &beta=(UseEMA)?cTensor_Beta_EMA:cTensor_Beta;

 //1. транспонирование (B,1,T,D) -> (B,1,D,T)
 cTensor_Tmp.ReinterpretSize(BatchSize,1,Features,Positions);//гарантировать вид (B,1,D,T) для Transpose
 CTensorMath<type_t>::Transpose(cTensor_Tmp,input);
 //2. та же память в представлении (B,D,T,1): признаки в Z, позиции в Y
 cTensor_Tmp.ReinterpretSize(BatchSize,Features,Positions,1);

 //3. LayerNorm: статистики по Z (=признаки) для каждой позиции (w,t)
 cTensor_H_Z.ReinterpretSize(BatchSize,Features,Positions,1);//гарантировать вид (B,D,T,1) для ядра
 CTensorMath<type_t>::LayerNormForwardGPT2(cTensor_H_Z,cTensor_Tmp,gamma,beta,cTensor_XHAT,cTensor_InvStd,static_cast<type_t>(GPT2_LN_EPSILON));

 //4. обратно в раскладку сети
 cTensor_H_Z.ReinterpretSize(BatchSize,1,Features,Positions);//вид (B,1,D,T) для Transpose
 CTensorMath<type_t>::Transpose(cTensor_H,cTensor_H_Z);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на выходной тензор слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerLayerNormalizationGPT2<type_t>::GetOutputTensor(void)
{
 cTensor_H.RestoreSize();
 return(cTensor_H);
}
//----------------------------------------------------------------------------------------------------
/*!задать указатель на последующий слой
\param[in] next_layer_ptr Указатель на последующий слой
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
{
 NextLayerPtr=next_layer_ptr;
}

//----------------------------------------------------------------------------------------------------
/*!сохранить параметры слоя (порядок: Beta, Gamma)
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerLayerNormalizationGPT2<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 cTensor_Beta.Save(iDataStream_Ptr);
 cTensor_Gamma.Save(iDataStream_Ptr);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя (порядок: Beta, Gamma)
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerLayerNormalizationGPT2<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 cTensor_Beta.Load(iDataStream_Ptr,check_size);
 cTensor_Gamma.Load(iDataStream_Ptr,check_size);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerLayerNormalizationGPT2<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
{
 cTensor_MK.Save(iDataStream_Ptr);
 cTensor_VK.Save(iDataStream_Ptr);
 cTensor_MB.Save(iDataStream_Ptr);
 cTensor_VB.Save(iDataStream_Ptr);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerLayerNormalizationGPT2<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 cTensor_MK.Load(iDataStream_Ptr);
 cTensor_VK.Load(iDataStream_Ptr);
 cTensor_MB.Load(iDataStream_Ptr);
 cTensor_VB.Load(iDataStream_Ptr);
 return(true);
}

//----------------------------------------------------------------------------------------------------
//!начать процесс обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::TrainingStart(void)
{
 cTensor_H.RestoreSize();
 cTensor_dGamma=CTensor<type_t>(1,Features,1,1);
 cTensor_dBeta=CTensor<type_t>(1,Features,1,1);

 cTensor_MK=cTensor_dGamma;
 cTensor_VK=cTensor_dGamma;
 cTensor_MB=cTensor_dBeta;
 cTensor_VB=cTensor_dBeta;

 CTensorMath<type_t>::Fill(cTensor_MK,0);
 CTensorMath<type_t>::Fill(cTensor_VK,0);
 CTensorMath<type_t>::Fill(cTensor_MB,0);
 CTensorMath<type_t>::Fill(cTensor_VB,0);

 //буфер ошибки во внутреннем представлении нужен только при обучении
 cTensor_PrevLayerError_Z=CTensor<type_t>(BatchSize,Features,Positions,1);

 TrainingEnabled=true;
}

//----------------------------------------------------------------------------------------------------
//!завершить процесс обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_dGamma=CTensor<type_t>(1,1,1,1);
 cTensor_dBeta=CTensor<type_t>(1,1,1,1);

 cTensor_MK=cTensor_dGamma;
 cTensor_VK=cTensor_dGamma;
 cTensor_MB=cTensor_dBeta;
 cTensor_VB=cTensor_dBeta;

 cTensor_PrevLayerError_Z=CTensor<type_t>(1,1,1,1);

 TrainingEnabled=false;
}
//----------------------------------------------------------------------------------------------------
//!выполнить обратный проход по сети для обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::TrainingBackward(bool create_delta_weight)
{
 if (create_delta_weight)
 {
  CTensorMath<type_t>::Fill(cTensor_dGamma,0);
  CTensorMath<type_t>::Fill(cTensor_dBeta,0);
 }

 //та же гамма, что и в прямом проходе
 CTensor<type_t> &gamma=(UseEMA)?cTensor_Gamma_EMA:cTensor_Gamma;

 //1. ошибка (B,1,T,D) -> (B,1,D,T) -> (B,D,T,1)
 cTensor_Tmp.ReinterpretSize(BatchSize,1,Features,Positions);//вид (B,1,D,T) для Transpose
 CTensorMath<type_t>::Transpose(cTensor_Tmp,cTensor_Delta);
 cTensor_Tmp.ReinterpretSize(BatchSize,Features,Positions,1);//вид (B,D,T,1) для ядра

 //2. LayerNormBackwardGPT2 НАКАПЛИВАЕТ dGamma/dBeta (сумма по позициям и пакету - как и требуется)
 cTensor_PrevLayerError_Z.ReinterpretSize(BatchSize,Features,Positions,1);//вид (B,D,T,1) для ядра
 CTensorMath<type_t>::LayerNormBackwardGPT2(cTensor_Tmp,cTensor_XHAT,gamma,cTensor_InvStd,
                                            cTensor_PrevLayerError_Z,cTensor_dGamma,cTensor_dBeta);

 //3. ошибка предыдущему слою: (B,D,T,1) -> (B,1,D,T) -> (B,1,T,D)
 cTensor_PrevLayerError_Z.ReinterpretSize(BatchSize,1,Features,Positions);//вид (B,1,D,T) для Transpose
 CTensorMath<type_t>::Transpose(cTensor_PrevLayerError,cTensor_PrevLayerError_Z);

 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
}

//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::TrainingResetDeltaWeight(void)
{
 CTensorMath<type_t>::Fill(cTensor_dGamma,0);
 CTensorMath<type_t>::Fill(cTensor_dBeta,0);
}

//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
 CTensorMath<type_t>::ClipByNormXY(cTensor_dGamma,cTensor_dGamma,ClipByNormThresHold);
 CTensorMath<type_t>::ClipByNormXY(cTensor_dBeta,cTensor_dBeta,ClipByNormThresHold);

 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_ADAM)
 {
  //применяем алгоритм Adam
  CTensorMath<type_t>::Adam(cTensor_Gamma,cTensor_dGamma,cTensor_MK,cTensor_VK,BatchSize*batch_scale,speed,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_Beta,cTensor_dBeta,cTensor_MB,cTensor_VB,BatchSize*batch_scale,speed,Beta1,Beta2,Epsilon,iteration);
 }
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_GRADIENT)
 {
  double s=speed/static_cast<double>(BatchSize);
  CTensorMath<type_t>::Sub(cTensor_Gamma,cTensor_Gamma,cTensor_dGamma,1,s/batch_scale);
  CTensorMath<type_t>::Sub(cTensor_Beta,cTensor_Beta,cTensor_dBeta,1,s/batch_scale);
 }

 //ограничиваем Gamma и Beta в диапазоне [-100,100] для защиты от взрыва
 CTensorMath<type_t>::Clip(cTensor_Gamma,-100.0,100.0);
 CTensorMath<type_t>::Clip(cTensor_Beta,-100.0,100.0);

 if (EMAEnabled)
 {
  CTensorMath<type_t>::Add(cTensor_Gamma_EMA,cTensor_Gamma_EMA,cTensor_Gamma,EMA_K,1.0-EMA_K);
  CTensorMath<type_t>::Add(cTensor_Beta_EMA,cTensor_Beta_EMA,cTensor_Beta,EMA_K,1.0-EMA_K);
 }
}

//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя (в раскладке сети)
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerLayerNormalizationGPT2<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки в раскладке сети (Batch,1,Positions,Features)
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::SetOutputError(CTensor<type_t>& error)
{
 if (error.GetSizeW()!=BatchSize || error.GetSizeZ()!=1 ||
     error.GetSizeY()!=Positions || error.GetSizeX()!=Features)
  throw("CNetLayerLayerNormalizationGPT2::SetOutputError - неверная размерность тензора ошибки!");
 cTensor_Delta=error;
}

//----------------------------------------------------------------------------------------------------
/*!ограничить веса в диапазон (в LayerNorm ограничивать нечего)
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::ClipWeight(type_t min,type_t max)
{
 (void)min;
 (void)max;
}
//----------------------------------------------------------------------------------------------------
/*!задать временной шаг (LayerNorm от временных шагов не зависит)
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
 (void)index;
 (void)time_step;
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" LayerNormalizationGPT2: input",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::PrintOutputTensorSize(const std::string &name)
{
 cTensor_H.RestoreSize();
 GetOutputTensor().Print(name+" LayerNormalizationGPT2: output",false);
}

//----------------------------------------------------------------------------------------------------
/*!разрешить/запретить использование усреднённых весов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerLayerNormalizationGPT2<type_t>::EnableEMA(bool state)
{
 EMAEnabled=state;
 if (state)
 {
  cTensor_Gamma_EMA=cTensor_Gamma;
  cTensor_Beta_EMA=cTensor_Beta;
 }
}

//----------------------------------------------------------------------------------------------------
/*!загрузить усреднённые веса (порядок: Beta, Gamma)
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerLayerNormalizationGPT2<type_t>::LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size)
{
 cTensor_Beta_EMA.Load(iDataStream_Ptr,check_size);
 cTensor_Gamma_EMA.Load(iDataStream_Ptr,check_size);
 return(true);
}

//----------------------------------------------------------------------------------------------------
/*!сохранить усреднённые веса (порядок: Beta, Gamma)
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerLayerNormalizationGPT2<type_t>::SaveEMAWeight(IDataStream *iDataStream_Ptr)
{
 cTensor_Beta_EMA.Save(iDataStream_Ptr);
 cTensor_Gamma_EMA.Save(iDataStream_Ptr);
 return(true);
}

#endif
