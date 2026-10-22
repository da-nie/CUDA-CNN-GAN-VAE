#ifndef C_NET_LAYER_SELF_ATTENTION_H
#define C_NET_LAYER_SELF_ATTENTION_H

//****************************************************************************************************
//\file Слой самовнимания
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdio.h>
#include <fstream>
#include <vector>
#include <math.h>
#include <memory>

#include "../common/idatastream.h"
#include "inetlayer.cu.h"
#include "../cuda/tensor.cu.h"
#include "neuron.cu.h"
#include "../common/crandom.h"

#include "cnetlayerlinear.cu.h"
#include "cnetlayersplitter.cu.h"

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
//! Слой самовнимания
//****************************************************************************************************
template<class type_t>
class CNetLayerSelfAttention:public INetLayer<type_t>
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

  uint32_t InputSize_X;///<размер входного тензора по X
  uint32_t InputSize_Y;///<размер входного тензора по Y
  uint32_t InputSize_Z;///<размер входного тензора по Z

  uint32_t OutputSize_X;///<размер выходного тензора по X
  uint32_t OutputSize_Y;///<размер выходного тензора по Y
  uint32_t OutputSize_Z;///<размер выходного тензора по Z

  uint32_t TSize;///<размерность входа слоя внимания

  //структура слоя
  CTensor<type_t> cTensor_Wq;
  CTensor<type_t> cTensor_Wk;
  CTensor<type_t> cTensor_Wv;

  CTensor<type_t> cTensor_Wq_EMA;
  CTensor<type_t> cTensor_Wk_EMA;
  CTensor<type_t> cTensor_Wv_EMA;

  CTensor<type_t> cTensor_Bq;
  CTensor<type_t> cTensor_Bk;
  CTensor<type_t> cTensor_Bv;

  CTensor<type_t> cTensor_Bq_EMA;
  CTensor<type_t> cTensor_Bk_EMA;
  CTensor<type_t> cTensor_Bv_EMA;

  CTensor<type_t> cTensor_Bq_Batch;///<рабочая копия смещения, размноженная по батчу [Batch,1,1,D]
  CTensor<type_t> cTensor_Bk_Batch;
  CTensor<type_t> cTensor_Bv_Batch;

  CTensor<type_t> cTensor_Qh;
  CTensor<type_t> cTensor_Kh;
  CTensor<type_t> cTensor_Vh;

  CTensor<type_t> cTensor_Sh;
  CTensor<type_t> cTensor_Oh;

  CTensor<type_t> cTensor_H;///<выходной тензор

  std::shared_ptr<INetLayer<type_t> > iNetLayer_SoftMaxInput;
  std::shared_ptr<INetLayer<type_t> > iNetLayer_SoftMax;

  uint32_t NumberOfHead;///<количество голов внимания слоя
  uint32_t HeadDimension;///<размер одной головы внимания

  bool ApplyCausalMask;///<можно ли использовать каузальную маску

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<градиент выхода слоя dH [Batch,1,T,D]
  CTensor<type_t> cTensor_PrevLayerError;///<градиент входа слоя dX [Batch,1,T,D]

  CTensor<type_t> cTensor_dOh;///<градиент результата внимания по головам [Batch,h,T,Dk]
  CTensor<type_t> cTensor_dP;///<градиент матрицы внимания (вход softmax) [Batch,h,T,T]
  CTensor<type_t> cTensor_dSh;///<градиент Sh после масштабирования [Batch,h,T,T]

  CTensor<type_t> cTensor_dQh;///<градиент Qh [Batch,h,T,Dk]
  CTensor<type_t> cTensor_dKh;///<градиент Kh [Batch,h,T,Dk]
  CTensor<type_t> cTensor_dVh;///<градиент Vh [Batch,h,T,Dk]

  CTensor<type_t> cTensor_dQ;///<градиент Q [Batch,1,T,D]
  CTensor<type_t> cTensor_dK;///<градиент K [Batch,1,T,D]
  CTensor<type_t> cTensor_dV;///<градиент V [Batch,1,T,D]

  CTensor<type_t> cTensor_dWq;///<градиент Wq [1,1,D,D]
  CTensor<type_t> cTensor_dWk;///<градиент Wk [1,1,D,D]
  CTensor<type_t> cTensor_dWv;///<градиент Wv [1,1,D,D]

  CTensor<type_t> cTensor_dWq_Batch;///<градиент Wq по каждому элементу батча [Batch,1,D,D] (для редукции)
  CTensor<type_t> cTensor_dWk_Batch;///<градиент Wk по каждому элементу батча [Batch,1,D,D]
  CTensor<type_t> cTensor_dWv_Batch;///<градиент Wv по каждому элементу батча [Batch,1,D,D]

  CTensor<type_t> cTensor_dBq_Batch;///<частичные градиенты смещения: сумма по токенам для каждого w [Batch,1,1,D]
  CTensor<type_t> cTensor_dBk_Batch;
  CTensor<type_t> cTensor_dBv_Batch;

  CTensor<type_t> cTensor_dBq;///<суммарный градиент смещения [1,1,1,D]
  CTensor<type_t> cTensor_dBk;
  CTensor<type_t> cTensor_dBv;

  CTensor<type_t> cTensor_M_Bq;///<моменты Adam для смещений [1,1,1,D]
  CTensor<type_t> cTensor_V_Bq;
  CTensor<type_t> cTensor_M_Bk;
  CTensor<type_t> cTensor_V_Bk;
  CTensor<type_t> cTensor_M_Bv;
  CTensor<type_t> cTensor_V_Bv;

  CTensor<type_t> cTensor_Ones;///<константный вектор единиц [Batch,1,T,1] для редукции суммы по токенам

  CTensor<type_t> cTensor_dX;///<временный тензор: dQ@Wq.T [Batch,1,T,D]
  CTensor<type_t> cTensor_dX2;///<временный тензор: dK@Wk.T / dV@Wv.T [Batch,1,T,D]

  //моменты Adam для весов
  CTensor<type_t> cTensor_M_Wq;
  CTensor<type_t> cTensor_V_Wq;
  CTensor<type_t> cTensor_M_Wk;
  CTensor<type_t> cTensor_V_Wk;
  CTensor<type_t> cTensor_M_Wv;
  CTensor<type_t> cTensor_V_Wv;

  using INetLayer<type_t>::Beta1;///<параметры алгоритма Adam
  using INetLayer<type_t>::Beta2;
  using INetLayer<type_t>::Epsilon;
  //режим усреднения
  using INetLayer<type_t>::EMAEnabled;
  using INetLayer<type_t>::UseEMA;
  using INetLayer<type_t>::EMA_K;
  //ограничение нормы
  using INetLayer<type_t>::ClipByNormThresHold;///<ограничение нормы
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerSelfAttention(uint32_t head,bool apply_causal_mask,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerSelfAttention(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerSelfAttention();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(uint32_t head,bool apply_causal_mask,INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
  void Reset(type_t scale=1);///<выполнить инициализацию слоя
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
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerSelfAttention<type_t>::CNetLayerSelfAttention(uint32_t head,bool apply_causal_mask,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 Create(head,apply_causal_mask,prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerSelfAttention<type_t>::CNetLayerSelfAttention(void)
{
 Create();
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerSelfAttention<type_t>::~CNetLayerSelfAttention()
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
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя
\param[in] batch_size Количество элементов минипакета
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::Create(uint32_t head,bool apply_causal_mask,INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;
 BatchSize=batch_size;
 ApplyCausalMask=apply_causal_mask;

 NumberOfHead=head;
 if (PrevLayerPtr==NULL) throw("Слой самовнимания не может быть входным!");//слой без предшествующего считается входным
 //запомним размеры входного тензора, чтобы потом всегда к ним приводить
 InputSize_X=PrevLayerPtr->GetOutputTensor().GetSizeX();
 InputSize_Y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 InputSize_Z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 if (InputSize_Z!=1) throw("Входной тензор слоя самовнимания должен быть единичного размера по Z!");

 if (InputSize_X%head!=0) throw("Размерность по X слоя самовнимания не кратна количеству голов внимания!");

 HeadDimension=InputSize_X/head;//размер одной головы внимания

 NextLayerPtr=NULL;

 //размер выходного тензора
 OutputSize_X=InputSize_X;
 OutputSize_Y=InputSize_Y;
 OutputSize_Z=1;

 TSize=InputSize_Y;

 //создаём слой
 cTensor_Wq=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_Wk=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_Wv=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);

 cTensor_Bq=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_Bk=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_Bv=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);

 cTensor_Qh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,HeadDimension);
 cTensor_Kh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,HeadDimension);
 cTensor_Vh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,HeadDimension);

 cTensor_Sh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,TSize);
 cTensor_Oh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,HeadDimension);

 iNetLayer_SoftMaxInput=std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolutionInput<type_t>(NumberOfHead,TSize,TSize,BatchSize));
 iNetLayer_SoftMax=std::shared_ptr<INetLayer<type_t>>(new CNetLayerSoftMax<type_t>(iNetLayer_SoftMaxInput.get(),BatchSize));

 //создаём выходные тензоры
 cTensor_H=CTensor<type_t>(BatchSize,1,TSize,NumberOfHead*HeadDimension);
 //задаём предшествующему слою, что мы его последующий слой
 PrevLayerPtr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::Reset(type_t scale)
{
 type_t size=static_cast<type_t>(cTensor_Wq.GetSizeX()*cTensor_Wq.GetSizeY());
 type_t koeff=static_cast<type_t>(sqrt(6.0/size));

 //веса матриц внимания
 for(uint32_t y=0;y<cTensor_Wq.GetSizeY();y++)
 {
  for(uint32_t x=0;x<cTensor_Wq.GetSizeX();x++)
  {
   //используем метод инициализации He (Ге)
   type_t rnd=static_cast<type_t>(CRandom<type_t>::GetRandValue(2.0)-1.0);
   type_t init=rnd*koeff;
   init*=scale;
   cTensor_Wq.SetElement(0,0,y,x,init);

   rnd=static_cast<type_t>(CRandom<type_t>::GetRandValue(2.0)-1.0);
   init=rnd*koeff;
   init*=scale;

   cTensor_Wk.SetElement(0,0,y,x,init);

   rnd=static_cast<type_t>(CRandom<type_t>::GetRandValue(2.0)-1.0);
   init=rnd*koeff;
   init*=scale;
   cTensor_Wv.SetElement(0,0,y,x,init);
  }
 }
 CTensorMath<type_t>::Fill(cTensor_Bq,0);
 CTensorMath<type_t>::Fill(cTensor_Bk,0);
 CTensorMath<type_t>::Fill(cTensor_Bv,0);

 iNetLayer_SoftMaxInput->Reset(scale);
 iNetLayer_SoftMax->Reset(scale);
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::SetOutput(CTensor<type_t> &output)
{
 cTensor_H.RestoreSize();
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerSelfAttention<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerSelfAttention<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerSelfAttention<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerSelfAttention<type_t>::SetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::GetOutput(CTensor<type_t> &output)
{
 cTensor_H.RestoreSize();
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerSelfAttention<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerSelfAttention<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerSelfAttention<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerSelfAttention<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::Forward(void)
{
 cTensor_H.RestoreSize();
 //CCUDATimeSpent cCUDATimeSpent;
 //считаем внимание

 //cCUDATimeSpent.Start();

 //8 мс
 if (UseEMA==true)
 {
  CTensorMath<type_t>::MulAttentionTranspose(cTensor_Qh,PrevLayerPtr->GetOutputTensor(),cTensor_Wq_EMA,cTensor_Bq_EMA);
  CTensorMath<type_t>::MulAttentionTranspose(cTensor_Kh,PrevLayerPtr->GetOutputTensor(),cTensor_Wk_EMA,cTensor_Bk_EMA);
  CTensorMath<type_t>::MulAttentionTranspose(cTensor_Vh,PrevLayerPtr->GetOutputTensor(),cTensor_Wv_EMA,cTensor_Bv_EMA);
 }
 else
 {
  CTensorMath<type_t>::MulAttentionTranspose(cTensor_Qh,PrevLayerPtr->GetOutputTensor(),cTensor_Wq,cTensor_Bq);
  CTensorMath<type_t>::MulAttentionTranspose(cTensor_Kh,PrevLayerPtr->GetOutputTensor(),cTensor_Wk,cTensor_Bk);
  CTensorMath<type_t>::MulAttentionTranspose(cTensor_Vh,PrevLayerPtr->GetOutputTensor(),cTensor_Wv,cTensor_Bv);
 }

 //каждую голову нужно обрабатывать отдельно
 //заполняем матрицы с перестановкой осей Z и Y и изменением размерности

 /*
 //16 мс
 for(uint32_t w=0;w<BatchSize;w++)
 {
  for(uint32_t z=0;z<NumberOfHead;z++)//голова
  {
   for(uint32_t y=0;y<TSize;y++)//токен
   {
    for(uint32_t x=0;x<HeadDimension;x++)//признак
    {
     type_t value_q=cTensor_Q.GetElement(w,0,y,z*HeadDimension+x);
     type_t value_k=cTensor_K.GetElement(w,0,y,z*HeadDimension+x);
     type_t value_v=cTensor_V.GetElement(w,0,y,z*HeadDimension+x);

     cTensor_Qh.SetElement(w,z,y,x,value_q);//(w, голова, токен, признак)
     cTensor_Kh.SetElement(w,z,y,x,value_k);
     cTensor_Vh.SetElement(w,z,y,x,value_v);
    }
   }
  }
 }
 */
 //5 мс
 CTensorMath<type_t>::RightTransposeMul(cTensor_Sh,cTensor_Qh,cTensor_Kh);
 CTensorMath<type_t>::Mul(iNetLayer_SoftMaxInput->GetOutputTensor(),cTensor_Sh,1.0/sqrt(HeadDimension));

 //здесь можно задать каузальную маску, если она нужна
 if (ApplyCausalMask==true) CTensorMath<type_t>::SetCausalMask(iNetLayer_SoftMaxInput->GetOutputTensor());
/*
 //37 мс
 CTensor<type_t>& s=iNetLayer_SoftMaxInput->GetOutputTensor();
 for(uint32_t w=0;w<s.GetSizeW();w++)
  for(uint32_t z=0;z<s.GetSizeZ();z++)
   for(uint32_t y=0;y<s.GetSizeY();y++)
    for(uint32_t x=y+1;x<s.GetSizeX();x++)
     s.SetElement(w,z,y,x,static_cast<type_t>(-1e9));
*/

 iNetLayer_SoftMaxInput->Forward();
 iNetLayer_SoftMax->Forward();
 /*
 cCUDATimeSpent.Start();
 float gpu_time=cCUDATimeSpent.Stop();
 printf("На слой ушло:%.2f мс.\r\n",gpu_time);
 */

 CTensorMath<type_t>::Mul(cTensor_Oh,iNetLayer_SoftMax->GetOutputTensor(),cTensor_Vh);
 //переворачиваем обратно

 CTensorMath<type_t>::AttentionUnion(cTensor_H,cTensor_Oh);
/*
 for(uint32_t w=0;w<cTensor_Oh.GetSizeW();w++)
 {
  for(uint32_t z=0;z<cTensor_Oh.GetSizeZ();z++)
  {
   for(uint32_t y=0;y<cTensor_Oh.GetSizeY();y++)
   {
    for(uint32_t x=0;x<cTensor_Oh.GetSizeX();x++)
    {
     cTensor_H.SetElement(w,0,y,z*HeadDimension+x,cTensor_Oh.GetElement(w,z,y,x));
    }
   }
  }
 }
 */
 //float gpu_time=cCUDATimeSpent.Stop();
 //printf("На слой ушло:%.2f мс.\r\n",gpu_time);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerSelfAttention<type_t>::GetOutputTensor(void)
{
 cTensor_H.RestoreSize();
 return(cTensor_H);
}
//----------------------------------------------------------------------------------------------------
/*!задать указатель на последующий слой
\param[in] next_layer_ptr Указатель на последующий слой
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerSelfAttention<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 cTensor_Wq.Save(iDataStream_Ptr);
 cTensor_Wk.Save(iDataStream_Ptr);
 cTensor_Wv.Save(iDataStream_Ptr);
 cTensor_Bq.Save(iDataStream_Ptr);
 cTensor_Bk.Save(iDataStream_Ptr);
 cTensor_Bv.Save(iDataStream_Ptr);

 iNetLayer_SoftMaxInput->Save(iDataStream_Ptr);
 iNetLayer_SoftMax->Save(iDataStream_Ptr);

 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerSelfAttention<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 cTensor_Wq.Load(iDataStream_Ptr,check_size);
 cTensor_Wk.Load(iDataStream_Ptr,check_size);
 cTensor_Wv.Load(iDataStream_Ptr,check_size);
 cTensor_Bq.Load(iDataStream_Ptr,check_size);
 cTensor_Bk.Load(iDataStream_Ptr,check_size);
 cTensor_Bv.Load(iDataStream_Ptr,check_size);

 iNetLayer_SoftMaxInput->Load(iDataStream_Ptr,check_size);
 iNetLayer_SoftMax->Load(iDataStream_Ptr,check_size);

 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerSelfAttention<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
{
 cTensor_M_Wq.Save(iDataStream_Ptr);
 cTensor_V_Wq.Save(iDataStream_Ptr);
 cTensor_M_Wk.Save(iDataStream_Ptr);
 cTensor_V_Wk.Save(iDataStream_Ptr);
 cTensor_M_Wv.Save(iDataStream_Ptr);
 cTensor_V_Wv.Save(iDataStream_Ptr);

 cTensor_M_Bq.Save(iDataStream_Ptr);
 cTensor_V_Bq.Save(iDataStream_Ptr);
 cTensor_M_Bk.Save(iDataStream_Ptr);
 cTensor_V_Bk.Save(iDataStream_Ptr);
 cTensor_M_Bv.Save(iDataStream_Ptr);
 cTensor_V_Bv.Save(iDataStream_Ptr);

 iNetLayer_SoftMaxInput->SaveTrainingParam(iDataStream_Ptr);
 iNetLayer_SoftMax->SaveTrainingParam(iDataStream_Ptr);

 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerSelfAttention<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 cTensor_M_Wq.Load(iDataStream_Ptr);
 cTensor_V_Wq.Load(iDataStream_Ptr);
 cTensor_M_Wk.Load(iDataStream_Ptr);
 cTensor_V_Wk.Load(iDataStream_Ptr);
 cTensor_M_Wv.Load(iDataStream_Ptr);
 cTensor_V_Wv.Load(iDataStream_Ptr);

 cTensor_M_Bq.Load(iDataStream_Ptr);
 cTensor_V_Bq.Load(iDataStream_Ptr);
 cTensor_M_Bk.Load(iDataStream_Ptr);
 cTensor_V_Bk.Load(iDataStream_Ptr);
 cTensor_M_Bv.Load(iDataStream_Ptr);
 cTensor_V_Bv.Load(iDataStream_Ptr);

 iNetLayer_SoftMaxInput->LoadTrainingParam(iDataStream_Ptr);
 iNetLayer_SoftMax->LoadTrainingParam(iDataStream_Ptr);

 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::TrainingStart(void)
{
 cTensor_H.RestoreSize();
 //создаём все вспомогательные тензоры
 cTensor_Delta=cTensor_H;
 cTensor_PrevLayerError=cTensor_H;

 uint32_t D=NumberOfHead*HeadDimension;

 cTensor_dWq=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_dWk=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_dWv=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_dWq_Batch=CTensor<type_t>(BatchSize,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_dWk_Batch=CTensor<type_t>(BatchSize,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_dWv_Batch=CTensor<type_t>(BatchSize,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);

 cTensor_dQ=CTensor<type_t>(BatchSize,1,TSize,NumberOfHead*HeadDimension);
 cTensor_dK=CTensor<type_t>(BatchSize,1,TSize,NumberOfHead*HeadDimension);
 cTensor_dV=CTensor<type_t>(BatchSize,1,TSize,NumberOfHead*HeadDimension);
 cTensor_dQh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,HeadDimension);
 cTensor_dKh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,HeadDimension);
 cTensor_dVh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,HeadDimension);

 cTensor_dOh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,HeadDimension);
 cTensor_dP=CTensor<type_t>(BatchSize,NumberOfHead,TSize,TSize);
 cTensor_dSh=CTensor<type_t>(BatchSize,NumberOfHead,TSize,TSize);

 cTensor_dX=CTensor<type_t>(BatchSize,1,TSize,NumberOfHead*HeadDimension);
 cTensor_dX2=CTensor<type_t>(BatchSize,1,TSize,NumberOfHead*HeadDimension);

 cTensor_M_Wq=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_M_Wk=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_M_Wv=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);

 cTensor_V_Wq=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_V_Wk=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);
 cTensor_V_Wv=CTensor<type_t>(1,1,NumberOfHead*HeadDimension,NumberOfHead*HeadDimension);


 CTensorMath<type_t>::Fill(cTensor_dWq,0);
 CTensorMath<type_t>::Fill(cTensor_dWk,0);
 CTensorMath<type_t>::Fill(cTensor_dWv,0);
 CTensorMath<type_t>::Fill(cTensor_M_Wq,0);
 CTensorMath<type_t>::Fill(cTensor_V_Wq,0);
 CTensorMath<type_t>::Fill(cTensor_M_Wk,0);
 CTensorMath<type_t>::Fill(cTensor_V_Wk,0);
 CTensorMath<type_t>::Fill(cTensor_M_Wv,0);
 CTensorMath<type_t>::Fill(cTensor_V_Wv,0);


 cTensor_Bq_Batch=CTensor<type_t>(BatchSize,1,1,NumberOfHead*HeadDimension);
 cTensor_Bk_Batch=CTensor<type_t>(BatchSize,1,1,NumberOfHead*HeadDimension);
 cTensor_Bv_Batch=CTensor<type_t>(BatchSize,1,1,NumberOfHead*HeadDimension);
 cTensor_dBq_Batch=CTensor<type_t>(BatchSize,1,1,NumberOfHead*HeadDimension);
 cTensor_dBk_Batch=CTensor<type_t>(BatchSize,1,1,NumberOfHead*HeadDimension);
 cTensor_dBv_Batch=CTensor<type_t>(BatchSize,1,1,NumberOfHead*HeadDimension);
 cTensor_dBq=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_dBk=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_dBv=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_M_Bq=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_V_Bq=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_M_Bk=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_V_Bk=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_M_Bv=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_V_Bv=CTensor<type_t>(1,1,1,NumberOfHead*HeadDimension);
 cTensor_Ones=CTensor<type_t>(BatchSize,1,TSize,1);

 //инициализация: bias — нули (как в GPT-2), моменты — нули, единицы — единицы
 CTensorMath<type_t>::Fill(cTensor_Bq,0);
 CTensorMath<type_t>::Fill(cTensor_Bk,0);
 CTensorMath<type_t>::Fill(cTensor_Bv,0);
 CTensorMath<type_t>::Fill(cTensor_M_Bq,0);
 CTensorMath<type_t>::Fill(cTensor_V_Bq,0);
 CTensorMath<type_t>::Fill(cTensor_M_Bk,0);
 CTensorMath<type_t>::Fill(cTensor_V_Bk,0);
 CTensorMath<type_t>::Fill(cTensor_M_Bv,0);
 CTensorMath<type_t>::Fill(cTensor_V_Bv,0);
 CTensorMath<type_t>::Fill(cTensor_Ones,1);

 iNetLayer_SoftMaxInput->TrainingStart();
 iNetLayer_SoftMax->TrainingStart();
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
 cTensor_PrevLayerError=CTensor<type_t>(1,1,1,1);

 iNetLayer_SoftMaxInput->TrainingStop();
 iNetLayer_SoftMax->TrainingStop();
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::TrainingBackward(bool create_delta_weight)
{
 //выход предыдущего слоя — тот же X, что использовался в прямом проходе
 CTensor<type_t>& cTensor_X=PrevLayerPtr->GetOutputTensor();

 //обратный коэффициент масштабирования: 1/sqrt(Dk) — тот же, что в Forward
 const type_t attention_scale=static_cast<type_t>(1.0/sqrt(static_cast<double>(HeadDimension)));

 //1. Обратное merge-преобразование градиента выхода:
 //   dOh[w,h,t,k] = dH[w,0,t,h*Dk+k]
 for(uint32_t w=0;w<BatchSize;w++)
 {
  for(uint32_t z=0;z<NumberOfHead;z++)
  {
   for(uint32_t y=0;y<TSize;y++)
   {
    for(uint32_t x=0;x<HeadDimension;x++)
    {
     cTensor_dOh.SetElement(w,z,y,x,cTensor_Delta.GetElement(w,0,y,z*HeadDimension+x));
    }
   }
  }
 }


 //2. Oh = P @ Vh  →  dVh = P.T @ dOh;  dP = dOh @ Vh.T
 CTensorMath<type_t>::LeftTransposeMul(cTensor_dVh,iNetLayer_SoftMax->GetOutputTensor(),cTensor_dOh);
 CTensorMath<type_t>::RightTransposeMul(cTensor_dP,cTensor_dOh,cTensor_Vh);


 //3. Обратный проход softmax: dS_scaled = Jacobian(dP)
 //   после этого градиент лежит в тензоре входного слоя softmax [Batch,h,T,T]

 iNetLayer_SoftMax->SetOutputError(cTensor_dP);
 iNetLayer_SoftMax->TrainingBackward(create_delta_weight);
 CTensor<type_t>& cTensor_dS=iNetLayer_SoftMaxInput->GetDeltaTensor();


 //4. Обратный масштаб: dSh = dS_scaled * (1/sqrt(Dk))
 CTensorMath<type_t>::Mul(cTensor_dSh,cTensor_dS,attention_scale);

 //5. Sh = Qh @ Kh.T  →  dQh = dSh @ Kh;  dKh = dSh.T @ Qh
 CTensorMath<type_t>::Mul(cTensor_dQh,cTensor_dSh,cTensor_Kh);
 CTensorMath<type_t>::LeftTransposeMul(cTensor_dKh,cTensor_dSh,cTensor_Qh);




 //6. Обратное split-преобразование: dQ,dK,dV — зеркально прямому split-у
 for(uint32_t w=0;w<BatchSize;w++)
 {
  for(uint32_t z=0;z<NumberOfHead;z++)
  {
   for(uint32_t y=0;y<TSize;y++)
   {
    for(uint32_t x=0;x<HeadDimension;x++)
    {
     cTensor_dQ.SetElement(w,0,y,z*HeadDimension+x,cTensor_dQh.GetElement(w,z,y,x));
     cTensor_dK.SetElement(w,0,y,z*HeadDimension+x,cTensor_dKh.GetElement(w,z,y,x));
     cTensor_dV.SetElement(w,0,y,z*HeadDimension+x,cTensor_dVh.GetElement(w,z,y,x));
    }
   }
  }
 }

 //7. Градиенты весов и смещений
 if (create_delta_weight)
 {
  //7.1 dWq += Σ_батч X.T @ dQ: батчевое умножение даёт [Batch,1,D,D], затем сумма по W
  //    в аккумулятор dWq: dWq = 1*sum_W(dWq) + 1*sum_W(dWq_Batch)
  CTensorMath<type_t>::LeftTransposeMul(cTensor_dWq_Batch,cTensor_X,cTensor_dQ);
  CTensorMath<type_t>::AddSumW(cTensor_dWq,cTensor_dWq,cTensor_dWq_Batch,1,1);

  CTensorMath<type_t>::LeftTransposeMul(cTensor_dWk_Batch,cTensor_X,cTensor_dK);
  CTensorMath<type_t>::AddSumW(cTensor_dWk,cTensor_dWk,cTensor_dWk_Batch,1,1);

  CTensorMath<type_t>::LeftTransposeMul(cTensor_dWv_Batch,cTensor_X,cTensor_dV);
  CTensorMath<type_t>::AddSumW(cTensor_dWv,cTensor_dWv,cTensor_dWv_Batch,1,1);

  //7.2 градиенты смещений: dBq[x] = Σ_w Σ_t dQ[w,0,t,x], накопление в dBq
  CTensorMath<type_t>::LeftTransposeMul(cTensor_dBq_Batch,cTensor_Ones,cTensor_dQ);
  CTensorMath<type_t>::LeftTransposeMul(cTensor_dBk_Batch,cTensor_Ones,cTensor_dK);
  CTensorMath<type_t>::LeftTransposeMul(cTensor_dBv_Batch,cTensor_Ones,cTensor_dV);

  CTensorMath<type_t>::AddSumW(cTensor_dBq,cTensor_dBq,cTensor_dBq_Batch,1,1);
  CTensorMath<type_t>::AddSumW(cTensor_dBk,cTensor_dBk,cTensor_dBk_Batch,1,1);
  CTensorMath<type_t>::AddSumW(cTensor_dBv,cTensor_dBv,cTensor_dBv_Batch,1,1);
 }

 //8. Градиент входа: dX = dQ@Wq.T + dK@Wk.T + dV@Wv.T
 //используем те же веса, что были задействованы в прямом проходе
 const CTensor<type_t>* ptr_wq=(UseEMA? &cTensor_Wq_EMA:&cTensor_Wq);
 const CTensor<type_t>* ptr_wk=(UseEMA? &cTensor_Wk_EMA:&cTensor_Wk);
 const CTensor<type_t>* ptr_wv=(UseEMA? &cTensor_Wv_EMA:&cTensor_Wv);

 CTensorMath<type_t>::RightTransposeMul(cTensor_dX,cTensor_dQ,*ptr_wq);
 CTensorMath<type_t>::RightTransposeMul(cTensor_dX2,cTensor_dK,*ptr_wk);
 CTensorMath<type_t>::Add(cTensor_PrevLayerError,cTensor_dX,cTensor_dX2,1,1);
 CTensorMath<type_t>::RightTransposeMul(cTensor_dX2,cTensor_dV,*ptr_wv);
 CTensorMath<type_t>::Add(cTensor_PrevLayerError,cTensor_PrevLayerError,cTensor_dX2,1,1);//аккумуляция: output==left безопасно (каждый поток пишет только свою ячейку)

 //передаём ошибку предыдущему слою
 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::TrainingResetDeltaWeight(void)
{
 CTensorMath<type_t>::Fill(cTensor_dWq,0);
 CTensorMath<type_t>::Fill(cTensor_dWk,0);
 CTensorMath<type_t>::Fill(cTensor_dWv,0);
 CTensorMath<type_t>::Fill(cTensor_dBq,0);
 CTensorMath<type_t>::Fill(cTensor_dBk,0);
 CTensorMath<type_t>::Fill(cTensor_dBv,0);

 iNetLayer_SoftMaxInput->TrainingResetDeltaWeight();
 iNetLayer_SoftMax->TrainingResetDeltaWeight();

}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
 iNetLayer_SoftMaxInput->TrainingUpdateWeight(speed,iteration,batch_scale);
 iNetLayer_SoftMax->TrainingUpdateWeight(speed,iteration,batch_scale);

 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_ADAM)
 {
  CTensorMath<type_t>::Adam(cTensor_Wq,cTensor_dWq,cTensor_M_Wq,cTensor_V_Wq,BatchSize,speed*batch_scale,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_Wk,cTensor_dWk,cTensor_M_Wk,cTensor_V_Wk,BatchSize,speed*batch_scale,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_Wv,cTensor_dWv,cTensor_M_Wv,cTensor_V_Wv,BatchSize,speed*batch_scale,Beta1,Beta2,Epsilon,iteration);

  //градиенты bias уже просуммированы по всему батчу, поэтому batch_size=1
  CTensorMath<type_t>::Adam(cTensor_Bq,cTensor_dBq,cTensor_M_Bq,cTensor_V_Bq,1,speed*batch_scale,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_Bk,cTensor_dBk,cTensor_M_Bk,cTensor_V_Bk,1,speed*batch_scale,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_Bv,cTensor_dBv,cTensor_M_Bv,cTensor_V_Bv,1,speed*batch_scale,Beta1,Beta2,Epsilon,iteration);
 }
 if (EMAEnabled==true)
 {
  CTensorMath<type_t>::Add(cTensor_Bq_EMA,cTensor_Bq_EMA,cTensor_Bq,EMA_K,1.0-EMA_K);
  CTensorMath<type_t>::Add(cTensor_Bk_EMA,cTensor_Bk_EMA,cTensor_Bk,EMA_K,1.0-EMA_K);
  CTensorMath<type_t>::Add(cTensor_Bv_EMA,cTensor_Bv_EMA,cTensor_Bv,EMA_K,1.0-EMA_K);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerSelfAttention<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::SetOutputError(CTensor<type_t>& error)
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
void CNetLayerSelfAttention<type_t>::ClipWeight(type_t min,type_t max)
{
 iNetLayer_SoftMaxInput->ClipWeight(min,max);
 iNetLayer_SoftMax->ClipWeight(min,max);

 //TODO: добавить clip весов слоя
}

//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
 iNetLayer_SoftMaxInput->SetTimeStep(index,time_step);
 iNetLayer_SoftMax->SetTimeStep(index,time_step);
}

//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" SelfAttention: input ",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::PrintOutputTensorSize(const std::string &name)
{
 cTensor_H.RestoreSize();
 GetOutputTensor().Print(name+" SelfAttention: output",false);
}
//----------------------------------------------------------------------------------------------------
/*!<разрешить/запретить использование усреднённых весов
\param[in] state - разрешить запретить использование усреднённых весов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerSelfAttention<type_t>::EnableEMA(bool state)
{
 EMAEnabled=state;
 if (state==true)
 {
  cTensor_Wq_EMA=cTensor_Wq;
  cTensor_Wk_EMA=cTensor_Wk;
  cTensor_Wv_EMA=cTensor_Wv;
  cTensor_Bq_EMA=cTensor_Bq;
  cTensor_Bk_EMA=cTensor_Bk;
  cTensor_Bv_EMA=cTensor_Bv;
 }
 else
 {
  cTensor_Wq_EMA=CTensor<type_t>(1,1,1,1);
  cTensor_Wk_EMA=CTensor<type_t>(1,1,1,1);
  cTensor_Wv_EMA=CTensor<type_t>(1,1,1,1);
  cTensor_Bq_EMA=CTensor<type_t>(1,1,1,1);
  cTensor_Bk_EMA=CTensor<type_t>(1,1,1,1);
  cTensor_Bv_EMA=CTensor<type_t>(1,1,1,1);
 }
 iNetLayer_SoftMaxInput->EnableEMA(state);
 iNetLayer_SoftMax->EnableEMA(state);

}
//----------------------------------------------------------------------------------------------------
/*!загрузить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerSelfAttention<type_t>::LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size)
{
 cTensor_Wq_EMA.Load(iDataStream_Ptr,check_size);
 cTensor_Wk_EMA.Load(iDataStream_Ptr,check_size);
 cTensor_Wv_EMA.Load(iDataStream_Ptr,check_size);
 cTensor_Bq_EMA.Load(iDataStream_Ptr,check_size);
 cTensor_Bk_EMA.Load(iDataStream_Ptr,check_size);
 cTensor_Bv_EMA.Load(iDataStream_Ptr,check_size);

 iNetLayer_SoftMaxInput->LoadEMAWeight(iDataStream_Ptr,check_size);
 iNetLayer_SoftMax->LoadEMAWeight(iDataStream_Ptr,check_size);

 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerSelfAttention<type_t>::SaveEMAWeight(IDataStream *iDataStream_Ptr)
{
 cTensor_Wq_EMA.Save(iDataStream_Ptr);
 cTensor_Wk_EMA.Save(iDataStream_Ptr);
 cTensor_Wv_EMA.Save(iDataStream_Ptr);
 cTensor_Bq_EMA.Save(iDataStream_Ptr);
 cTensor_Bk_EMA.Save(iDataStream_Ptr);
 cTensor_Bv_EMA.Save(iDataStream_Ptr);

 iNetLayer_SoftMaxInput->SaveEMAWeight(iDataStream_Ptr);
 iNetLayer_SoftMax->SaveEMAWeight(iDataStream_Ptr);
 return(true);
}


#endif
