#ifndef C_NET_LAYER_BACK_CONVOLUTION_H
#define C_NET_LAYER_BACK_CONVOLUTION_H

//****************************************************************************************************
//\file Обратно-свёрточный слой нейронов
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
//!Обратно-свёрточный слой нейронов
//****************************************************************************************************
template<class type_t>
class CNetLayerBackConvolution:public INetLayer<type_t>
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

  CTensor<type_t> cTensor_H;///<тензор значений нейронов до функции активации
  CTensor<type_t> cTensor_Kernel;///<ядра свёртки  [1,количество ядер,kx*ky*kz]
  CTensor<type_t> cTensor_Bias;///<сдвиги [количество,1,1]

  size_t Kernel_X;///<размер ядра по X
  size_t Kernel_Y;///<размер ядра по Y
  size_t Kernel_Z;///<размер ядра по Z
  size_t Kernel_Amount;///<количество ядер
  size_t Padding_X;///<дополнение нулями по X
  size_t Padding_Y;///<дополнение нулями по Y
  size_t Stride_X;///<шаг свёртки по x
  size_t Stride_Y;///<шаг свёртки по Y

  size_t InputSize_X;///<размер входного тензора по X
  size_t InputSize_Y;///<размер входного тензора по Y
  size_t InputSize_Z;///<размер входного тензора по Z

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_dKernel;///<поправки для ядер свёртки [1,количество ядер,kx*ky*kz]
  CTensor<type_t> cTensor_dBias;///<поправки для сдвигов [количество,1,1]

  CTensor<type_t> cTensor_dKernel_Batch;///<поправки для ядер свёртки [BatchSize,1,количество ядер,kx*ky*kz] для каждого входа
  CTensor<type_t> cTensor_dBias_Batch;///<поправки для сдвигов [BatchSize,количество,1,1] для каждого входа

  CTensor<type_t> cTensor_Delta;///<тензоры дельты слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя

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
  CNetLayerBackConvolution(size_t kernel_size,size_t kernel_depth,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);
  CNetLayerBackConvolution(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerBackConvolution();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(size_t kernel_size,size_t kernel_depth,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y,INetLayer<type_t> *prev_layer_ptr=NULL,size_t batch_size=1);///<создать слой
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
  void TrainingUpdateWeight(double speed,double iteration);///<выполнить обновления весов
  CTensor<type_t>& GetDeltaTensor(void);///<получить ссылку на тензор дельты слоя

  void SetOutputError(CTensor<type_t>& error);///<задать ошибку и расчитать дельту

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
CNetLayerBackConvolution<type_t>::CNetLayerBackConvolution(size_t kernel_size,size_t kernel_depth,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 Padding_X=padding_x;///<дополнение нулями по X
 Padding_Y=padding_y;///<дополнение нулями по Y
 Stride_X=stride_x;///<шаг свёртки по x
 Stride_Y=stride_y;///<шаг свёртки по Y
 Create(kernel_size,kernel_depth,stride_x,stride_y,padding_x,padding_y,prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerBackConvolution<type_t>::CNetLayerBackConvolution(void)
{
 Create(1,1,1,1,0,0);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerBackConvolution<type_t>::~CNetLayerBackConvolution()
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
\param[in] kernel_size Размер одного ядра свёртки
\param[in] kernel_depth Глубина одного ядра свёртки
\param[in] neuron_function Функция активации нейронов
\param[in] prev_layer_ptr Указатель на класс предшествующего слоя (NULL-слой входной)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::Create(size_t kernel_size,size_t kernel_depth,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y,INetLayer<type_t> *prev_layer_ptr,size_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;

 BatchSize=batch_size;

 if (prev_layer_ptr==NULL)//слой без предшествующего считается входным
 {
  throw("Свёрточный слой не может быть входным. Создайте слой CNetLayerConvolutionInput перед ним");
 }
 //размер входного тензора
 size_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();

 InputSize_X=input_x;
 InputSize_Y=input_y;
 InputSize_Z=input_z;

 Padding_X=padding_x;///<дополнение нулями по X
 Padding_Y=padding_y;///<дополнение нулями по Y
 Stride_X=stride_x;///<шаг свёртки по x
 Stride_Y=stride_y;///<шаг свёртки по Y

 //запомним размер ядер и их количество
 Kernel_X=kernel_size;
 Kernel_Y=kernel_size;
 Kernel_Z=kernel_depth;
 Kernel_Amount=input_z;

 if (Kernel_Amount==0) throw("В свёрточном слое должно быть хотя бы одно ядро свёртки");

 //создаём ядра свёрток
 cTensor_Kernel=CTensor<type_t>(1,Kernel_Amount,Kernel_Z*Kernel_X*Kernel_Y);
 //создаём сдвиги свёрток
 cTensor_Bias=CTensor<type_t>(1,Kernel_Amount,1,1);

 //размер выходного тензора
 int32_t output_x=Stride_X*(input_x-1)+Kernel_X-2*Padding_X;
 int32_t output_y=Stride_Y*(input_y-1)+Kernel_Y-2*Padding_Y;
 int32_t output_z=Kernel_Z;
 //создаём выходные тензоры
 cTensor_H=CTensor<type_t>(BatchSize,output_z,output_y,output_x);
 //задаём предшествующему слою, что мы его последующий слой
 prev_layer_ptr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию весов и сдвигов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::Reset(void)
{
 if (PrevLayerPtr==NULL) return;

 for(size_t n=0;n<Kernel_Amount;n++)
 {
  type_t size=static_cast<type_t>(Kernel_X*Kernel_Y*Kernel_Z);
  type_t koeff=static_cast<type_t>(sqrt(2.0/size));
  CTensor<type_t> cTensor_Rand(1,1,1,size);
  CRandom<type_t>::SetRandomNormal(cTensor_Rand,-koeff,koeff);
  //веса
  for(size_t m=0;m<Kernel_X*Kernel_Y*Kernel_Z;m++)
  {
   //используем метод инициализации He (Ге)
   //type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
   //type_t init=rnd*koeff;
   type_t init=cTensor_Rand.GetElement(0,0,m);
   cTensor_Kernel.SetElement(0,0,n,m,init);
  }
 }
 //сдвиги
 type_t size=static_cast<type_t>(Kernel_Amount);
 type_t koeff=static_cast<type_t>(sqrt(2.0/size));
 for(size_t z=0;z<Kernel_Amount;z++)
 {
  //используем метод инициализации He (Ге)
  //type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
  type_t init=0;//rnd*koeff;//TODO: обнулено, пока не будет ясно, как считать поправки к смещениям
  cTensor_Bias.SetElement(0,z,0,0,init);
 }
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::SetOutput(CTensor<type_t> &output)
{
 cTensor_H.CopyItem(output);
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::GetOutput(CTensor<type_t> &output)
{
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerBackConvolution<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerBackConvolution<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerBackConvolution<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerBackConvolution<type_t>::GetOutput(CTensor<type_t> &output) - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::Forward(void)
{
 for(size_t n=0;n<BatchSize;n++)
 {
  size_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
  size_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
  size_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
  //приведём входной тензор к нужному виду
  PrevLayerPtr->GetOutputTensor().ReinterpretSize(BatchSize,InputSize_Z,InputSize_Y,InputSize_X);

  //выполняем обратную свёртку
  CTensorConv<type_t>::BackwardConvolution(cTensor_H,PrevLayerPtr->GetOutputTensor(),cTensor_Kernel,Kernel_X,Kernel_Y,Kernel_Z,Kernel_Amount,cTensor_Bias,Stride_X,Stride_Y,Padding_X,Padding_Y);
  PrevLayerPtr->GetOutputTensor().ReinterpretSize(BatchSize,input_z,input_y,input_x);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerBackConvolution<type_t>::GetOutputTensor(void)
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
void CNetLayerBackConvolution<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerBackConvolution<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 iDataStream_Ptr->SaveInt32(Kernel_Amount);
 iDataStream_Ptr->SaveInt32(Kernel_X);
 iDataStream_Ptr->SaveInt32(Kernel_Y);
 iDataStream_Ptr->SaveInt32(Kernel_Z);
 iDataStream_Ptr->SaveInt32(Stride_X);
 iDataStream_Ptr->SaveInt32(Stride_Y);
 iDataStream_Ptr->SaveInt32(Padding_X);
 iDataStream_Ptr->SaveInt32(Padding_Y);
 cTensor_Kernel.Save(iDataStream_Ptr);
 cTensor_Bias.Save(iDataStream_Ptr);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerBackConvolution<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 if (check_size==true)
 {
  if (iDataStream_Ptr->LoadUInt32()!=Kernel_Amount) throw("Ошибка загрузки обратносвёрточного слоя: неверное количество ядер.");
  if (iDataStream_Ptr->LoadInt32()!=Kernel_X) throw("Ошибка загрузки обратносвёрточного слоя: неверный размер ядер.");
  if (iDataStream_Ptr->LoadInt32()!=Kernel_Y) throw("Ошибка загрузки обратносвёрточного слоя: неверный размер ядер.");
  if (iDataStream_Ptr->LoadInt32()!=Kernel_Z) throw("Ошибка загрузки обратносвёрточного слоя: неверный размер ядер.");
  if (iDataStream_Ptr->LoadInt32()!=Stride_X) throw("Ошибка загрузки обратносвёрточного слоя: неверный шаг свёртки.");
  if (iDataStream_Ptr->LoadInt32()!=Stride_Y) throw("Ошибка загрузки обратносвёрточного слоя: неверный шаг свёртки.");
  if (iDataStream_Ptr->LoadInt32()!=Padding_X) throw("Ошибка загрузки обратносвёрточного слоя: неверный размер дополнения.");
  if (iDataStream_Ptr->LoadInt32()!=Padding_Y) throw("Ошибка загрузки обратносвёрточного слоя: неверный размер дополнения.");
 }
 else
 {
  Kernel_Amount=iDataStream_Ptr->LoadUInt32();
  Kernel_X=iDataStream_Ptr->LoadInt32();
  Kernel_Y=iDataStream_Ptr->LoadInt32();
  Kernel_Z=iDataStream_Ptr->LoadInt32();
  Stride_X=iDataStream_Ptr->LoadInt32();
  Stride_Y=iDataStream_Ptr->LoadInt32();
  Padding_X=iDataStream_Ptr->LoadInt32();
  Padding_Y=iDataStream_Ptr->LoadInt32();
 }
 cTensor_Kernel.Load(iDataStream_Ptr,check_size);
 cTensor_Bias.Load(iDataStream_Ptr,check_size);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerBackConvolution<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
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
bool CNetLayerBackConvolution<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
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
void CNetLayerBackConvolution<type_t>::TrainingStart(void)
{
 //создаём все вспомогательные тензоры
 cTensor_Delta=cTensor_H;
 cTensor_PrevLayerError=PrevLayerPtr->GetOutputTensor();
 //создаём тензор поправок ядер слоя
 cTensor_dKernel=cTensor_Kernel;
 cTensor_dKernel_Batch=CTensor<type_t>(BatchSize,cTensor_Kernel.GetSizeZ(),cTensor_Kernel.GetSizeY(),cTensor_Kernel.GetSizeX());
 //создаём поправки сдвигов слоя
 cTensor_dBias=cTensor_Bias;
 cTensor_dBias_Batch=CTensor<type_t>(BatchSize,cTensor_Bias.GetSizeZ(),cTensor_Bias.GetSizeY(),cTensor_Bias.GetSizeX());
 //для оптимизации Adam
 cTensor_MK=cTensor_Kernel;
 cTensor_VK=cTensor_Kernel;
 cTensor_MK.Zero();
 cTensor_VK.Zero();

 cTensor_MB=cTensor_Bias;
 cTensor_VB=cTensor_Bias;
 cTensor_MB.Zero();
 cTensor_VB.Zero();
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_dKernel=CTensor<type_t>(1,1,1,1);
 cTensor_dBias=CTensor<type_t>(1,1,1,1);
 cTensor_dKernel_Batch=CTensor<type_t>(1,1,1,1);
 cTensor_dBias_Batch=CTensor<type_t>(1,1,1,1);

 cTensor_MK=CTensor<type_t>(1,1,1,1);
 cTensor_VK=CTensor<type_t>(1,1,1,1);
 cTensor_MB=CTensor<type_t>(1,1,1,1);
 cTensor_VB=CTensor<type_t>(1,1,1,1);

 cTensor_Delta=CTensor<type_t>(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::TrainingBackward(bool create_delta_weight)
{
 size_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 //приведём входной тензор к нужному виду
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(BatchSize,InputSize_Z,InputSize_Y,InputSize_X);
 cTensor_PrevLayerError.ReinterpretSize(BatchSize,InputSize_Z,InputSize_Y,InputSize_X);

 //вычисляем ошибку предшествующего слоя
 CTensor<type_t> cTensor_BiasZero=cTensor_Bias;
 cTensor_BiasZero.Zero();
 CTensorConv<type_t>::ForwardConvolution(cTensor_PrevLayerError,cTensor_Delta,cTensor_Kernel,Kernel_X,Kernel_Y,Kernel_Z,Kernel_Amount,cTensor_BiasZero,Stride_X,Stride_Y,Padding_X,Padding_Y);
 if (create_delta_weight==true)
 {
  cTensor_dKernel_Batch.Zero();
  cTensor_dBias_Batch.Zero();

  CTensorConv<type_t>::CreateBackDeltaWeightAndBias(cTensor_dKernel_Batch,Kernel_X,Kernel_Y,Kernel_Z,Kernel_Amount,cTensor_dBias_Batch,PrevLayerPtr->GetOutputTensor(),cTensor_Delta,Stride_X,Stride_Y,Padding_X,Padding_Y);

  CTensorMath<type_t>::AddSumW(cTensor_dKernel,cTensor_dKernel,cTensor_dKernel_Batch);
  CTensorMath<type_t>::AddSumW(cTensor_dBias,cTensor_dBias,cTensor_dBias_Batch);
 }
 //задаём ошибку предыдущего слоя
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(BatchSize,input_z,input_y,input_x);
 cTensor_PrevLayerError.ReinterpretSize(BatchSize,input_z,input_y,input_x);

 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::TrainingResetDeltaWeight(void)
{
 cTensor_dKernel.Zero();
 cTensor_dBias.Zero();
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::TrainingUpdateWeight(double speed,double iteration)
{
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_ADAM)
 {
  //применяем алгоритм Adam
  CTensorMath<type_t>::Adam(cTensor_Kernel,cTensor_dKernel,cTensor_MK,cTensor_VK,BatchSize,speed,Beta1,Beta2,Epsilon,iteration);
  CTensorMath<type_t>::Adam(cTensor_Bias,cTensor_dBias,cTensor_MB,cTensor_VB,BatchSize,speed,Beta1,Beta2,Epsilon,iteration);
 }
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_GRADIENT)
 {
  speed/=static_cast<double>(BatchSize);
  CTensorMath<type_t>::Sub(cTensor_Kernel,cTensor_Kernel,cTensor_dKernel,1,speed);
  CTensorMath<type_t>::Sub(cTensor_Bias,cTensor_Bias,cTensor_dBias,1,speed);
 }
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerBackConvolution<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!задать ошибку и расчитать дельту
\param[in] error Тензор ошибки
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::SetOutputError(CTensor<type_t>& error)
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
void CNetLayerBackConvolution<type_t>::ClipWeight(type_t min,type_t max)
{
 CTensorMath<type_t>::Clip(cTensor_Kernel,min,max);
 CTensorMath<type_t>::Clip(cTensor_Bias,min,max);
}

#endif
