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

  CTensor<type_t> cTensor_Z;///<тензор значений нейронов до функции активации
  CTensor<type_t> cTensor_H;///<тензор значений нейронов после функции активации
  std::vector<CTensor<type_t> > cTensor_Kernel;///<ядра свёртки
  std::vector<type_t> Bias;///<сдвиги
  NNeuron::NEURON_FUNCTION NeuronFunction;///<функция активации нейронов слоя

  size_t Padding_X;///<дополнение нулями по X
  size_t Padding_Y;///<дополнение нулями по Y
  size_t Stride_X;///<шаг свёртки по x
  size_t Stride_Y;///<шаг свёртки по Y

  size_t InputSize_X;///<размер входного тензора по X
  size_t InputSize_Y;///<размер входного тензора по Y
  size_t InputSize_Z;///<размер входного тензора по Z

  //тензоры, используемые при обучении
  std::vector<CTensor<type_t> > cTensor_dKernel;///<поправки для ядер свёртки
  std::vector<type_t> dBias;///<поправки для сдвигов
  CTensor<type_t> cTensor_Delta;///<тензор дельты слоя
  CTensor<type_t> cTensor_PrevLayerError;///<тензор ошибки предыдущего слоя

  //для оптимизации Adam
  std::vector<CTensor<type_t> > cTensor_MK;///<тензор фильтра 1
  std::vector<CTensor<type_t> > cTensor_VK;///<тензор фильтра 2
  std::vector<type_t> MB;///<коэффициент фильтра 1 сдвигов
  std::vector<type_t> VB;///<коэффициент фильтра 2 сдвигов

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerBackConvolution(size_t kernel_size,size_t kernel_depth,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y,NNeuron::NEURON_FUNCTION neuron_function=NNeuron::NEURON_FUNCTION_SIGMOID,INetLayer<type_t> *prev_layer_ptr=NULL);
  CNetLayerBackConvolution(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerBackConvolution();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(size_t kernel_size,size_t kernel_depth,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y,NNeuron::NEURON_FUNCTION neuron_function=NNeuron::NEURON_FUNCTION_SIGMOID,INetLayer<type_t> *prev_layer_ptr=NULL);///<создать слой
  void Reset(void);///<выполнить инициализацию весов и сдвигов
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
CNetLayerBackConvolution<type_t>::CNetLayerBackConvolution(size_t kernel_size,size_t kernel_depth,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y,NNeuron::NEURON_FUNCTION neuron_function,INetLayer<type_t> *prev_layer_ptr)
{
 Padding_X=padding_x;///<дополнение нулями по X
 Padding_Y=padding_y;///<дополнение нулями по Y
 Stride_X=stride_x;///<шаг свёртки по x
 Stride_Y=stride_y;///<шаг свёртки по Y

 Create(kernel_size,kernel_depth,stride_x,stride_y,padding_x,padding_y,neuron_function,prev_layer_ptr);
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


//----------------------------------------------------------------------------------------------------
/*!получить случайное число
\param[in] max_value Максимальное значение случайного числа
\return Случайное число в диапазоне [0...max_value]
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CNetLayerBackConvolution<type_t>::GetRandValue(type_t max_value)
{
 return((static_cast<type_t>(rand())*max_value)/static_cast<type_t>(RAND_MAX));
}

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
void CNetLayerBackConvolution<type_t>::Create(size_t kernel_size,size_t kernel_depth,int32_t stride_x,int32_t stride_y,int32_t padding_x,int32_t padding_y,NNeuron::NEURON_FUNCTION neuron_function,INetLayer<type_t> *prev_layer_ptr)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;
 NeuronFunction=neuron_function;
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

 //создаём ядра свёрток
 size_t kernel_amount=input_z;

 if (kernel_amount==0) throw("В свёрточном слое должно быть хотя бы одно ядро свёртки");

 cTensor_Kernel.resize(kernel_amount);
 for(size_t n=0;n<kernel_amount;n++) cTensor_Kernel[n]=CTensor<type_t>(kernel_depth,kernel_size,kernel_size);
 //создаём сдвиги свёрток
 Bias.resize(kernel_amount);

 size_t kernel_x=cTensor_Kernel[0].GetSizeX();
 size_t kernel_y=cTensor_Kernel[0].GetSizeY();
 size_t kernel_z=cTensor_Kernel[0].GetSizeZ();
 //размер выходного тензора
 int32_t output_x=Stride_X*(input_x-1)+kernel_x-2*Padding_X;
 int32_t output_y=Stride_Y*(input_y-1)+kernel_y-2*Padding_Y;
 int32_t output_z=kernel_z;
 //создаём выходные тензоры
 cTensor_Z=CTensor<type_t>(output_z,output_y,output_x);
 cTensor_H=CTensor<type_t>(output_z,output_y,output_x);
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

 for(size_t n=0;n<cTensor_Kernel.size();n++)
 {
  type_t size=static_cast<type_t>(cTensor_Kernel[n].GetSizeX()*cTensor_Kernel[n].GetSizeY()*cTensor_Kernel[n].GetSizeZ());
  type_t koeff=static_cast<type_t>(sqrt(2.0/size));
  //веса
  for(size_t z=0;z<cTensor_Kernel[n].GetSizeZ();z++)
  {
   for(size_t y=0;y<cTensor_Kernel[n].GetSizeY();y++)
   {
    for(size_t x=0;x<cTensor_Kernel[n].GetSizeX();x++)
    {
     //используем метод инициализации He (Ге)
     type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
     type_t init=rnd*koeff;
     cTensor_Kernel[n].SetElement(z,y,x,init);
    }
   }
  }
 }
 //сдвиги
 type_t size=static_cast<type_t>(Bias.size());
 type_t koeff=static_cast<type_t>(sqrt(2.0/size));
 for(size_t y=0;y<Bias.size();y++)
 {
  //используем метод инициализации He (Ге)
  type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
  type_t init=rnd*koeff;
  Bias[y]=0;//init;//TODO: обнулено, пока не будет ясно, как считать поправки к смещениям
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
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::Forward(void)
{
 size_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 //приведём входной тензор к нужному виду
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);

 //выполняем обратную свёртку
 CTensorConv<type_t>::BackwardConvolution(cTensor_Z,PrevLayerPtr->GetOutputTensor(),cTensor_Kernel,Bias,Stride_X,Stride_Y,Padding_X,Padding_Y);
 //применяем функцию активации нейронов
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_SIGMOID) CTensorApplyFunc<type_t>::ApplySigmoid(cTensor_H,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_RELU) CTensorApplyFunc<type_t>::ApplyReLU(cTensor_H,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LEAKY_RELU) CTensorApplyFunc<type_t>::ApplyLeakyReLU(cTensor_H,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LINEAR) CTensorApplyFunc<type_t>::ApplyLinear(cTensor_H,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_TANGENCE) CTensorApplyFunc<type_t>::ApplyTangence(cTensor_H,cTensor_Z);
 //if (NeuronFunction==NNeuron::NEURON_FUNCTION_SOFTMAX) CTensorApplyFunc<type_t>::ApplySoftMax(cTensor_H,cTensor_Z);

 PrevLayerPtr->GetOutputTensor().ReinterpretSize(input_z,input_y,input_x);
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
 iDataStream_Ptr->SaveUInt32(NeuronFunction);
 iDataStream_Ptr->SaveUInt32(cTensor_Kernel.size());
 for(size_t n=0;n<cTensor_Kernel.size();n++)
 {
  cTensor_Kernel[n].Save(iDataStream_Ptr);
  iDataStream_Ptr->SaveDouble(Bias[n]);
  //cTensor_Kernel[n].Print("BackKernel",true);
 }
 iDataStream_Ptr->SaveInt32(Stride_X);
 iDataStream_Ptr->SaveInt32(Stride_Y);
 iDataStream_Ptr->SaveInt32(Padding_X);
 iDataStream_Ptr->SaveInt32(Padding_Y);
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerBackConvolution<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 NeuronFunction=iDataStream_Ptr->LoadUInt32();
 size_t kernel_amount=iDataStream_Ptr->LoadUInt32();
 cTensor_Kernel.resize(kernel_amount);
 for(size_t n=0;n<kernel_amount;n++)
 {
  cTensor_Kernel[n].Load(iDataStream_Ptr);
  Bias[n]=static_cast<type_t>(iDataStream_Ptr->LoadDouble());
 }
 Stride_X=iDataStream_Ptr->LoadInt32();
 Stride_Y=iDataStream_Ptr->LoadInt32();
 Padding_X=iDataStream_Ptr->LoadInt32();
 Padding_Y=iDataStream_Ptr->LoadInt32();
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
 cTensor_dKernel.resize(cTensor_Kernel.size());
 for(size_t n=0;n<cTensor_dKernel.size();n++) cTensor_dKernel[n]=cTensor_Kernel[n];
 //создаём поправки сдвигов слоя
 dBias.resize(cTensor_dKernel.size());

 //для оптимизации Adam
 cTensor_MK.resize(cTensor_Kernel.size());
 cTensor_VK.resize(cTensor_Kernel.size());
 MB.resize(cTensor_Kernel.size());
 VB.resize(cTensor_Kernel.size());
 for(size_t n=0;n<cTensor_dKernel.size();n++)
 {
  cTensor_MK[n]=cTensor_Kernel[n];
  cTensor_VK[n]=cTensor_Kernel[n];

  cTensor_MK[n].Zero();
  cTensor_VK[n].Zero();

  MB[n]=0;
  VB[n]=0;
 }
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::TrainingStop(void)
{
 //удаляем все вспомогательные тензоры
 cTensor_dKernel.clear();
 dBias.clear();

 cTensor_MK.clear();
 cTensor_VK.clear();
 MB.clear();
 VB.clear();

 cTensor_Delta=CTensor<type_t>(1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::TrainingBackward(void)
{
 size_t input_x=PrevLayerPtr->GetOutputTensor().GetSizeX();
 size_t input_y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 size_t input_z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 //приведём входной тензор к нужному виду
 PrevLayerPtr->GetOutputTensor().ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);
 cTensor_PrevLayerError.ReinterpretSize(InputSize_Z,InputSize_Y,InputSize_X);

 //вычисляем ошибку предшествующего слоя
 std::vector<type_t> bias(cTensor_dKernel.size(),0);
 CTensorConv<type_t>::ForwardConvolution(cTensor_PrevLayerError,cTensor_Delta,cTensor_Kernel,bias,Stride_X,Stride_Y,Padding_X,Padding_Y);
 CTensorConv<type_t>::CreateBackDeltaWeightAndBias(cTensor_dKernel,dBias,PrevLayerPtr->GetOutputTensor(),cTensor_Delta,Stride_X,Stride_Y,Padding_X,Padding_Y);
 //задаём ошибку предыдущего слоя

 PrevLayerPtr->GetOutputTensor().ReinterpretSize(input_z,input_y,input_x);
 cTensor_PrevLayerError.ReinterpretSize(input_z,input_y,input_x);

 PrevLayerPtr->SetOutputError(cTensor_PrevLayerError);
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerBackConvolution<type_t>::TrainingResetDeltaWeight(void)
{
 for(size_t n=0;n<cTensor_dKernel.size();n++)
 {
  cTensor_dKernel[n].Zero();
  dBias[n]=0;
 }
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
  double beta1=0.9;
  double beta2=0.999;
  static const double epsilon=1E-8;
  //применяем алгоритм Adam
  for(size_t n=0;n<cTensor_Kernel.size();n++)
  {
   CTensorMath<type_t>::Adam(cTensor_Kernel[n],cTensor_dKernel[n],cTensor_MK[n],cTensor_VK[n],speed,beta1,beta2,epsilon,iteration);

   type_t dw=dBias[n];
   type_t m=MB[n];
   type_t v=VB[n];

   m=beta1*m+(1-beta1)*dw;
   v=beta2*v+(1-beta2)*dw*dw;

   type_t mc=m/(1.0-pow(beta1,iteration));
   type_t vc=v/(1.0-pow(beta2,iteration));

   dw=speed*mc/sqrt(vc+epsilon);

   MB[n]=m;
   VB[n]=v;

   Bias[n]-=dw;
  }
 }
 if (INetLayer<type_t>::GetTrainingMode()==INetLayer<type_t>::TRAINING_MODE_GRADIENT)
 {
  for(size_t n=0;n<cTensor_Kernel.size();n++)
  {
   CTensorMath<type_t>::Sub(cTensor_Kernel[n],cTensor_Kernel[n],cTensor_dKernel[n],1,speed);
   Bias[n]-=dBias[n]*speed;
  }
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
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_SIGMOID) CTensorApplyFunc<type_t>::ApplyDifferentialSigmoid(cTensor_Delta,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_RELU) CTensorApplyFunc<type_t>::ApplyDifferentialReLU(cTensor_Delta,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LEAKY_RELU) CTensorApplyFunc<type_t>::ApplyDifferentialLeakyReLU(cTensor_Delta,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_LINEAR) CTensorApplyFunc<type_t>::ApplyDifferentialLinear(cTensor_Delta,cTensor_Z);
 if (NeuronFunction==NNeuron::NEURON_FUNCTION_TANGENCE) CTensorApplyFunc<type_t>::ApplyDifferentialTangence(cTensor_Delta,cTensor_Z);
 //if (NeuronFunction==NNeuron::NEURON_FUNCTION_SOFTMAX) CTensorApplyFunc<type_t>::ApplyDifferentialSoftMax(cTensor_Delta,cTensor_Z);

 CTensorMath<type_t>::TensorItemProduction(cTensor_Delta,error,cTensor_Delta);
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
// return;
 for(size_t n=0;n<cTensor_Kernel.size();n++) CTensorMath<type_t>::Clip(cTensor_Kernel[n],min,max);
 for(size_t n=0;n<Bias.size();n++)
 {
  if (Bias[n]>max) Bias[n]=max;
  if (Bias[n]<min) Bias[n]=min;
 }
}

#endif
