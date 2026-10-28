#ifndef C_NET_LAYER_POSITIONAL_ENCODING_H
#define C_NET_LAYER_POSITIONAL_ENCODING_H

//****************************************************************************************************
//\file Слой позиционного кодирования (синусоидальное, "Attention Is All You Need")
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
//! Слой синусоидального позиционного кодирования
/*!Формула:
   PE[pos, 2i]   = sin(pos / 10000^(2i/D))
   PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
   где pos in [0,T) — индекс токена, i in [0,D/2) — индекс частотной компоненты.
   PE не обучается (детерминированный), распространяется на весь батч.
   Используется перед слоем самовнимания для нарушения перестановочной инвариантности.
*/
//****************************************************************************************************
template<class type_t>
class CNetLayerPositionalEncoding:public INetLayer<type_t>
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

  uint32_t TSize;///<количество токенов (размерность по Y входа)
  uint32_t DSize;///<размерность признаков (размерность по X входа)

  CTensor<type_t> cTensor_PE;///<тензор позиционного кодирования [1,1,T,D] — канонический, необучаемый
  CTensor<type_t> cTensor_PE_Batch;///<тензор PE, размазанный по батчу [Batch,1,T,D] — для прямого сложения без broadcasting
  CTensor<type_t> cTensor_H;///<выходной тензор [Batch,1,T,D]

  //тензоры, используемые при обучении
  CTensor<type_t> cTensor_Delta;///<градиент выхода слоя dH [Batch,1,T,D]

  using INetLayer<type_t>::Beta1;///<параметры алгоритма Adam
  using INetLayer<type_t>::Beta2;
  using INetLayer<type_t>::Epsilon;
  //режим усреднения
  using INetLayer<type_t>::EMAEnabled;
  using INetLayer<type_t>::UseEMA;
  using INetLayer<type_t>::EMA_K;
  //ограничение нормы
  using INetLayer<type_t>::ClipByNormThresHold;///<ограничение нормы
  using INetLayer<type_t>::InferenceMode;
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CNetLayerPositionalEncoding(INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);
  CNetLayerPositionalEncoding(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CNetLayerPositionalEncoding();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Create(INetLayer<type_t> *prev_layer_ptr=NULL,uint32_t batch_size=1);///<создать слой
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

  void ClipWeight(type_t min,type_t max);///<ограничить веса в диапазон

  void SetTimeStep(uint32_t index,uint32_t time_step);///<задать временной шаг

  void PrintInputTensorSize(const std::string &name);///<вывести размерность входного тензора слоя
  void PrintOutputTensorSize(const std::string &name);///<вывести размерность выходного тензора слоя

  void EnableEMA(bool state);///<разрешить/запретить использование усреднённых весов
  bool LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size=false);///<загрузить усреднённые веса
  bool SaveEMAWeight(IDataStream *iDataStream_Ptr);///<сохранить усреднённые веса

  void SetInferenceMode(bool state) override;///<включить режим вывода
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  void BuildPositionalEncoding(void);///<построить синусоидальное позиционное кодирование и заполнить cTensor_PE / cTensor_PE_Batch
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerPositionalEncoding<type_t>::CNetLayerPositionalEncoding(INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 Create(prev_layer_ptr,batch_size);
}
//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerPositionalEncoding<type_t>::CNetLayerPositionalEncoding(void)
{
 Create(NULL,1);
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CNetLayerPositionalEncoding<type_t>::~CNetLayerPositionalEncoding()
{
}

//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
/*!построить синусоидальное позиционное кодирование
   PE[pos, 2i]   = sin(pos / 10000^(2i/D))
   PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
   Если D нечётное — последний элемент заполняется sin для полноты (в современных архитектурах D обычно чётное).
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::BuildPositionalEncoding(void)
{
 CTensorMath<type_t>::Fill(cTensor_PE,0);
 CTensorMath<type_t>::Fill(cTensor_PE_Batch,0);

 const double log_base=log(10000.0);
 const uint32_t D=DSize;

 //сначала заполняем канонический PE [1,1,T,D]
 for(uint32_t pos=0;pos<TSize;pos++)
 {
  for(uint32_t i=0;i<D/2;i++)
  {
   double freq=exp(-log_base*(2.0*i)/(double)D);
   double angle=(double)pos*freq;

   type_t sin_val=static_cast<type_t>(sin(angle));
   type_t cos_val=static_cast<type_t>(cos(angle));

   cTensor_PE.SetElement(0,0,pos,2*i,sin_val);
   cTensor_PE.SetElement(0,0,pos,2*i+1,cos_val);
  }
  if ((D&1)==1)
  {
   //для нечётного D — заполняем последний элемент sin'ом с отдельной частотой
   double freq=exp(-log_base*(2.0*(D/2))/(double)D);
   double angle=(double)pos*freq;
   cTensor_PE.SetElement(0,0,pos,D-1,static_cast<type_t>(sin(angle)));
  }
 }

 //далее размазываем PE по всем элементам батча (PE не зависит от элемента батча)
 for(uint32_t w=0;w<BatchSize;w++)
 {
  for(uint32_t pos=0;pos<TSize;pos++)
  {
   for(uint32_t x=0;x<D;x++)
   {
    cTensor_PE_Batch.SetElement(w,0,pos,x,cTensor_PE.GetElement(0,0,pos,x));
   }
  }
 }
}

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
void CNetLayerPositionalEncoding<type_t>::Create(INetLayer<type_t> *prev_layer_ptr,uint32_t batch_size)
{
 PrevLayerPtr=prev_layer_ptr;
 NextLayerPtr=NULL;
 BatchSize=batch_size;

 if (PrevLayerPtr==NULL) throw("Слой позиционного кодирования не может быть входным!");//слой без предшествующего считается входным
 //запомним размеры входного тензора, чтобы потом всегда к ним приводить
 InputSize_X=PrevLayerPtr->GetOutputTensor().GetSizeX();
 InputSize_Y=PrevLayerPtr->GetOutputTensor().GetSizeY();
 InputSize_Z=PrevLayerPtr->GetOutputTensor().GetSizeZ();
 if (InputSize_Z!=1) throw("Входной тензор слоя позиционного кодирования должен быть единичного размера по Z!");

 TSize=InputSize_Y;
 DSize=InputSize_X;

 NextLayerPtr=NULL;

 //размер выходного тензора
 OutputSize_X=InputSize_X;
 OutputSize_Y=InputSize_Y;
 OutputSize_Z=1;

 //создаём тензор позиционного кодирования
 cTensor_PE=CTensor<type_t>(1,1,TSize,DSize);
 cTensor_PE_Batch=CTensor<type_t>(BatchSize,1,TSize,DSize);
 //создаём выходной тензор
 cTensor_H=CTensor<type_t>(BatchSize,1,TSize,DSize);

 //задаём предшествующему слою, что мы его последующий слой
 PrevLayerPtr->SetNextLayerPtr(this);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить инициализацию слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::Reset(type_t scale)
{
 (void)scale;//PE — детерминированный, scale не используется
 //строим и заполняем тензоры PE один раз при инициализации
 BuildPositionalEncoding();
}
//----------------------------------------------------------------------------------------------------
/*!задать выход слоя
\param[in] output Матрица задаваемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::SetOutput(CTensor<type_t> &output)
{
 cTensor_H.RestoreSize();
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerPositionalEncoding<type_t>::SetOutput - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerPositionalEncoding<type_t>::SetOutput - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerPositionalEncoding<type_t>::SetOutput - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerPositionalEncoding<type_t>::SetOutput - ошибка размерности тензора output!");
 cTensor_H=output;
}
//----------------------------------------------------------------------------------------------------
/*!получить выход слоя
\param[out] output Матрица возвращаемых выходных значений (H)
\return Ничего не возвращается
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::GetOutput(CTensor<type_t> &output)
{
 cTensor_H.RestoreSize();
 if (output.GetSizeX()!=cTensor_H.GetSizeX()) throw("void CNetLayerPositionalEncoding<type_t>::GetOutput - ошибка размерности тензора output!");
 if (output.GetSizeY()!=cTensor_H.GetSizeY()) throw("void CNetLayerPositionalEncoding<type_t>::GetOutput - ошибка размерности тензора output!");
 if (output.GetSizeZ()!=cTensor_H.GetSizeZ()) throw("void CNetLayerPositionalEncoding<type_t>::GetOutput - ошибка размерности тензора output!");
 if (output.GetSizeW()!=cTensor_H.GetSizeW()) throw("void CNetLayerPositionalEncoding<type_t>::GetOutput - ошибка размерности тензора output!");
 output=cTensor_H;
}
//----------------------------------------------------------------------------------------------------
///!выполнить прямой проход по слою
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::Forward(void)
{
 cTensor_H.RestoreSize();

 //выход = вход + PE: H = 1.0 * X + 1.0 * PE_Batch
 //используем PE_Batch (размазанный по батчу), чтобы не зависеть от поддержки broadcasting в CTensorMath::Add
 CTensorMath<type_t>::Add(cTensor_H,PrevLayerPtr->GetOutputTensor(),cTensor_PE_Batch,1.0,1.0);
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на выходной тензор
\return Ссылка на матрицу выхода слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerPositionalEncoding<type_t>::GetOutputTensor(void)
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
void CNetLayerPositionalEncoding<type_t>::SetNextLayerPtr(INetLayer<type_t> *next_layer_ptr)
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
bool CNetLayerPositionalEncoding<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 //PE — детерминированный, не сохраняем (восстанавливается из формулы при Reset/Load)
 (void)iDataStream_Ptr;
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerPositionalEncoding<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 (void)iDataStream_Ptr;
 (void)check_size;
 //PE — детерминированный, не загружаем, а заново строим
 BuildPositionalEncoding();
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerPositionalEncoding<type_t>::SaveTrainingParam(IDataStream *iDataStream_Ptr)
{
 (void)iDataStream_Ptr;
 //обучаемых параметров нет
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоя
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerPositionalEncoding<type_t>::LoadTrainingParam(IDataStream *iDataStream_Ptr)
{
 (void)iDataStream_Ptr;
 //обучаемых параметров нет
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!начать процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::TrainingStart(void)
{
 cTensor_H.RestoreSize();
 //дельта такого же размера, как выход
 cTensor_Delta=CTensor<type_t>(BatchSize,1,TSize,DSize);
 //PE уже построен в Reset(), но на всякий случай перестрахуемся
 BuildPositionalEncoding();
}
//----------------------------------------------------------------------------------------------------
/*!завершить процесс обучения
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::TrainingStop(void)
{
 //освобождаем временный тензор дельты
 cTensor_Delta=CTensor<type_t>(1,1,1,1);
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обратный проход по сети для обучения

Так как PE — константа, градиент по входу равен градиенту по выходу:
   dX = dH
Градиент тривиально передаётся предыдущему слою.
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::TrainingBackward(bool create_delta_weight)
{
 (void)create_delta_weight;//весов нет — нечего накапливать

 //градиент предыдущего слоя = градиент выхода этого слоя
 CTensor<type_t> &prev_delta=PrevLayerPtr->GetDeltaTensor();

 //prev_delta = 0 * prev_delta + 1 * cTensor_Delta = cTensor_Delta
 //(перезаписываем, не накапливаем, так как для pass-through слоя предыдущий delta
 // уже был сброшен внешним циклом обучения или будет аккумулирован корректно
 // последующими слоями)
 CTensorMath<type_t>::Add(prev_delta,prev_delta,cTensor_Delta,static_cast<type_t>(0.0),static_cast<type_t>(1.0));
}
//----------------------------------------------------------------------------------------------------
/*!сбросить поправки к весам
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::TrainingResetDeltaWeight(void)
{
 //весов нет — сбрасывать нечего
}
//----------------------------------------------------------------------------------------------------
/*!выполнить обновления весов
\param[in] speed Скорость обучения
\param[in] iteration Номер итерации (для Adam)
\param[in] batch_scale Масштаб батча
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::TrainingUpdateWeight(double speed,double iteration,double batch_scale)
{
 (void)speed;
 (void)iteration;
 (void)batch_scale;
 //весов нет — обновлять нечего
}
//----------------------------------------------------------------------------------------------------
/*!получить ссылку на тензор дельты слоя
\return Ссылка на тензор дельты слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CNetLayerPositionalEncoding<type_t>::GetDeltaTensor(void)
{
 return(cTensor_Delta);
}
//----------------------------------------------------------------------------------------------------
/*!ограничить веса в диапазон
\param[in] min Минимальное значение веса
\param[in] max Максимальное значение веса
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::ClipWeight(type_t min,type_t max)
{
 (void)min;
 (void)max;
 //весов нет — ограничивать нечего
}
//----------------------------------------------------------------------------------------------------
/*!задать временной шаг
\param[in] index Индекс элемента пакета
\param[in] time_step Временной шаг
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::SetTimeStep(uint32_t index,uint32_t time_step)
{
 (void)index;
 (void)time_step;
 //PE не зависит от времени — игнорируем
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность входного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::PrintInputTensorSize(const std::string &name)
{
 if (PrevLayerPtr!=NULL) PrevLayerPtr->GetOutputTensor().Print(name+" PositionalEncoding: input ",false);
}
//----------------------------------------------------------------------------------------------------
/*!вывести размерность выходного тензора слоя
\param[in] name Название слоя
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::PrintOutputTensorSize(const std::string &name)
{
 cTensor_H.RestoreSize();
 GetOutputTensor().Print(name+" PositionalEncoding: output",false);
}
//----------------------------------------------------------------------------------------------------
/*!<разрешить/запретить использование усреднённых весов
\param[in] state - разрешить запретить использование усреднённых весов
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::EnableEMA(bool state)
{
 EMAEnabled=state;
}
//----------------------------------------------------------------------------------------------------
/*!загрузить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerPositionalEncoding<type_t>::LoadEMAWeight(IDataStream *iDataStream_Ptr,bool check_size)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!сохранить усреднённые веса
\param[in] iDataStream_Ptr Указатель на класс ввода-вывода
\return Успех операции
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CNetLayerPositionalEncoding<type_t>::SaveEMAWeight(IDataStream *iDataStream_Ptr)
{
 return(true);
}
//----------------------------------------------------------------------------------------------------
/*!включить режим вывода
\param[in] state Состояние режима вывода
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CNetLayerPositionalEncoding<type_t>::SetInferenceMode(bool state)
{
 INetLayer<type_t>::SetInferenceMode(state);
}

#endif
