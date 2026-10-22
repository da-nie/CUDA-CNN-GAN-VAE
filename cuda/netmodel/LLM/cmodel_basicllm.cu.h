#ifndef C_MODEL_BASIC_LLM_H
#define C_MODEL_BASIC_LLM_H

//****************************************************************************************************
//Класс-основа для сетей LLM
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <math.h>
#include <random>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../../../system/system.h"
#include "../../ctimestamp.cu.h"

#include "../../../netlayer/cnetlayerfunction.cu.h"
#include "../../../netlayer/cnetlayerlinear.cu.h"
#include "../../../netlayer/cnetlayerdropout.cu.h"
#include "../../../netlayer/cnetlayerbatchnormalization.cu.h"
#include "../../../netlayer/cnetlayergroupnormalization.cu.h"
#include "../../../netlayer/cnetlayerlayernormalization.cu.h"
#include "../../../netlayer/cnetlayerlayernormalizationgpt2.cu.h"
#include "../../../netlayer/cnetlayerconvolution.cu.h"
#include "../../../netlayer/cnetlayerconvolutioninput.cu.h"
#include "../../../netlayer/cnetlayerbackconvolution.cu.h"
#include "../../../netlayer/cnetlayermaxpooling.cu.h"
#include "../../../netlayer/cnetlayermaxdepooling.cu.h"
#include "../../../netlayer/cnetlayerupsampling.cu.h"
#include "../../../netlayer/cnetlayeraveragepooling.cu.h"
#include "../../../netlayer/cnetlayertimeembedding.cu.h"
#include "../../../netlayer/cnetlayertimeembeddingmlp.cu.h"
#include "../../../netlayer/cnetlayersplitter.cu.h"
#include "../../../netlayer/cnetlayerconcatenator.cu.h"
#include "../../../netlayer/cnetlayervaecoderoutput.cu.h"
#include "../../../netlayer/cnetlayersoftmax.cu.h"
#include "../../../netlayer/cnetlayerselfattention.cu.h"
#include "../../../netlayer/cnetlayeradder.cu.h"
#include "../../../netlayer/cnetlayerview.cu.h"

#include "cgpt2tokenizer.cu.h"

//****************************************************************************************************
//Класс-основа для сетей VAE
//****************************************************************************************************
template<class type_t>
class CModelBasicLLM
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
  const int32_t MAX_SEQ_LEN=1024; //Максимальная длина контекста архитектуры (менять нельзя, ограничено wpe_)
  const int32_t MODEL_SCALE=1;//0 - 124M, 1 - 355M, 2 - 774M, 3 - 1558M
  const int32_t ARCH_NHEAD=(MODEL_SCALE==3)?25:(12+4*MODEL_SCALE);
  const int32_t HEAD_DIM=64;
  const int32_t ARCH_DIM=ARCH_NHEAD*HEAD_DIM;   //768
  const int32_t ARCH_NLAYER=12+MODEL_SCALE*12;  //12
  const int32_t MLP_DIM=4*ARCH_DIM;             //3072
  const int32_t VOCAB_SIZE=50257;

 protected:
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
  static const uint32_t STRING_BUFFER_SIZE=1024;///<размер буфера строки
  //-переменные-----------------------------------------------------------------------------------------
  uint32_t BATCH_AMOUNT;///<количество пакетов
  uint32_t BATCH_SIZE;///<размер пакета

  double SPEED;///<скорость обучения

  uint32_t Iteration;///<итерация
  std::vector<std::shared_ptr<INetLayer<type_t> > > LLMNet;///<сеть LLM

  CGPT2Tokenizer cGPT2Tokenizer;///<токенизатор GPT2

  CTensor<type_t> wpe_;
  CTensor<type_t> wte_;
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelBasicLLM(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelBasicLLM();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Execute(void);///<выполнить
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  virtual void CreateLLMNet(void)=0;///<создать сеть
  void LoadNet(void);///<загрузить сети
  void SaveNet(void);///<сохранить сети
  void LoadTrainingParam(void);///<загрузить параметры обучения
  void SaveTrainingParam(void);///<сохранить параметры обучения
  void SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,bool backward=false);///<сохранить слои сети
  void LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,bool backward=false);///<загрузить слои сети
  void SaveNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t iteration,bool backward=false);///<сохранить параметры обучения слоёв сети
  void LoadNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t &iteration,bool backward=false);///<загрузить параметры обучения слоёв сети
  void TrainingLLMNet(uint32_t mini_batch_index,double &cost);///<обучение сети LLM
  void Training(void);///<обучение нейросети
   int32_t SampleFromTopK(int32_t top_k_indices[],float top_k_values[],int32_t top_k);///<сэмплирование токена из топ-K
  void ApplyRepetitionPenalty(CTensor<type_t>& cTensor_Logits,const std::vector<int32_t>& vHistoryTokens,int32_t num_total_tokens);///<применение штрафа за повторение токенов
  void SelectTopK(const CTensor<type_t>& cTensor_Logits,int32_t top_k_indices[],float top_k_values[],int32_t top_k);///<выбор топ-K токенов
  void InitializeTensorFromTokens(CTensor<type_t> &cTensor_Embedding,const std::vector<int32_t>& input_tokens,const CTensor<type_t>& cTensor_WTE,const CTensor<type_t>& cTensor_WPE,int32_t dim);///<инициализация тензора из токенов
  bool GenerateAnswer(std::vector<int32_t>& history_tokens,int32_t max_context_len,CGPT2Tokenizer &cGPT2Tokenizer);///<получение ответа
  virtual void TrainingNet(bool mnist);///<запуск обучения нейросети
  void InitLLM(void);///<инициализация параметров LLM
};

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicLLM<type_t>::CModelBasicLLM(void)
{
 Iteration=0;
 BATCH_SIZE=1;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicLLM<type_t>::~CModelBasicLLM()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************
//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::LoadNet(void)
{
 FILE *file=fopen("llm_neuronet_355.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("llm_neuronet_355.net",false));
  LoadNetLayers(iDataStream_Disc_Ptr.get(),LLMNet);
  SYSTEM::PutMessageToConsole("Сеть LLM модели загружена.");
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::SaveNet(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("llm_neuronet_355.net",true));
 SaveNetLayers(iDataStream_Disc_Ptr.get(),LLMNet);
}
//----------------------------------------------------------------------------------------------------
//загрузить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::LoadTrainingParam(void)
{
 FILE *file=fopen("llm_training_param.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("llm_training_param.net",false));
  LoadNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),LLMNet,Iteration);
  SYSTEM::PutMessageToConsole("Параметры обучения LLM сети загружены.");
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::SaveTrainingParam(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("llm_training_param.net",true));
 SaveNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),LLMNet,Iteration);
}


//----------------------------------------------------------------------------------------------------
/*!сохранить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::SaveNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,bool backward)
{
 if (backward==false) for(uint32_t n=0;n<net.size();n++) net[n]->Save(iDataStream_Ptr);
                 else for(uint32_t n=net.size();n>0;n--) net[n-1]->Save(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить слои сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::LoadNetLayers(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,bool backward)
{
 if (backward==false)
 {
  for(uint32_t n=0;n<net.size();n++)
  {
   net[n]->PrintInputTensorSize("Loading...");
   if (net[n]->IsMark()==true) continue;
   net[n]->Load(iDataStream_Ptr,true);
  }
 }
 else
 {
  for(uint32_t n=net.size();n>0;n--)
  {
   if (net[n-1]->IsMark()==true) continue;
   net[n-1]->Load(iDataStream_Ptr,true);
  }
 }
}
//----------------------------------------------------------------------------------------------------
/*!сохранить параметры обучения слоёв сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::SaveNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t iteration,bool backward)
{
 iDataStream_Ptr->SaveUInt32(iteration);
 if (backward==false) for(uint32_t n=0;n<net.size();n++) net[n]->SaveTrainingParam(iDataStream_Ptr);
                 else for(uint32_t n=net.size();n>0;n--) net[n-1]->SaveTrainingParam(iDataStream_Ptr);
}
//----------------------------------------------------------------------------------------------------
/*!загрузить параметры обучения слоёв сети
*/
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::LoadNetLayersTrainingParam(IDataStream *iDataStream_Ptr,std::vector<std::shared_ptr<INetLayer<type_t> > > &net,uint32_t &iteration,bool backward)
{
 iteration=iDataStream_Ptr->LoadUInt32();
 if (backward==false)
 {
  for(uint32_t n=0;n<net.size();n++)
  {
   if (net[n]->IsMark()==true) continue;
   net[n]->LoadTrainingParam(iDataStream_Ptr);
  }
 }
 else
 {
  for(uint32_t n=net.size();n>0;n--)
  {
   if (net[n-1]->IsMark()==true) continue;
   net[n-1]->LoadTrainingParam(iDataStream_Ptr);
  }
 }
}

//----------------------------------------------------------------------------------------------------
//обучение LLM сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::TrainingLLMNet(uint32_t mini_batch_index,double &cost)
{
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::Training(void)
{
}


//----------------------------------------------------------------------------------------------------
//сэмплирование токена из топ-K
//----------------------------------------------------------------------------------------------------
template<class type_t>
int32_t CModelBasicLLM<type_t>::SampleFromTopK(int32_t top_k_indices[],float top_k_values[],int32_t top_k)
{
 const float SAMPLING_TEMP=0.01f;    //Температура сэмплирования

 float sum_exp=0.0f;
 for(int32_t i=0;i<top_k;i++)
 {
  top_k_values[i]=expf((top_k_values[i]-top_k_values[0])/SAMPLING_TEMP);
  sum_exp+=top_k_values[i];
 }

 for(int32_t i=0;i<top_k;i++) top_k_values[i]/=sum_exp;

 float r=static_cast<type_t>(rand())/RAND_MAX;
 float cumulative_prob=0.0f;
 int32_t generated_token_id=top_k_indices[0];

 for(int32_t i=0;i<top_k;i++)
 {
  cumulative_prob+=top_k_values[i];
  if (cumulative_prob>r)
  {
   generated_token_id=top_k_indices[i];
   break;
  }
 }
 return(generated_token_id);
}

//----------------------------------------------------------------------------------------------------
//применение штрафа за повторение токенов
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::ApplyRepetitionPenalty(CTensor<type_t>& cTensor_Logits,const std::vector<int32_t>& vHistoryTokens,int32_t num_total_tokens)
{
 const int32_t REPETITION_LOOKBACK=50; //Глубина просмотра для штрафа повторений
 const float REPETITION_PENALTY=1.15f; //Штраф за повторение токенов

 int32_t start_idx=0;
 if (num_total_tokens>REPETITION_LOOKBACK) start_idx=num_total_tokens-REPETITION_LOOKBACK;
 for(int32_t i=start_idx;i<num_total_tokens;i++)
 {
  int32_t token_id=vHistoryTokens[i];
  if (token_id>=0 && token_id<VOCAB_SIZE)
  {
   float val=cTensor_Logits.GetElement(0,0,0,token_id);
   if (val>0) cTensor_Logits.SetElement(0,0,0,token_id,val/REPETITION_PENALTY);
         else cTensor_Logits.SetElement(0,0,0,token_id,val*REPETITION_PENALTY);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//выбор топ-K токенов
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::SelectTopK(const CTensor<type_t>& cTensor_Logits,int32_t top_k_indices[],float top_k_values[],int32_t top_k)
{
 for(int32_t i=0;i<top_k;i++)
 {
  top_k_values[i]=-1e9;
  top_k_indices[i]=0;
 }
 for(int32_t i=0;i<VOCAB_SIZE;i++)
 {
  float val=cTensor_Logits.GetElement(0,0,0,i);
  if (val>top_k_values[top_k-1])
  {
   top_k_values[top_k-1]=val;
   top_k_indices[top_k-1]=i;
   for(int32_t j=top_k-1;j>0;j--)
   {
    if (top_k_values[j]>top_k_values[j-1])
    {
     std::swap(top_k_values[j],top_k_values[j-1]);
     std::swap(top_k_indices[j],top_k_indices[j-1]);
    }
    else break;
   }
  }
 }
}


//----------------------------------------------------------------------------------------------------
//инициализация тензора из токенов
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::InitializeTensorFromTokens(CTensor<type_t> &cTensor_Embedding,const std::vector<int32_t>& input_tokens,const CTensor<type_t>& cTensor_WTE,const CTensor<type_t>& cTensor_WPE,int32_t dim)
{
 int32_t num_total_tokens=input_tokens.size();
 int32_t align_token_size=MAX_SEQ_LEN;//num_total_tokens+CONTEXT_ALIGN-(num_total_tokens%CONTEXT_ALIGN);//дополняем количество токенов до выровненного по группам 32 токена

 cTensor_Embedding=CTensor<type_t>(1,1,align_token_size,dim);
 CTensorMath<type_t>::Fill(cTensor_Embedding,0);//очищаем матрицу токенов

 for(int32_t y=0;y<num_total_tokens;y++)
 {
  int32_t token=input_tokens[y];
  for(int32_t x=0;x<dim;x++)//раскладываем токен на вектор
  {
   float val=cTensor_WTE.GetElement(0,0,token,x);//значение вектора токенов
   val+=cTensor_WPE.GetElement(0,0,y,x);//позиция токена
   cTensor_Embedding.SetElement(0,0,y,x,val);
  }
 }
}


//----------------------------------------------------------------------------------------------------
//получение ответа
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CModelBasicLLM<type_t>::GenerateAnswer(std::vector<int32_t>& history_tokens,int32_t max_context_len,CGPT2Tokenizer &cGPT2Tokenizer)
{
 if (history_tokens.size()==0) return(false);

 const int32_t TOP_K=10;            //K для Top-K сэмплирования
 CTensor<type_t> cTensor_Embedding;
 CCUDATimeSpent cCUDATimeSpent;
 char str_b[STRING_BUFFER_SIZE];
 while(1)
 {
  //инициализируем тензор эмбеддингов
  InitializeTensorFromTokens(cTensor_Embedding,history_tokens,wte_,wpe_,ARCH_DIM);

  int32_t num_total_tokens=history_tokens.size();

  //подаём на вход сети
  LLMNet[0]->GetOutputTensor()=cTensor_Embedding;
  //вычисляем сеть

  for(uint32_t layer=0;layer<LLMNet.size();layer++)
  {
   //cCUDATimeSpent.Start();
   LLMNet[layer]->Forward();
   //float gpu_time=cCUDATimeSpent.Stop();
   //sprintf(str_b,"На слой %i ушло:%.2f мс.",layer,gpu_time);
   //LLMNet[layer]->PrintInputTensorSize(str_b);
  }

  //throw("Стоп");

  CTensor<type_t> &cTensor_FinalLN=LLMNet.back()->GetOutputTensor();
  CTensor<type_t> cTensor_LastToken(1,1,1,ARCH_DIM);

  for(uint32_t x=0;x<ARCH_DIM;x++) cTensor_LastToken.SetElement(0,0,0,x,cTensor_FinalLN.GetElement(0,0,num_total_tokens-1,x));

  //вычисляем логиты
  CTensor<type_t> cTensor_Logits(1,1,1,VOCAB_SIZE);
  CTensorMath<type_t>::RightTransposeMul(cTensor_Logits,cTensor_LastToken,wte_);
/*
  CTensor<type_t> &cTensor_Logits=LLMNet.back()->GetOutputTensor();
  cTensor_Logits.Print("Logits",false);
  throw("Стоп");
  */

  //применяем штраф за повторение
  ApplyRepetitionPenalty(cTensor_Logits,history_tokens,num_total_tokens);

  //выбираем топ-K токенов
  int32_t top_k_indices[TOP_K];
  float top_k_values[TOP_K];
  SelectTopK(cTensor_Logits,top_k_indices,top_k_values,TOP_K);

  //сэмплируем токен из топ-K
  int32_t generated_token_id=SampleFromTopK(top_k_indices,top_k_values,TOP_K);
  if (generated_token_id==(VOCAB_SIZE-1)) return(false);

  if (num_total_tokens>=max_context_len)
  {
   int32_t half_length=max_context_len/2;
   std::rotate(history_tokens.begin(),history_tokens.begin()+half_length,history_tokens.end());
   history_tokens.resize(half_length);
   num_total_tokens=half_length;
  }

  history_tokens.push_back(generated_token_id);

  //декодируем сгенерированный токен в читаемый текст
  std::vector<int32_t> tmp(1);
  tmp[0]=generated_token_id;
  std::string decoded=cGPT2Tokenizer.Decode(tmp);
  //выводим без дополнительного форматирования
  printf("%s",decoded.c_str());
  fflush(stdout);

  //проверяем, есть ли перевод строки в декодированном тексте
  if (decoded.find('\n') != std::string::npos) return(true);
 }
 return(false);
}


//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicLLM<type_t>::TrainingNet(bool mnist)
{
 char str[STRING_BUFFER_SIZE];

 if (cGPT2Tokenizer.Load("encoder.json","vocab.bpe")==false) throw("Не удалось загрузить encoder.json или vocab.bpe!");
 cGPT2Tokenizer.Test();

 CreateLLMNet();

 for(uint32_t n=0;n<LLMNet.size();n++)
 {
  LLMNet[n]->Reset();
 }

 LoadNet();//одновременно будут загружены средние веса

 //загружаем таблицы WTE и WPE (временное и пространственное кодирование)
 {
  std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile("wpe.bin",false));
  if (iDataStream_Ptr.get()==NULL) throw("Не удалось загрузить wpe.bin!");
  wpe_.Load(iDataStream_Ptr.get());
 }

 {
  std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile("wte.bin",false));
  if (iDataStream_Ptr.get()==NULL) throw("Не удалось загрузить wte.bin!");
  wte_.Load(iDataStream_Ptr.get());
 }

 std::vector<int32_t> history_tokens;
 history_tokens.reserve(MAX_SEQ_LEN);

 std::string initial="";
 initial+="\nAlice: Hello, who are you?";
 initial+="\nBob: I am Bob. I am an AI assistant.";
 initial+="\nAlice: What is your name?";
 initial+="\nBob: My name is Bob.";
 initial+="\nAlice: Are you Alice?";
 initial+="\nBob: No, I am Bob. I am not Alice.";
 initial+="\nAlice: Okay Bob, how can you help me?";
 initial+="\nBob: I can answer your questions. Ask me anything.";

 history_tokens=cGPT2Tokenizer.Encode(initial.c_str());
 int32_t num_total_tokens=history_tokens.size();

 printf("Загружено %d токенов в историю\n", num_total_tokens);
 printf("Bob: I can answer your questions. Ask me anything.\n");

 bool bob_ended_with_newline=false;

 while(1)
 {
  char buf[512]={0};
  if (bob_ended_with_newline) strcat(buf,"Alice: ");
                         else strcat(buf,"\nAlice: ");
  printf(bob_ended_with_newline?"\nAlice: ":"\nAlice: ");//TODO:непонятно, в чём различие?
  fflush(stdout);

  if (fgets(buf+strlen(buf),sizeof(buf)-strlen(buf),stdin)==NULL) break;

  int32_t len=static_cast<int32_t>(strlen(buf));
  if (len>0 && buf[len-1]=='\n') buf[len-1]='\0';

  strcat(buf,"\nBob: ");
  //выполняем токенизацию нового буфера слов
  std::vector<int32_t> tokens=cGPT2Tokenizer.Encode(buf);
  //добавляем к уже имеющимся токенам новые
  for(size_t n=0;n<tokens.size();n++) history_tokens.push_back(tokens[n]);

  printf("Bob: ");
  fflush(stdout);

  //получаем ответ
  bob_ended_with_newline=GenerateAnswer(history_tokens,MAX_SEQ_LEN,cGPT2Tokenizer);
 }
 /*

 //включаем обучение
 for(uint32_t n=0;n<LLMNet.size();n++)
 {
  LLMNet[n]->SetInferenceMode(false);
  LLMNet[n]->TrainingModeAdam(0.9,0.999);
  LLMNet[n]->TrainingStart();
 }

 //загружаем параметры обучения
 LoadTrainingParam();
 //запускаем обучение
 Training();
 //отключаем обучение
 for(uint32_t n=0;n<LLMNet.size();n++) LLMNet[n]->TrainingStop();

 SaveNet();
 */
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelBasicLLM<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);
 TrainingNet(true);
}

#endif
