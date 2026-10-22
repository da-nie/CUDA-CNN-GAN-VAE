#ifndef C_MODEL_LLM_H
#define C_MODEL_LLM_H

//****************************************************************************************************
//Модель диффузионной сети
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "cmodel_basicllm.cu.h"

//****************************************************************************************************
//Модель диффузионной сети
//****************************************************************************************************
template<class type_t>
class CModelLLM:public CModelBasicLLM<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-структуры------------------------------------------------------------------------------------------
  //-переменные-----------------------------------------------------------------------------------------
  using CModelBasicLLM<type_t>::BATCH_AMOUNT;
  using CModelBasicLLM<type_t>::BATCH_SIZE;
  using CModelBasicLLM<type_t>::SPEED;

  using CModelBasicLLM<type_t>::LLMNet;

  using CModelBasicLLM<type_t>::MODEL_SCALE;//0 - 124M, 1 - 355M, 2 - 774M, 3 - 1558M
  using CModelBasicLLM<type_t>::ARCH_NHEAD;
  using CModelBasicLLM<type_t>::HEAD_DIM;
  using CModelBasicLLM<type_t>::ARCH_DIM;   //768
  using CModelBasicLLM<type_t>::ARCH_NLAYER;  //12
  using CModelBasicLLM<type_t>::MAX_SEQ_LEN;
  using CModelBasicLLM<type_t>::MLP_DIM;             //3072
  using CModelBasicLLM<type_t>::VOCAB_SIZE;
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelLLM(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelLLM();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  void CreateLLMNet(void) override;///<создать диффузионную сеть
  void TrainingNet(bool mnist) override;///<запуск обучения нейросети
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelLLM<type_t>::CModelLLM(void)
{
 SPEED=0.0003;
 BATCH_SIZE=1;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelLLM<type_t>::~CModelLLM()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
// создание сети LLM
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelLLM<type_t>::CreateLLMNet(void)
{
 LLMNet.clear();
 const double  DROP_OUT=0.1;                   //resid_pdrop GPT-2 (0 - отключить)

 //вход: ГОТОВЫЕ эмбеддинги wte[token]+wpe[pos], заполняются снаружи через SetOutput
 LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolutionInput<type_t>(1,MAX_SEQ_LEN,ARCH_DIM,BATCH_SIZE)));

 for(uint32_t layer=0;layer<ARCH_NLAYER;layer++)
 {
  //=== x = x + attn(ln_1(x)) ===
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerSplitter<type_t>(2,LLMNet.back().get(),BATCH_SIZE)));
  uint32_t residual_1=LLMNet.size()-1;

  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerLayerNormalizationGPT2<type_t>(LLMNet.back().get(),BATCH_SIZE)));//ln_1
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerSelfAttention<type_t>(ARCH_NHEAD,true,LLMNet.back().get(),BATCH_SIZE)));//c_attn (QKV+внимание)
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerLinear<type_t>(ARCH_DIM,LLMNet.back().get(),BATCH_SIZE)));//attn.c_proj: 768->768
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerDropOut<type_t>(DROP_OUT,LLMNet.back().get(),BATCH_SIZE)));
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerAdder<type_t>(LLMNet.back().get(),LLMNet[residual_1].get(),BATCH_SIZE)));

  //=== x = x + mlp(ln_2(x)) ===
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerSplitter<type_t>(2,LLMNet.back().get(),BATCH_SIZE)));
  uint32_t residual_2=LLMNet.size()-1;

  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerLayerNormalizationGPT2<type_t>(LLMNet.back().get(),BATCH_SIZE)));//ln_2
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerLinear<type_t>(MLP_DIM,LLMNet.back().get(),BATCH_SIZE)));//mlp.c_fc: 768->3072
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,LLMNet.back().get(),BATCH_SIZE)));
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerLinear<type_t>(ARCH_DIM,LLMNet.back().get(),BATCH_SIZE)));//mlp.c_proj: 3072->768
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerDropOut<type_t>(DROP_OUT,LLMNet.back().get(),BATCH_SIZE)));
  LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerAdder<type_t>(LLMNet.back().get(),LLMNet[residual_2].get(),BATCH_SIZE)));
 }

 //=== голова ===
 LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerLayerNormalizationGPT2<type_t>(LLMNet.back().get(),BATCH_SIZE)));//ln_f

 //LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerLinear<type_t>(VOCAB_SIZE,LLMNet.back().get(),BATCH_SIZE)));//lm_head: 768->50257
 //LLMNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerSoftMax<type_t>(LLMNet.back().get(),BATCH_SIZE)));//P(следующий токен)


 for(size_t n=0;n<LLMNet.size();n++)
 {
  LLMNet[n].get()->PrintInputTensorSize("LLMNet");
  LLMNet[n].get()->PrintOutputTensorSize("LLMNet");
  printf("\r\n");
 }
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelLLM<type_t>::TrainingNet(bool mnist)
{
 CModelBasicLLM<type_t>::TrainingNet(mnist);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************



#endif
