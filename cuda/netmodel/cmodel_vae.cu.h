#ifndef C_MODEL_VAE_H
#define C_MODEL_VAE_H

//****************************************************************************************************
//Модель VAE
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "cmodel_basicvae.cu.h"

//****************************************************************************************************
//Модель VAE
//****************************************************************************************************
template<class type_t>
class CModelVAE:public CModelBasicVAE<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-структуры------------------------------------------------------------------------------------------
  //-переменные-----------------------------------------------------------------------------------------

  using CModelBasicVAE<type_t>::BATCH_AMOUNT;
  using CModelBasicVAE<type_t>::BATCH_SIZE;

  using CModelBasicVAE<type_t>::IMAGE_WIDTH;
  using CModelBasicVAE<type_t>::IMAGE_HEIGHT;
  using CModelBasicVAE<type_t>::IMAGE_DEPTH;
  using CModelBasicVAE<type_t>::NOISE_LAYER_SIDE_X;
  using CModelBasicVAE<type_t>::NOISE_LAYER_SIDE_Y;
  using CModelBasicVAE<type_t>::NOISE_LAYER_SIDE_Z;
  using CModelBasicVAE<type_t>::NOISE_LAYER_SIZE;

  using CModelBasicVAE<type_t>::SPEED;

  using CModelBasicVAE<type_t>::CoderNet;
  using CModelBasicVAE<type_t>::DecoderNet;

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelVAE(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelVAE();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  void CreateCoder(void) override;//создать сеть кодировщика
  void CreateDecoder(void) override;//создать сеть декодировщика
  void TrainingNet(bool mnist) override;//запуск обучения нейросети
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelVAE<type_t>::CModelVAE(void)
{
 IMAGE_HEIGHT=224;//192;
 IMAGE_WIDTH=224;//256;
 IMAGE_DEPTH=3;
 NOISE_LAYER_SIDE_X=2;
 NOISE_LAYER_SIDE_Y=2;
 NOISE_LAYER_SIDE_Z=IMAGE_DEPTH;
 NOISE_LAYER_SIZE=NOISE_LAYER_SIDE_X*NOISE_LAYER_SIDE_Y*NOISE_LAYER_SIDE_Z;

 SPEED=0.01;

 BATCH_SIZE=20;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelVAE<type_t>::~CModelVAE()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//создать сеть кодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateCoder(void)
{
 CoderNet.clear();

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH)));

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,8,1,1,0,0,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,CoderNet[CoderNet.size()-1].get())));

 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,1,1,0,0,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,CoderNet[CoderNet.size()-1].get())));

 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,0,0,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,CoderNet[CoderNet.size()-1].get())));

 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,CoderNet[CoderNet.size()-1].get())));

 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*16,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get())));

 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,CoderNet[CoderNet.size()-1].get())));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_SIGMOID,CoderNet[CoderNet.size()-1].get())));

 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor final",false);
}
//----------------------------------------------------------------------------------------------------
//создать сеть декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateDecoder(void)
{
 DecoderNet.clear();

 size_t convert=CoderNet.size()-1;
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,CoderNet[convert].get())));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get())));

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*16,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get())));

 convert=DecoderNet.size()-1;
 DecoderNet[convert]->GetOutputTensor().ReinterpretSize(NOISE_LAYER_SIDE_Z,NOISE_LAYER_SIDE_Y*4,NOISE_LAYER_SIDE_X*4);
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet[convert]->GetOutputTensor().RestoreSize();
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(7,64,1,1,0,0,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(7,32,1,1,0,0,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(9,16,1,1,0,0,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(7,8,1,1,0,0,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,IMAGE_DEPTH,1,1,0,0,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_TANGENCE,DecoderNet[DecoderNet.size()-1].get())));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor final",false);
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::TrainingNet(bool mnist)
{
 CModelBasicVAE<type_t>::TrainingNet(mnist);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************



#endif
