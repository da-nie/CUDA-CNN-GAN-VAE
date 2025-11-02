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

  using CModelBasicVAE<type_t>::ITERATION_OF_SAVE_IMAGE;
  using CModelBasicVAE<type_t>::ITERATION_OF_SAVE_NET;
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

  void CreateCoder_1(void);//создать сеть кодировщика
  void CreateDecoder_1(void);//создать сеть декодировщика

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
 IMAGE_HEIGHT=128;
 IMAGE_WIDTH=128;
 IMAGE_DEPTH=3;
 NOISE_LAYER_SIDE_X=32;
 NOISE_LAYER_SIDE_Y=32;
 NOISE_LAYER_SIDE_Z=1;
 NOISE_LAYER_SIZE=NOISE_LAYER_SIDE_X*NOISE_LAYER_SIDE_Y*NOISE_LAYER_SIDE_Z;

 SPEED=0.0001;

 BATCH_SIZE=64;

 ITERATION_OF_SAVE_IMAGE=1;
 ITERATION_OF_SAVE_NET=10;
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
 CreateCoder_1();
}
//----------------------------------------------------------------------------------------------------
//создать сеть декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateDecoder(void)
{
 CreateDecoder_1();
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::TrainingNet(bool mnist)
{
 CModelBasicVAE<type_t>::TrainingNet(mnist);
}



//----------------------------------------------------------------------------------------------------
//создать сеть кодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateCoder_1(void)
{
 CoderNet.clear();

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);


 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 //делаем разделение для переноса
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 uint32_t log_var_layer=CoderNet.size()-1;
 //слой мю
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 uint32_t mu_layer=CoderNet.size()-1;
 //слой log_var
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,CoderNet[log_var_layer].get(),BATCH_SIZE)));
 //объединяем на выходном слое кодера VAE
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerVAECoderOutput<type_t>(CoderNet[mu_layer].get(),CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor final",false);
}
//----------------------------------------------------------------------------------------------------
//создать сеть декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateDecoder_1(void)
{
 DecoderNet.clear();

 uint32_t convert=CoderNet.size()-1;
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(512*4*4,CoderNet[convert].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 convert=DecoderNet.size()-1;
 DecoderNet[convert]->GetOutputTensor().ReinterpretSize(BATCH_SIZE,512,4,4);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));

 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor final",false);

 DecoderNet[convert]->GetOutputTensor().RestoreSize();
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************



#endif
