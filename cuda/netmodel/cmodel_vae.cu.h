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

  void CreateCoder_Kernel11(void);//создать сеть кодировщика
  void CreateDecoder_Kernel11(void);//создать сеть декодировщика

  void CreateCoder_Kernel5(void);//создать сеть кодировщика
  void CreateDecoder_Kernel5(void);//создать сеть декодировщика

  void CreateCoder_Kernel7(void);//создать сеть кодировщика
  void CreateDecoder_Kernel7(void);//создать сеть декодировщика

  void CreateCoder_Kernel3(void);//создать сеть кодировщика
  void CreateDecoder_Kernel3(void);//создать сеть декодировщика

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
 NOISE_LAYER_SIDE_X=16;
 NOISE_LAYER_SIDE_Y=16;
 NOISE_LAYER_SIDE_Z=8;
 NOISE_LAYER_SIZE=NOISE_LAYER_SIDE_X*NOISE_LAYER_SIDE_Y*NOISE_LAYER_SIDE_Z;

 SPEED=0.001;//0.00001;

 BATCH_SIZE=50;

 ITERATION_OF_SAVE_IMAGE=1;
 ITERATION_OF_SAVE_NET=1;
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
 CreateCoder_Kernel7();
}
//----------------------------------------------------------------------------------------------------
//создать сеть декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateDecoder(void)
{
 CreateDecoder_Kernel7();
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
void CModelVAE<type_t>::CreateCoder_Kernel11(void)
{
 CoderNet.clear();

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,11,2,2,5,5,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,11,2,2,5,5,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,11,2,2,5,5,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,11,2,2,5,5,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(4096,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_SIGMOID,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor final",false);
}
//----------------------------------------------------------------------------------------------------
//создать сеть декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateDecoder_Kernel11(void)
{
 DecoderNet.clear();

 uint32_t convert=CoderNet.size()-1;
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(4096,CoderNet[convert].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 convert=DecoderNet.size()-1;
 DecoderNet[convert]->GetOutputTensor().ReinterpretSize(64,8,8);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(11,64,3,3,8,8,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(11,64,3,3,12,12,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(11,64,3,3,20,20,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
// DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(11,64,3,3,36,36,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(11,IMAGE_DEPTH,1,1,5,5,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_TANGENCE,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));

 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor final",false);

 DecoderNet[convert]->GetOutputTensor().RestoreSize();
}

//----------------------------------------------------------------------------------------------------
//создать сеть кодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateCoder_Kernel5(void)
{
 CoderNet.clear();

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);


 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,2,2,2,2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(4096,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_SIGMOID,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor final",false);
}
//----------------------------------------------------------------------------------------------------
//создать сеть декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateDecoder_Kernel5(void)
{
 DecoderNet.clear();

 uint32_t convert=CoderNet.size()-1;
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(4096,CoderNet[convert].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 convert=DecoderNet.size()-1;
 DecoderNet[convert]->GetOutputTensor().ReinterpretSize(64,8,8);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,64,3,3,5,5,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,64,3,3,9,9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,64,3,3,17,17,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
// DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,64,3,3,33,33,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,IMAGE_DEPTH,1,1,2,2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_TANGENCE,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));

 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor final",false);

 DecoderNet[convert]->GetOutputTensor().RestoreSize();
}


//----------------------------------------------------------------------------------------------------
//создать сеть кодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateCoder_Kernel7(void)
{
 CoderNet.clear();

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,7,2,2,3,3,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);


 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,7,2,2,3,3,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,7,2,2,3,3,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,7,2,2,3,3,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(4096,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_SIGMOID,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor final",false);

}
//----------------------------------------------------------------------------------------------------
//создать сеть декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateDecoder_Kernel7(void)
{
 DecoderNet.clear();

 uint32_t convert=CoderNet.size()-1;
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(4096,CoderNet[convert].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 convert=DecoderNet.size()-1;
 DecoderNet[convert]->GetOutputTensor().ReinterpretSize(64,8,8);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(7,256,3,3,6,6,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(7,128,3,3,10,10,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(7,64,3,3,18,18,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
// DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(7,64,3,3,34,34,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(7,IMAGE_DEPTH,1,1,3,3,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_TANGENCE,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));

 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor final",false);

 DecoderNet[convert]->GetOutputTensor().RestoreSize();
}

//----------------------------------------------------------------------------------------------------
//создать сеть кодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateCoder_Kernel3(void)
{
 CoderNet.clear();

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,2,2,1,1,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);


 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,2,2,1,1,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,2,2,1,1,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,2,2,1,1,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(4096,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 //CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor",false);

 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_SIGMOID,CoderNet[CoderNet.size()-1].get(),BATCH_SIZE)));
 CoderNet[CoderNet.size()-1]->GetOutputTensor().Print("Coder output tensor final",false);
}
//----------------------------------------------------------------------------------------------------
//создать сеть декодировщика
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelVAE<type_t>::CreateDecoder_Kernel3(void)
{
 DecoderNet.clear();

 uint32_t convert=CoderNet.size()-1;
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(4096,CoderNet[convert].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 convert=DecoderNet.size()-1;
 DecoderNet[convert]->GetOutputTensor().ReinterpretSize(64,8,8);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,3,3,4,4,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,3,3,8,8,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,3,3,16,16,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
// DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,3,3,32,32,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_GELU,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 //DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.2,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor",false);

 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,IMAGE_DEPTH,1,1,1,1,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));
 DecoderNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_TANGENCE,DecoderNet[DecoderNet.size()-1].get(),BATCH_SIZE)));

 DecoderNet[DecoderNet.size()-1]->GetOutputTensor().Print("Decoder output tensor final",false);

 DecoderNet[convert]->GetOutputTensor().RestoreSize();
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************



#endif
