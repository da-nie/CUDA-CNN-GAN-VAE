#ifndef C_MODEL_SR_GAN_H
#define C_MODEL_SR_GAN_H

//****************************************************************************************************
//Модель SR-GAN
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "cmodel_basicsrgan.cu.h"

//****************************************************************************************************
//Модель SR-GAN
//****************************************************************************************************
template<class type_t>
class CModelSR_GAN:public CModelBasicSR_GAN<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-структуры------------------------------------------------------------------------------------------
  //-переменные-----------------------------------------------------------------------------------------

  using CModelBasicSR_GAN<type_t>::BATCH_AMOUNT;
  using CModelBasicSR_GAN<type_t>::BATCH_SIZE;

  using CModelBasicSR_GAN<type_t>::INPUT_IMAGE_WIDTH;
  using CModelBasicSR_GAN<type_t>::INPUT_IMAGE_HEIGHT;
  using CModelBasicSR_GAN<type_t>::INPUT_IMAGE_DEPTH;

  using CModelBasicSR_GAN<type_t>::OUTPUT_IMAGE_WIDTH;
  using CModelBasicSR_GAN<type_t>::OUTPUT_IMAGE_HEIGHT;
  using CModelBasicSR_GAN<type_t>::OUTPUT_IMAGE_DEPTH;

  using CModelBasicSR_GAN<type_t>::SPEED;

  using CModelBasicSR_GAN<type_t>::Net;

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelSR_GAN(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelSR_GAN();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  void CreateSRGAN(void) override;//создать сеть
  void TrainingNet(bool mnist) override;//запуск обучения нейросети
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelSR_GAN<type_t>::CModelSR_GAN(void)
{
 INPUT_IMAGE_HEIGHT=100;
 INPUT_IMAGE_WIDTH=100;
 INPUT_IMAGE_DEPTH=3;

 SPEED=0.001;

 BATCH_SIZE=1;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelSR_GAN<type_t>::~CModelSR_GAN()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//создать сеть
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSR_GAN<type_t>::CreateSRGAN(void)
{
Net.clear();

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(INPUT_IMAGE_DEPTH,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH)));
 /*
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,8,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);
 */
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);
 /*
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);*/

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(INPUT_IMAGE_DEPTH,1,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));

 /*
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,64,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);
 */

 /*
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,32,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);

 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,32,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));
 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor",false);
 */

 /*Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,INPUT_IMAGE_DEPTH,1,1,0,0,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_TANGENCE,Net[Net.size()-1].get())));
 Net.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,Net[Net.size()-1].get())));
 */

 Net[Net.size()-1]->GetOutputTensor().Print("Output tensor final",false);

 OUTPUT_IMAGE_WIDTH=Net[Net.size()-1]->GetOutputTensor().GetSizeX();
 OUTPUT_IMAGE_HEIGHT=Net[Net.size()-1]->GetOutputTensor().GetSizeY();
 OUTPUT_IMAGE_DEPTH=Net[Net.size()-1]->GetOutputTensor().GetSizeZ();
}
//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelSR_GAN<type_t>::TrainingNet(bool mnist)
{
 CModelBasicSR_GAN<type_t>::TrainingNet(mnist);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************



#endif
