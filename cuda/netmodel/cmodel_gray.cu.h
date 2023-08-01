#ifndef C_MODEL_GRAY_H
#define C_MODEL_GRAY_H

//****************************************************************************************************
//Главный класс программы
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "cmodelmain.cu.h"

//****************************************************************************************************
//Главный класс программы
//****************************************************************************************************
template<class type_t>
class CModelGray:public CModelMain<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-структуры------------------------------------------------------------------------------------------
  //-переменные-----------------------------------------------------------------------------------------

  using CModelMain<type_t>::BATCH_AMOUNT;
  using CModelMain<type_t>::BATCH_SIZE;

  using CModelMain<type_t>::IMAGE_WIDTH;
  using CModelMain<type_t>::IMAGE_HEIGHT;
  using CModelMain<type_t>::IMAGE_DEPTH;
  using CModelMain<type_t>::NOISE_LAYER_SIDE_X;
  using CModelMain<type_t>::NOISE_LAYER_SIDE_Y;
  using CModelMain<type_t>::NOISE_LAYER_SIDE_Z;
  using CModelMain<type_t>::NOISE_LAYER_SIZE;

  using CModelMain<type_t>::SPEED_DISCRIMINATOR;
  using CModelMain<type_t>::SPEED_GENERATOR;


  using CModelMain<type_t>::GeneratorNet;
  using CModelMain<type_t>::DiscriminatorNet;

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelGray(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelGray();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  void CreateGenerator(void) override;//создать сеть генератора
  void CreateDiscriminator(void) override;//создать сеть дискриминатора
  void TrainingNet(bool mnist) override;//запуск обучения нейросети
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelGray<type_t>::CModelGray(void)
{
 IMAGE_WIDTH=44;
 IMAGE_HEIGHT=44;
 IMAGE_DEPTH=3;
 NOISE_LAYER_SIDE_X=10;
 NOISE_LAYER_SIDE_Y=10;
 NOISE_LAYER_SIDE_Z=3;
 NOISE_LAYER_SIZE=NOISE_LAYER_SIDE_X*NOISE_LAYER_SIDE_Y*NOISE_LAYER_SIDE_Z;

// SPEED_DISCRIMINATOR=0.0002;
// SPEED_GENERATOR=0.001;

 SPEED_DISCRIMINATOR=0.0002*10;
 SPEED_GENERATOR=0.001*100;

 BATCH_SIZE=1;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelGray<type_t>::~CModelGray()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************
//----------------------------------------------------------------------------------------------------
//создать сеть генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelGray<type_t>::CreateGenerator(void)
{

 GeneratorNet.clear();
 //полносвязные слои
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,NULL)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*2,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 //GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*4,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*8,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 //GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*16,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 //свёрточные слои
 size_t convert=GeneratorNet.size()-1;
 GeneratorNet[convert]->GetOutputTensor().ReinterpretSize(NOISE_LAYER_SIDE_Z,NOISE_LAYER_SIDE_Y*4,NOISE_LAYER_SIDE_X*4);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,128,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[convert]->GetOutputTensor().RestoreSize();

 /*
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,32,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
*/
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,IMAGE_DEPTH,1,1,0,0,NNeuron::NEURON_FUNCTION_TANGENCE,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);


 /*
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,NULL)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 //GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 //GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));

 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 size_t convert=GeneratorNet.size()-1;
 GeneratorNet[convert]->GetOutputTensor().ReinterpretSize(NOISE_LAYER_SIDE_Z,NOISE_LAYER_SIDE_Y,NOISE_LAYER_SIDE_X);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[convert]->GetOutputTensor().RestoreSize();

 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 //GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxDePooling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,256,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,512,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 */
 /*
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 */

 /*
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,IMAGE_DEPTH,1,1,0,0,NNeuron::NEURON_FUNCTION_TANGENCE,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 */
}
//----------------------------------------------------------------------------------------------------
//создать сеть дискриминатора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelGray<type_t>::CreateDiscriminator(void)
{
 DiscriminatorNet.clear();
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().ReinterpretSize(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().RestoreSize();
/*
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 */
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
/* DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 */

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*8,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*4,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*2,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1,NNeuron::NEURON_FUNCTION_LINEAR,DiscriminatorNet[DiscriminatorNet.size()-1].get())));

 /*
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().ReinterpretSize(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().RestoreSize();
 */
 /*
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 */
 /*
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,0,0,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutputTensor().Print("Discriminator output tensor",false);
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(256,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 //DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(128,NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 //DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1,NNeuron::NEURON_FUNCTION_LINEAR,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 */
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelGray<type_t>::TrainingNet(bool mnist)
{
 CModelMain<type_t>::TrainingNet(false);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************



#endif
