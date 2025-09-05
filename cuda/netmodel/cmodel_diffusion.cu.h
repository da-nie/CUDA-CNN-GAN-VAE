#ifndef C_MODEL_DIFFUSION_H
#define C_MODEL_DIFFUSION_H

//****************************************************************************************************
//Модель VAE
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "cmodel_basicdiffusion.cu.h"

//****************************************************************************************************
//Модель VAE
//****************************************************************************************************
template<class type_t>
class CModelDiffusion:public CModelBasicDiffusion<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-структуры------------------------------------------------------------------------------------------
  //-переменные-----------------------------------------------------------------------------------------

  using CModelBasicDiffusion<type_t>::BATCH_AMOUNT;
  using CModelBasicDiffusion<type_t>::BATCH_SIZE;

  using CModelBasicDiffusion<type_t>::IMAGE_WIDTH;
  using CModelBasicDiffusion<type_t>::IMAGE_HEIGHT;
  using CModelBasicDiffusion<type_t>::IMAGE_DEPTH;
  using CModelBasicDiffusion<type_t>::HIDDEN_LAYER_SIDE_X;
  using CModelBasicDiffusion<type_t>::HIDDEN_LAYER_SIDE_Y;
  using CModelBasicDiffusion<type_t>::HIDDEN_LAYER_SIDE_Z;
  using CModelBasicDiffusion<type_t>::HIDDEN_LAYER_SIZE;

  using CModelBasicDiffusion<type_t>::ITERATION_OF_SAVE_IMAGE;
  using CModelBasicDiffusion<type_t>::ITERATION_OF_SAVE_NET;
  using CModelBasicDiffusion<type_t>::SPEED;

  using CModelBasicDiffusion<type_t>::DiffusionNet;
  using CModelBasicDiffusion<type_t>::DiffusionNet;

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelDiffusion(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelDiffusion();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  void CreateDiffusionNet(void) override;///<создать диффузионную сеть
  void TrainingNet(bool mnist) override;///<запуск обучения нейросети
  void CreateDownNet(void);///<создать сеть уменьшения до скрытого слоя
  void CreateUpNet(void);///<создать сеть увеличения от скрытого слоя
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelDiffusion<type_t>::CModelDiffusion(void)
{
 /*
 IMAGE_HEIGHT=128;
 IMAGE_WIDTH=128;
 IMAGE_DEPTH=3;
 HIDDEN_LAYER_SIDE_X=32;
 HIDDEN_LAYER_SIDE_Y=32;
 HIDDEN_LAYER_SIDE_Z=1;
 HIDDEN_LAYER_SIZE=HIDDEN_LAYER_SIDE_X*HIDDEN_LAYER_SIDE_Y*HIDDEN_LAYER_SIDE_Z;
*/

 IMAGE_HEIGHT=32;//128;
 IMAGE_WIDTH=32;//128;
 IMAGE_DEPTH=1;
 HIDDEN_LAYER_SIDE_X=8;
 HIDDEN_LAYER_SIDE_Y=8;
 HIDDEN_LAYER_SIDE_Z=1;
 HIDDEN_LAYER_SIZE=HIDDEN_LAYER_SIDE_X*HIDDEN_LAYER_SIDE_Y*HIDDEN_LAYER_SIDE_Z;

 SPEED=0.00002;

 BATCH_SIZE=8;

 ITERATION_OF_SAVE_IMAGE=1;
 ITERATION_OF_SAVE_NET=1;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelDiffusion<type_t>::~CModelDiffusion()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//создать диффузионную сеть
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelDiffusion<type_t>::CreateDiffusionNet(void)
{
 DiffusionNet.clear();

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 //блок 1
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //делаем разделение для переноса через U-модель
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 uint32_t split_1=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);
 //блок 2
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //делаем разделение для переноса через U-модель
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 uint32_t split_2=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);
 //блок 3
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //делаем разделение для переноса через U-модель
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 uint32_t split_3=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);

 //промежуточный слой
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);

 //объединяем блоки
 uint32_t concat_3=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[split_3].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConcatecator<type_t>(DiffusionNet[concat_3].get(),DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 //блок 4
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 //объединяем блоки
 uint32_t concat_2=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[split_2].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConcatecator<type_t>(DiffusionNet[concat_2].get(),DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);
 //блок 5
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 //объединяем блоки
 uint32_t concat_1=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[split_1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConcatecator<type_t>(DiffusionNet[concat_1].get(),DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);
 //блок 6
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 //выходной тензор
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor final",false);


 /*
 CreateDownNet();
 CreateUpNet();
 */
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelDiffusion<type_t>::TrainingNet(bool mnist)
{
 CModelBasicDiffusion<type_t>::TrainingNet(mnist);
}



//----------------------------------------------------------------------------------------------------
//создать сеть уменьшения до скрытого слоя
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelDiffusion<type_t>::CreateDownNet(void)
{
/*
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);


 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(HIDDEN_LAYER_SIZE,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_SIGMOID,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor final",false);
 */

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
/*
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);
*/
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,5,2,2,2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor",false);
/*
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(HIDDEN_LAYER_SIZE,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("DownNet output tensor final",false);
 */
}
//----------------------------------------------------------------------------------------------------
//создать сеть увеличения от скрытого слоя
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelDiffusion<type_t>::CreateUpNet(void)
{
 /*
 uint32_t convert=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(512*4*4,DiffusionNet[convert].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 convert=DiffusionNet.size()-1;
 DiffusionNet[convert]->GetOutputTensor().ReinterpretSize(BATCH_SIZE,512,4,4);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_TANGENCE,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor final",false);

 DiffusionNet[convert]->GetOutputTensor().RestoreSize();
 */

/*
 uint32_t convert=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(128*4*4,DiffusionNet[convert].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);
 DiffusionNet[convert]->GetOutputTensor().ReinterpretSize(BATCH_SIZE,128,4,4);
 convert=DiffusionNet.size()-1;
 */

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);
 /*

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor",false);*/

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor().Print("UpNet output tensor final",false);

 //DiffusionNet[convert]->GetOutputTensor().RestoreSize();

}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************



#endif
