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

 IMAGE_HEIGHT=128;
 IMAGE_WIDTH=128;
 IMAGE_DEPTH=3;
 HIDDEN_LAYER_SIDE_X=8;
 HIDDEN_LAYER_SIDE_Y=8;
 HIDDEN_LAYER_SIDE_Z=1;
 HIDDEN_LAYER_SIZE=HIDDEN_LAYER_SIDE_X*HIDDEN_LAYER_SIDE_Y*HIDDEN_LAYER_SIDE_Z;

 SPEED=0.0001;

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
 uint32_t k=1;

 DiffusionNet.clear();

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));
 /*
 //блок 1
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(162,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //делаем разделение для переноса через U-модель
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 uint32_t split_1=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
*/
 //блок 2
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //делаем разделение для переноса через U-модель
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 uint32_t split_2=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //блок 3
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //делаем разделение для переноса через U-модель
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 uint32_t split_3=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //блок 4
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //делаем разделение для переноса через U-модель
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 uint32_t split_4=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //блок 5
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //делаем разделение для переноса через U-модель
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 uint32_t split_5=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //блок 6
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //делаем разделение для переноса через U-модель
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerSplitter<type_t>(2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 uint32_t split_6=DiffusionNet.size()-1;
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerMaxPooling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //промежуточный слой
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //промежуточный слой

 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

   //промежуточный слой
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 size_t up_pos=DiffusionNet.size();

 //объединяем блоки
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConcatenator<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),DiffusionNet[split_6].get(),BATCH_SIZE)));
 //блок 7
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //объединяем блоки
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConcatenator<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),DiffusionNet[split_5].get(),BATCH_SIZE)));
 //блок 8
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //объединяем блоки
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConcatenator<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),DiffusionNet[split_4].get(),BATCH_SIZE)));
 //блок 9
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //объединяем блоки
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConcatenator<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),DiffusionNet[split_3].get(),BATCH_SIZE)));
 //блок 4
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //объединяем блоки
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConcatenator<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),DiffusionNet[split_2].get(),BATCH_SIZE)));
 //блок 5
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32/k,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
/*
 //объединяем блоки
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConcatenator<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),DiffusionNet[split_1].get(),BATCH_SIZE)));
 //блок 6
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerTimeEmbedding<type_t>(DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 */
 //дополнительный тензор перед выходным
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 //выходной тензор
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,1,1,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet[DiffusionNet.size()-1].get(),BATCH_SIZE)));

 for(size_t n=0;n<up_pos;n++)
 {
  DiffusionNet[n].get()->PrintInputTensorSize("DownNet");
  DiffusionNet[n].get()->PrintOutputTensorSize("DownNet");
 }

 for(size_t n=up_pos;n<DiffusionNet.size();n++)
 {
  DiffusionNet[n].get()->PrintInputTensorSize("UpNet");
  DiffusionNet[n].get()->PrintOutputTensorSize("UpNet");
 }
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelDiffusion<type_t>::TrainingNet(bool mnist)
{
 CModelBasicDiffusion<type_t>::TrainingNet(mnist);
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************



#endif
