#ifndef C_MODEL_DIFFUSION_H
#define C_MODEL_DIFFUSION_H

//****************************************************************************************************
//Модель диффузионной сети
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "cmodel_basicdiffusion.cu.h"

//****************************************************************************************************
//Модель диффузионной сети
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

  using CModelBasicDiffusion<type_t>::ITERATION_OF_SAVE_IMAGE;
  using CModelBasicDiffusion<type_t>::ITERATION_OF_SAVE_NET;
  using CModelBasicDiffusion<type_t>::SPEED;

  using CModelBasicDiffusion<type_t>::TIME_COUNTER;

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
 IMAGE_HEIGHT=64;//256;//192;//128;
 IMAGE_WIDTH=64;//256;//256;//128;
 IMAGE_DEPTH=3;

 SPEED=0.0003;

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
// создание диффузионной сети (U-Net) с параметризуемым числом блоков
// ПОЛНОСТЬЮ эквивалентно развёрнутому варианту в CreateDiffusionNet
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelDiffusion<type_t>::CreateDiffusionNet(void)
{
 const type_t time_scale=1;//множитель временной добавки
 const uint32_t NUM_BLOCKS=4;//количество блоков энкодера и декодера
 const uint32_t BOTTLENECK_CONVS=3;//количество свёрток "бутылочного горлышка"
 const type_t BN_MOMENTUM=0.9;//фильтр нормализаций
 uint32_t kernels=64;
 uint32_t groups=8;

 uint32_t mlp_time_size=128;

 DiffusionNet.clear();

 // входной слой
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolutionInput<type_t>(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE)));

 // индексы слоёв-разделителей для skip-соединений
 std::vector<uint32_t> split_indices(NUM_BLOCKS,0);

 //сжатие
 for(uint32_t block=0;block<NUM_BLOCKS;block++,kernels*=2,groups*=2)
 {
  // Conv -> BN -> TimeEmb -> RELU
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolution<type_t>(kernels,3,1,1,1,1,DiffusionNet.back().get(),BATCH_SIZE)));
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerGroupNormalization<type_t>(groups,DiffusionNet.back().get(),BATCH_SIZE)));
  //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerBatchNormalization<type_t>(BN_MOMENTUM,DiffusionNet.back().get(),BATCH_SIZE)));
  //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerTimeEmbedding<type_t>(DiffusionNet.back().get(),time_scale,BATCH_SIZE)));
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet.back().get(),BATCH_SIZE)));
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerTimeEmbeddingMLP<type_t>(DiffusionNet.back().get(),mlp_time_size,TIME_COUNTER,BATCH_SIZE)));
  // Splitter для skip-соединения
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerSplitter<type_t>(2,DiffusionNet.back().get(),BATCH_SIZE)));
  split_indices[block] = static_cast<uint32_t>(DiffusionNet.size() - 1);
  // Downsampling (stride=2,каналы сохраняются)
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolution<type_t>(kernels,3,2,2,1,1,DiffusionNet.back().get(),BATCH_SIZE)));
 }


 uint32_t up_pos=DiffusionNet.size();

 //"бутылочное горлышко"
 for(uint32_t i=0;i<BOTTLENECK_CONVS;i++)
 {
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolution<type_t>(kernels,3,1,1,1,1,DiffusionNet.back().get(),BATCH_SIZE)));
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerGroupNormalization<type_t>(groups,DiffusionNet.back().get(),BATCH_SIZE)));
  //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerBatchNormalization<type_t>(BN_MOMENTUM,DiffusionNet.back().get(),BATCH_SIZE)));
  //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerTimeEmbedding<type_t>(DiffusionNet.back().get(),time_scale,BATCH_SIZE)));
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet.back().get(),BATCH_SIZE)));
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerTimeEmbeddingMLP<type_t>(DiffusionNet.back().get(),mlp_time_size,TIME_COUNTER,BATCH_SIZE)));
 }

 uint32_t bottle_pos=DiffusionNet.size();

 kernels/=2;
 groups/=2;

 // ------------------------------ ДЕКОДЕР -------------------------------
 for(int32_t block=static_cast<int32_t>(NUM_BLOCKS)-1;block>=0;block--,kernels/=2,groups/=2)
 {
  // Upsample x2
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerUpSampling<type_t>(2,2,DiffusionNet.back().get(),BATCH_SIZE)));
  // Concat со skip-соединением (split_indices[block] — индекс Splitter-слоя)
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConcatenator<type_t>(DiffusionNet.back().get(),DiffusionNet[split_indices[block]].get(),BATCH_SIZE)));
  // Conv -> BN -> TimeEmb -> GELU
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerConvolution<type_t>(kernels,3,1,1,1,1,DiffusionNet.back().get(),BATCH_SIZE)));
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerGroupNormalization<type_t>(groups,DiffusionNet.back().get(),BATCH_SIZE)));
  //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerBatchNormalization<type_t>(BN_MOMENTUM,DiffusionNet.back().get(),BATCH_SIZE)));
  //DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerTimeEmbedding<type_t>(DiffusionNet.back().get(),time_scale,BATCH_SIZE)));
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiffusionNet.back().get(),BATCH_SIZE)));
  DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerTimeEmbeddingMLP<type_t>(DiffusionNet.back().get(),mlp_time_size,TIME_COUNTER,BATCH_SIZE)));
  if (block>0) DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t>>(new CNetLayerDropOut<type_t>(0.15,DiffusionNet.back().get(),BATCH_SIZE)));
 }

 // --------------------------- ВЫХОДНОЙ СЛОЙ ----------------------------
 DiffusionNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,1,1,DiffusionNet.back().get(),BATCH_SIZE)));



 for(size_t n=0;n<up_pos;n++)
 {
  DiffusionNet[n].get()->PrintInputTensorSize("DownNet");
  DiffusionNet[n].get()->PrintOutputTensorSize("DownNet");
 }

 for(size_t n=up_pos;n<bottle_pos;n++)
 {
  DiffusionNet[n].get()->PrintInputTensorSize("BottleNet");
  DiffusionNet[n].get()->PrintOutputTensorSize("BottleNet");
 }

 for(size_t n=bottle_pos;n<DiffusionNet.size();n++)
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
