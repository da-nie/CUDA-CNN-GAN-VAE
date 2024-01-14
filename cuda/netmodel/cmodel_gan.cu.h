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
class CModelGAN:public CModelMain<type_t>
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
  CModelGAN(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelGAN();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  void CreateGenerator(void) override;//создать сеть генератора
  void CreateDiscriminator(void) override;//создать сеть дискриминатора
  void TrainingNet(bool mnist) override;//запуск обучения нейросети
  void CreateFrameForMovie(void);//создать кадры для видео
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelGAN<type_t>::CModelGAN(void)
{
 IMAGE_HEIGHT=224;//192;
 IMAGE_WIDTH=224;//256;
 IMAGE_DEPTH=3;
 NOISE_LAYER_SIDE_X=11;
 NOISE_LAYER_SIDE_Y=11;
 NOISE_LAYER_SIDE_Z=IMAGE_DEPTH;
 NOISE_LAYER_SIZE=NOISE_LAYER_SIDE_X*NOISE_LAYER_SIDE_Y*NOISE_LAYER_SIDE_Z;

// SPEED_DISCRIMINATOR=0.01;
// SPEED_GENERATOR=0.02;

 SPEED_DISCRIMINATOR=0.001;
 SPEED_GENERATOR=SPEED_DISCRIMINATOR*50;

 BATCH_SIZE=20;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelGAN<type_t>::~CModelGAN()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

// ffmpeg -start_number 0 -i test%05d-000.tga -vcodec mpeg4 test.avi
// ffmpeg -start_number 0 -i test%05d-000.tga -vcodec h264 test.avi

//----------------------------------------------------------------------------------------------------
//создать сеть генератора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelGAN<type_t>::CreateGenerator(void)
{
 //GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerDropOut<type_t>(0.8,GeneratorNet[GeneratorNet.size()-1].get())));

 GeneratorNet.clear();
 //полносвязные слои
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NULL)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE*4,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 //свёрточные слои
 size_t convert=GeneratorNet.size()-1;
 GeneratorNet[convert]->GetOutputTensor().ReinterpretSize(NOISE_LAYER_SIDE_Z,NOISE_LAYER_SIDE_Y*2,NOISE_LAYER_SIDE_X*2);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(4,256,1,1,0,0,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[convert]->GetOutputTensor().RestoreSize();
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(5,128,1,1,0,0,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,64,1,1,0,0,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,32,1,1,0,0,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBackConvolution<type_t>(3,IMAGE_DEPTH,1,1,0,0,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_TANGENCE,GeneratorNet[GeneratorNet.size()-1].get())));
// GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_SIGMOID,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
}
//----------------------------------------------------------------------------------------------------
//создать сеть дискриминатора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelGAN<type_t>::CreateDiscriminator(void)
{
 DiscriminatorNet.clear();
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().ReinterpretSize(IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(8,3,1,1,0,0,GeneratorNet[GeneratorNet.size()-1].get())));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().RestoreSize();
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,1,1,0,0,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,0,0,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,3,1,1,0,0,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get())));

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerAveragePooling<type_t>(2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get())));

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_SIGMOID,DiscriminatorNet[DiscriminatorNet.size()-1].get())));
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelGAN<type_t>::TrainingNet(bool mnist)
{
 CModelMain<type_t>::TrainingNet(false);
}

//----------------------------------------------------------------------------------------------------
//создать кадры для видео
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelGAN<type_t>::CreateFrameForMovie(void)
{
 int32_t fone_height;
 int32_t fone_width;
 uint32_t *fone_ptr=reinterpret_cast<uint32_t*>(LoadTGAFromFile("fone.tga",fone_width,fone_height));
 if (fone_ptr==NULL) return;

 uint32_t *last_image_ptr=new uint32_t[fone_width*fone_height*sizeof(uint32_t)];

 SYSTEM::MakeDirectory("Frame");

 size_t index=0;
 size_t frame=0;
 char str[255];

 memset(last_image_ptr,fone_width*fone_height*sizeof(uint32_t),0);

 while(1)
 {
  sprintf(str,"Test-3/test%05i-%03i.tga",index,0);
  int32_t image_height;
  int32_t image_width;

  uint32_t *image_ptr=reinterpret_cast<uint32_t*>(LoadTGAFromFile(str,image_width,image_height));
  if (image_ptr==NULL) break;

  size_t left_x=(fone_width-image_width)/2;
  size_t top_y=(fone_height-image_height)/2;
  //копируем изображение
  for(size_t f=0;f<9;f++,frame++)
  {
   for(size_t y=0;y<image_height;y++)
   {
    for(size_t x=0;x<image_width;x++)
    {
     uint32_t offset_image=x+y*image_width;
     uint32_t offset_fone=(x+left_x)+(y+top_y)*fone_width;

     uint32_t color_2=image_ptr[offset_image];
     float r_2=(color_2>>0)&0xff;
     float g_2=(color_2>>8)&0xff;
     float b_2=(color_2>>16)&0xff;

     uint32_t color_1=last_image_ptr[offset_fone];
     float r_1=(color_1>>0)&0xff;
     float g_1=(color_1>>8)&0xff;
     float b_1=(color_1>>16)&0xff;

     if (f==8) fone_ptr[offset_fone]=image_ptr[offset_image];
     else
     {
      float r=(r_2-r_1)/9.0;
      float g=(g_2-g_1)/9.0;
      float b=(b_2-b_1)/9.0;
      r*=f;
      g*=f;
      b*=f;

      r+=+r_1;
      g+=+g_1;
      b+=+b_1;

      uint8_t rc=static_cast<uint8_t>(r);
      uint8_t gc=static_cast<uint8_t>(g);
      uint8_t bc=static_cast<uint8_t>(b);

      uint32_t color=0xff;
      color<<=8;
      color|=bc;
      color<<=8;
      color|=gc;
      color<<=8;
      color|=rc;

      fone_ptr[offset_fone]=color;
     }
    }
   }
   sprintf(str,"Frame/frame%06i.tga",frame);
   SaveTGA(str,fone_width,fone_height,reinterpret_cast<uint8_t*>(fone_ptr));
  }

  memcpy(last_image_ptr,fone_ptr,fone_width*fone_height*sizeof(uint32_t));

  delete[](image_ptr);
  index++;
 }

  for(size_t f=0;f<500;f++,frame++)
  {
   sprintf(str,"Frame/frame%06i.tga",frame);
   SaveTGA(str,fone_width,fone_height,reinterpret_cast<uint8_t*>(fone_ptr));
  }

 delete[](fone_ptr);
 delete[](last_image_ptr);

 //fmpeg -start_number 0 -i frame%06d.tga -vcodec h264 test.avi
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************



#endif
