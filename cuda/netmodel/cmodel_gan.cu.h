#ifndef C_MODEL_GAN_H
#define C_MODEL_GAN_H

//****************************************************************************************************
//Модель GAN
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "cmodel_basicgan.cu.h"

//****************************************************************************************************
//Модель GAN
//****************************************************************************************************
template<class type_t>
class CModelGAN:public CModelBasicGAN<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-структуры------------------------------------------------------------------------------------------
  //-переменные-----------------------------------------------------------------------------------------

  using CModelBasicGAN<type_t>::BATCH_AMOUNT;
  using CModelBasicGAN<type_t>::BATCH_SIZE;

  using CModelBasicGAN<type_t>::IMAGE_WIDTH;
  using CModelBasicGAN<type_t>::IMAGE_HEIGHT;
  using CModelBasicGAN<type_t>::IMAGE_DEPTH;
  using CModelBasicGAN<type_t>::NOISE_LAYER_SIDE_X;
  using CModelBasicGAN<type_t>::NOISE_LAYER_SIDE_Y;
  using CModelBasicGAN<type_t>::NOISE_LAYER_SIDE_Z;
  using CModelBasicGAN<type_t>::NOISE_LAYER_SIZE;

  using CModelBasicGAN<type_t>::SPEED_DISCRIMINATOR;
  using CModelBasicGAN<type_t>::SPEED_GENERATOR;

  using CModelBasicGAN<type_t>::ITERATION_OF_SAVE_IMAGE;
  using CModelBasicGAN<type_t>::ITERATION_OF_SAVE_NET;

  using CModelBasicGAN<type_t>::INPUT_NOISE_MIN;
  using CModelBasicGAN<type_t>::INPUT_NOISE_MAX;

  using CModelBasicGAN<type_t>::GeneratorNet;
  using CModelBasicGAN<type_t>::DiscriminatorNet;

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
 INPUT_NOISE_MIN=-1;
 INPUT_NOISE_MAX=1;

 IMAGE_HEIGHT=256;//128;//192;
 IMAGE_WIDTH=256;//128;//256;
 IMAGE_DEPTH=3;
 NOISE_LAYER_SIDE_X=10;
 NOISE_LAYER_SIDE_Y=10;
 NOISE_LAYER_SIDE_Z=1;
 NOISE_LAYER_SIZE=NOISE_LAYER_SIDE_X*NOISE_LAYER_SIDE_Y*NOISE_LAYER_SIDE_Z;

 SPEED_DISCRIMINATOR=0.00005;
 SPEED_GENERATOR=0.0004;

 ITERATION_OF_SAVE_IMAGE=1;
 ITERATION_OF_SAVE_NET=1;

 BATCH_SIZE=32;

/*
 //ZX
 IMAGE_HEIGHT=192/1;//128;//192;
 IMAGE_WIDTH=256/1;//128;//256;
 IMAGE_DEPTH=3;
 NOISE_LAYER_SIDE_X=10;
 NOISE_LAYER_SIDE_Y=10;
 NOISE_LAYER_SIDE_Z=1;
 NOISE_LAYER_SIZE=NOISE_LAYER_SIDE_X*NOISE_LAYER_SIDE_Y*NOISE_LAYER_SIDE_Z;

 SPEED_DISCRIMINATOR=0.0001;
 SPEED_GENERATOR=0.0002;

 ITERATION_OF_SAVE_IMAGE=1;
 ITERATION_OF_SAVE_NET=1;

 BATCH_SIZE=8;
 */
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
 GeneratorNet.clear();

 //1
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(NOISE_LAYER_SIZE,NULL,BATCH_SIZE)));
 //2
 //ZX
 //GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(3*4*512,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));

 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(4*4*512,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 //3
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 //4

 uint32_t convert=GeneratorNet.size()-1;
 GeneratorNet[convert]->GetOutputTensor().ReinterpretSize(BATCH_SIZE,512,4,4);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 //5
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 //6
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,3,1,1,1,1,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 //7
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 //8
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,3,1,1,1,1,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 //9
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 //10
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,3,1,1,1,1,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 //11
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);

 //12
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,3,1,1,1,1,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 //13
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);


 //12
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,3,1,1,1,1,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);


 //13
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerUpSampling<type_t>(2,2,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 //14
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,3,1,1,1,1,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 //15
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(IMAGE_DEPTH,3,1,1,1,1,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_TANGENCE,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 //GeneratorNet[GeneratorNet.size()-1]->SetMark(true);
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().Print("Output tensor",false);

 GeneratorNet[convert]->GetOutputTensor().RestoreSize();

}
//----------------------------------------------------------------------------------------------------
//создать сеть дискриминатора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelGAN<type_t>::CreateDiscriminator(void)
{
 DiscriminatorNet.clear();

 //1
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().ReinterpretSize(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 //2

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(16,5,2,2,2,2,GeneratorNet[GeneratorNet.size()-1].get(),BATCH_SIZE)));
 GeneratorNet[GeneratorNet.size()-1]->GetOutputTensor().RestoreSize();
 //DiscriminatorNet[DiscriminatorNet.size()-1]->SetMark(true);

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 //DiscriminatorNet[DiscriminatorNet.size()-1]->SetMark(true);
 DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutputTensor().Print("Disc Output tensor",false);

 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(32,5,2,2,2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 //DiscriminatorNet[DiscriminatorNet.size()-1]->SetMark(true);
 DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutputTensor().Print("Disc Output tensor",false);

 //3
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(64,5,2,2,2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 //DiscriminatorNet[DiscriminatorNet.size()-1]->SetMark(true);
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 //DiscriminatorNet[DiscriminatorNet.size()-1]->SetMark(true);
 DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutputTensor().Print("Disc Output tensor",false);

 //4
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(128,5,2,2,2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 //DiscriminatorNet[DiscriminatorNet.size()-1]->SetMark(true);
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 //DiscriminatorNet[DiscriminatorNet.size()-1]->SetMark(true);
 DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutputTensor().Print("Disc Output tensor",false);

  //5
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(256,5,2,2,2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 //DiscriminatorNet[DiscriminatorNet.size()-1]->SetMark(true);
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 //DiscriminatorNet[DiscriminatorNet.size()-1]->SetMark(true);
 DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutputTensor().Print("Disc Output tensor",false);
 //6
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerConvolution<type_t>(512,5,2,2,2,2,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 //DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerBatchNormalization<type_t>(0.9,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerFunction<type_t>(NNeuron::NEURON_FUNCTION_LEAKY_RELU,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutputTensor().Print("Disc Output tensor",false);
 //7
 DiscriminatorNet.push_back(std::shared_ptr<INetLayer<type_t> >(new CNetLayerLinear<type_t>(1,DiscriminatorNet[DiscriminatorNet.size()-1].get(),BATCH_SIZE)));
 DiscriminatorNet[DiscriminatorNet.size()-1]->GetOutputTensor().Print("Disc Output tensor",false);
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelGAN<type_t>::TrainingNet(bool mnist)
{
 CModelBasicGAN<type_t>::TrainingNet(mnist);
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

 uint32_t index=0;
 uint32_t frame=0;
 char str[255];

 memset(last_image_ptr,fone_width*fone_height*sizeof(uint32_t),0);

 while(1)
 {
  sprintf(str,"Test-3/test%05i-%03i.tga",static_cast<int>(index),0);
  int32_t image_height;
  int32_t image_width;

  uint32_t *image_ptr=reinterpret_cast<uint32_t*>(LoadTGAFromFile(str,image_width,image_height));
  if (image_ptr==NULL) break;

  uint32_t left_x=(fone_width-image_width)/2;
  uint32_t top_y=(fone_height-image_height)/2;
  //копируем изображение
  for(uint32_t f=0;f<9;f++,frame++)
  {
   for(uint32_t y=0;y<image_height;y++)
   {
    for(uint32_t x=0;x<image_width;x++)
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
   sprintf(str,"Frame/frame%06i.tga",static_cast<int>(frame));
   SaveTGA(str,fone_width,fone_height,reinterpret_cast<uint8_t*>(fone_ptr));
  }

  memcpy(last_image_ptr,fone_ptr,fone_width*fone_height*sizeof(uint32_t));

  delete[](image_ptr);
  index++;
 }

  for(uint32_t f=0;f<500;f++,frame++)
  {
   sprintf(str,"Frame/frame%06i.tga",static_cast<int>(frame));
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
