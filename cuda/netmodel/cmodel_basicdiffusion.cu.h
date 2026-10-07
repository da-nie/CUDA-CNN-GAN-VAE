#ifndef C_MODEL_BASIC_DIFFUSION_H
#define C_MODEL_BASIC_DIFFUSION_H

//****************************************************************************************************
//Класс-основа для сетей Diffusion
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <math.h>
#include <random>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../../system/system.h"
#include "../../common/tga.h"
#include "../../common/ccolormodel.h"
#include "../ctimestamp.cu.h"

#include "cmodelmain.cu.h"

//****************************************************************************************************
//Класс-основа для сетей VAE
//****************************************************************************************************
template<class type_t>
class CModelBasicDiffusion:public CModelMain<type_t>
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 protected:
  //-структуры------------------------------------------------------------------------------------------
  //параметры диффузии
  struct SDiffusion
  {
   std::vector<type_t> Beta;///<параметры шума для каждого шага
   std::vector<type_t> Alpha;///<1-beta
   std::vector<type_t> AlphaBar;///<произведение alpha от 0 до t

   void Init(uint32_t time_counter)///<инициализация
   {
    Beta.resize(time_counter);
    Alpha.resize(time_counter);
    AlphaBar.resize(time_counter);
    //заполняем коэффициенты зашумления
   /*
    float beta_start=0.01f;
    float beta_end=0.3f;
    for(uint32_t i=0;i<time_counter;i++)
    {
     Beta[i]=beta_start+(beta_end-beta_start)*i/(time_counter-1);
     Alpha[i]=1-Beta[i];
     if (i==0) AlphaBar[i]=Alpha[i];
          else AlphaBar[i]=AlphaBar[i-1]*Alpha[i];
    }*/

    const float s=0.008f;
    const float PI=3.141592653589793f;

    //вычисляем AlphaBar[t] для t от 0 до time_counter-1
    for(uint32_t t=0;t<time_counter;t++)
    {
     float k=static_cast<float>(t)/static_cast<float>(time_counter);
     float value=cosf((k+s)/(1.0f+s)*PI*0.5f);
     AlphaBar[t]=value*value;
    }
    //нормировка (AlphaBar[0] станет строго 1.0)
    float alpha_0=AlphaBar[0];
    for(uint32_t t=0;t<time_counter;t++) AlphaBar[t]/=alpha_0;
    //вычисляем Alpha[t] и Beta[t]
    for(uint32_t t=0;t<time_counter;t++)
    {
     float prev_alpha_bar=(t==0)?1.0f:AlphaBar[t-1];
     if (AlphaBar[t]>1.0f) AlphaBar[t]=1.0f;
     Alpha[t]=AlphaBar[t]/prev_alpha_bar;
     Beta[t]=1.0f-Alpha[t];
    }
   }
  };

  //параметры обучающих изображений
  struct STrainingImage
  {
   uint32_t RealImageIndex;///<индекс истинного изображения
   typename CModelMain<type_t>::SImageTransformation sImageTransformation;///<преобразование изображения
  };
  //-переменные-----------------------------------------------------------------------------------------
  uint32_t IMAGE_WIDTH;///<ширина входных изображений
  uint32_t IMAGE_HEIGHT;///<высота входных изображений
  uint32_t IMAGE_DEPTH;///<глубина входных изображений

  uint32_t BATCH_AMOUNT;///<количество пакетов
  uint32_t BATCH_SIZE;///<размер пакета

  uint32_t ITERATION_OF_SAVE_IMAGE;///<какую итерацию сохранять изображения
  uint32_t ITERATION_OF_SAVE_NET;///<какую итерацию сохранять сеть

  uint32_t TIME_COUNTER;///<число шагов по времени для получения изображения

  double SPEED;///<скорость обучения

  std::vector<std::shared_ptr<INetLayer<type_t> > > DiffusionNet;///<сеть кодера-декодера

  CTensor<type_t> cTensor_Image;
  CTensor<type_t> cTensor_ImageSource;
  CTensor<type_t> cTensor_NoisyImage;
  CTensor<type_t> cTensor_Error;
  CTensor<type_t> cTensor_Noise;
  CTensor<type_t> cTensor_SqrtAlphaBar;
  CTensor<type_t> cTensor_OneMinusAlphaBar;

  std::vector< std::vector<type_t> > RealImage;///<образы истинных изображений
  std::vector<uint32_t> TrainingImageIndex;///<индексы изображений в обучающем наборе
  std::vector<STrainingImage> TrainingImage;///<изображения в обучающем наборе

  using CModelMain<type_t>::STRING_BUFFER_SIZE;
  using CModelMain<type_t>::CUDA_PAUSE_MS;

  using CModelMain<type_t>::SafeLog;
  using CModelMain<type_t>::CrossEntropy;
  using CModelMain<type_t>::IsExit;
  using CModelMain<type_t>::SetExitState;
  using CModelMain<type_t>::LoadMNISTImage;
  using CModelMain<type_t>::LoadImage;
  using CModelMain<type_t>::SaveNetLayers;
  using CModelMain<type_t>::LoadNetLayers;
  using CModelMain<type_t>::SaveNetLayersEMA;
  using CModelMain<type_t>::LoadNetLayersEMA;
  using CModelMain<type_t>::SaveNetLayersTrainingParam;
  using CModelMain<type_t>::LoadNetLayersTrainingParam;
  using CModelMain<type_t>::ExchangeImageIndex;
  using CModelMain<type_t>::SaveImage;
  using CModelMain<type_t>::SpeedTest;
  using CModelMain<type_t>::TransformationImage;

  uint32_t Iteration;///<итерация

  SDiffusion sDiffusion;///<параметры диффузии
  std::vector<type_t> NoisyImage;///<зашумлённое изображение
  std::vector<type_t> Noise;///<накладываемый шум
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CModelBasicDiffusion(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CModelBasicDiffusion();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Execute(void);///<выполнить
 protected:
  //-закрытые функции-----------------------------------------------------------------------------------
  virtual void CreateDiffusionNet(void)=0;///<создать сеть
  void LoadNet(void);///<загрузить сети
  void SaveNet(void);///<сохранить сети
  void LoadTrainingParam(void);///<загрузить параметры обучения
  void SaveTrainingParam(void);///<сохранить параметры обучения
  void TrainingDiffusionNet(uint32_t mini_batch_index,double &cost);///<обучение диффузионной сети
  void SaveRandomImageDDIM(int32_t step_counter);///<сохранить изображение с сокращённым проходом по шагам
  void SaveRandomImageDDPM(void);///<сохранить изображение с полным проходом по всем шагам
  void SaveKitImage(void);///<сохранить изображение из набора
  void Training(void);///<обучение нейросети
  virtual void TrainingNet(bool mnist);///<запуск обучения нейросети

  void InitDiffusion(void);///<инициализация параметров диффузии
  void SetNoisyImage(CTensor<type_t> &cTensor_NoisyImage,CTensor<type_t> &cTensor_Noise,const CTensor<type_t> &cTensor_Image,const std::vector<uint32_t> &time_step_array);///<заполнить тензор зашумлённым изображением
  void GetNoiseTensor(CTensor<type_t> &cTensor);///<получить тензор шума

  void DebugVarVsTime(void);///<диагностика: Var(eps_theta) как функция от t
  void DebugVarVsTimeForImage(void);///<диагностика: Var(eps_theta) как функция от t для изображений
};

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicDiffusion<type_t>::CModelBasicDiffusion(void)
{
 BATCH_AMOUNT=0;
 BATCH_SIZE=0;

 IMAGE_WIDTH=0;
 IMAGE_HEIGHT=0;
 IMAGE_DEPTH=0;

 SPEED=0;

 BATCH_SIZE=1;

 ITERATION_OF_SAVE_IMAGE=1;
 ITERATION_OF_SAVE_NET=1;

 Iteration=0;

 TIME_COUNTER=500;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CModelBasicDiffusion<type_t>::~CModelBasicDiffusion()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************
//----------------------------------------------------------------------------------------------------
//загрузить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::LoadNet(void)
{
 FILE *file=fopen("diffusion_neuronet.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_neuronet.net",false));
  LoadNetLayers(iDataStream_Disc_Ptr.get(),DiffusionNet);
  SYSTEM::PutMessageToConsole("Сеть диффузионной модели загружена.");
 }
 file=fopen("diffusion_neuronet_ema.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_neuronet_ema.net",false));
  LoadNetLayersEMA(iDataStream_Disc_Ptr.get(),DiffusionNet);
  SYSTEM::PutMessageToConsole("Сеть диффузионной модели со средними коэффициентами загружена.");
 }

}

//----------------------------------------------------------------------------------------------------
//сохранить сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveNet(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_neuronet.net",true));
 SaveNetLayers(iDataStream_Disc_Ptr.get(),DiffusionNet);
 std::unique_ptr<IDataStream> iDataStream_Disc_EMA_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_neuronet_ema.net",true));
 SaveNetLayersEMA(iDataStream_Disc_EMA_Ptr.get(),DiffusionNet);
}
//----------------------------------------------------------------------------------------------------
//загрузить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::LoadTrainingParam(void)
{
 FILE *file=fopen("diffusion_training_param.net","rb");
 if (file!=NULL)
 {
  fclose(file);
  std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_training_param.net",false));
  LoadNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),DiffusionNet,Iteration);
  SYSTEM::PutMessageToConsole("Параметры обучения диффузионной сети загружены.");
 }
}
//----------------------------------------------------------------------------------------------------
//сохранить параметры обучения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveTrainingParam(void)
{
 std::unique_ptr<IDataStream> iDataStream_Disc_Ptr(IDataStream::CreateNewDataStreamFile("diffusion_training_param.net",true));
 SaveNetLayersTrainingParam(iDataStream_Disc_Ptr.get(),DiffusionNet,Iteration);
}

//----------------------------------------------------------------------------------------------------
//обучение диффузионной сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::TrainingDiffusionNet(uint32_t mini_batch_index,double &cost)
{
 char str[STRING_BUFFER_SIZE];
 static std::vector<uint32_t> time_step_array(BATCH_SIZE);

 static std::vector<typename CModelMain<type_t>::SImageTransformation> sImageTransformation_List(BATCH_SIZE);//список преобразования изображений

 //создаём тензор входных изображений
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  uint32_t img=b+mini_batch_index*BATCH_SIZE;
  uint32_t training_index=TrainingImageIndex[img];
  uint32_t real_index=TrainingImage[training_index].RealImageIndex;
  uint32_t time_step=CRandom<float>::GetRandValue(TIME_COUNTER*10);
  time_step%=TIME_COUNTER;

  //задаём массив времени
  time_step_array[b]=time_step;
  //задаём тензор изображения
  type_t *ptr=&RealImage[real_index][0];
  uint32_t size=RealImage[real_index].size();
  cTensor_ImageSource.CopyItemLayerWToDevice(b,ptr,size);
  sImageTransformation_List[b]=TrainingImage[training_index].sImageTransformation;
  //задаём временную метку
  for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->SetTimeStep(b,time_step);
 }
 //преобразуем изображения в зависимости от настроек
 TransformationImage(cTensor_Image,cTensor_ImageSource,sImageTransformation_List);

 //накладываем шум на изображение
 //изображение с шумом, шум, изображение,массив времени
 SetNoisyImage(DiffusionNet[0]->GetOutputTensor(),cTensor_Noise,cTensor_Image,time_step_array);
/*
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  if (IsExit()==true) throw("Стоп");
  //задаём изображение
  {
   CTimeStamp cTimeStamp("Задание изображения:");
   //вход сети подключён к изображению с шумом
   uint32_t img=b+mini_batch_index*BATCH_SIZE;
   uint32_t training_index=TrainingImageIndex[img];
   uint32_t real_index=TrainingImage[training_index].RealImageIndex;
   uint32_t time_step=TrainingImage[training_index].TimeStep;
   //задаём изображение с шумом
   SetNoisyImage(b,time_step,DiffusionNet[0]->GetOutputTensor(),cTensor_Image,cTensor_Noise,RealImage[real_index]);
   //задаём временную метку
   for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->SetTimeStep(b,time_step);
  }
 }
*/


 //вычисляем сеть
 {
  CTimeStamp cTimeStamp("Вычисление сети:");
  for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->Forward();
 }
 {
  CTimeStamp cTimeStamp("Вычисление ошибки:");
  CTensorMath<type_t>::Sub(cTensor_Error,DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor(),cTensor_Noise,2,2);
 }
 {
  CTimeStamp cTimeStamp("Задание ошибки:");
  DiffusionNet[DiffusionNet.size()-1]->SetOutputError(cTensor_Error);
 }
 //считаем ошибку
 CTensorMath<type_t>::Pow2(cTensor_Error,cTensor_Error,static_cast<type_t>(1.0/4.0));
 static CTensor<type_t> cTensor_Loss(cTensor_Error.GetSizeW(),cTensor_Error.GetSizeZ(),1,1);
 CTensorMath<type_t>::SumXY(cTensor_Loss,cTensor_Error);

 double error=0;
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  for(uint32_t z=0;z<cTensor_Error.GetSizeZ();z++)
  {
   error+=cTensor_Loss.GetElement(b,z,0,0);
  }
 }
 error/=static_cast<double>(BATCH_SIZE);
 cost+=error;
 //выполняем вычисление весов
 {
  CTimeStamp cTimeStamp("Обучение сети:");
  for(uint32_t m=0,n=DiffusionNet.size()-1;m<DiffusionNet.size();m++,n--) DiffusionNet[n]->TrainingBackward();
 }
}
//----------------------------------------------------------------------------------------------------
// Универсальный сэмплер DDIM (η = 0, детерминированный)
// Численно устойчивая формула через явный x0_pred с clamp.
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveRandomImageDDIM(int32_t step_counter)
{
    // ---- 1. Задаём начальный шум x_T ~ N(0, I) ----
    GetNoiseTensor(DiffusionNet[0]->GetOutputTensor());

    // Диагностика стартового состояния (можно убрать после отладки)
    SaveImage(DiffusionNet[0]->GetOutputTensor(),
              "Test/ddim_start.tga", 0, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);

    // ---- 2. Переключаем сеть в режим инференса ----
    for (uint32_t layer = 0; layer < DiffusionNet.size(); layer++)
    {
        DiffusionNet[layer]->SetUseEMA(true);
        DiffusionNet[layer]->SetInferenceMode(true);
    }

    step_counter=30;

    // ---- 3. Вычисляем шаг пропуска между шагами t ----
    // step_counter = 50 → step = 500/50 = 10 (50 вызовов сети)
    int32_t step = std::max(1, (int32_t)(TIME_COUNTER / step_counter));

    // ---- 4. Основной цикл сэмплинга от t = T-1 до t = 1 ----
    int32_t t=TIME_COUNTER-1;
    for(int32_t n=0;n<step_counter;n++,t-=step)
    {
     if (t<1) t=1;
        // 4.1. Задаём номер шага всем слоям сети
        for (uint32_t b = 0; b < BATCH_SIZE; b++)
            for (uint32_t layer = 0; layer < DiffusionNet.size(); layer++)
                DiffusionNet[layer]->SetTimeStep(b, t);

        // 4.2. Прямой проход сети
        for (uint32_t layer = 0; layer < DiffusionNet.size(); layer++)
            DiffusionNet[layer]->Forward();

        // 4.3. Достаём вход и предсказание шума
        CTensor<type_t>& input_noise  = DiffusionNet[0]->GetOutputTensor();                      // x_t
        CTensor<type_t>& output_noise = DiffusionNet[DiffusionNet.size() - 1]->GetOutputTensor(); // ε_θ

        // 4.4. Коэффициенты для текущего и предыдущего шага
        int32_t t_prev = std::max(0, t - step);

        type_t alpha_bar_t    = sDiffusion.AlphaBar[t];
        type_t alpha_bar_prev = (t_prev == 0) ? (type_t)1.0f : sDiffusion.AlphaBar[t_prev];

        type_t sqrt_ab_t        = std::sqrt(std::max((type_t)1e-20f, alpha_bar_t));
        type_t sqrt_ab_prev     = std::sqrt(alpha_bar_prev);
        type_t sqrt_one_minus_t = std::sqrt(std::max((type_t)0.0f, (type_t)1.0f - alpha_bar_t));

        // 4.5. Обновляем каждый пиксель x_t → x_{t-1}
        for (uint32_t b = 0; b < BATCH_SIZE; b++)
        {
            for (uint32_t z = 0; z < input_noise.GetSizeZ(); z++)
            {
                for (uint32_t y = 0; y < input_noise.GetSizeY(); y++)
                {
                    for (uint32_t x = 0; x < input_noise.GetSizeX(); x++)
                    {
                        type_t input  = input_noise.GetElement(b, z, y, x);   // x_t
                        type_t output = output_noise.GetElement(b, z, y, x);  // ε_θ

                        // --- Шаг 1: оценка чистого изображения x_0 ---
                        type_t x0_pred = (input - sqrt_one_minus_t * output) / sqrt_ab_t;

                        // --- Шаг 2: КЛИППИРОВАНИЕ x_0_pred ---
                        // Это критично: без clamp формула взрывается,
                        // потому что при ᾱ_t → 0 множитель 1/√ᾱ_t огромен.
                        if (x0_pred >  (type_t)1.0f) x0_pred =  (type_t)1.0f;
                        if (x0_pred < (type_t)-1.0f) x0_pred = (type_t)-1.0f;
                        if (!std::isfinite(x0_pred)) x0_pred = (type_t)0.0f; // защита от NaN

                        // --- Шаг 3: DDIM-обновление (η = 0, без стохастики) ---
                        // x_{t-1} = √ᾱ_{t-1}·x_0 + √(1−ᾱ_{t-1})·ε_θ
                        type_t prev = sqrt_ab_prev * x0_pred
                                    + std::sqrt(std::max((type_t)0.0f,
                                                         (type_t)1.0f - alpha_bar_prev)) * output;

                        type_t bound=sqrt_ab_t*1.0f+sqrt_one_minus_t*4.0f;
                        type_t limit=bound*1.2f;


                        if (prev>limit) prev=limit;
                        if (prev<-limit) prev=--limit;


                        // Защита от NaN/Inf в результате
                        if (!std::isfinite(prev)) prev = (type_t)0.0f;

                        input_noise.SetElement(b, z, y, x, prev);
                    }
                }
            }
        }

        // ---- 4.6. Диагностика промежуточных шагов ----
        // (можно убрать после отладки; показывает, как x_t постепенно сходится)
        //if (n % 50 == 0 || t == TIME_COUNTER - 1)
        {
            auto& x = DiffusionNet[0]->GetOutputTensor();
            type_t mn = (type_t)1e30f, mx = (type_t)-1e30f;
            type_t sum = 0, sum2 = 0;
            uint64_t n = 0;
            for (uint32_t b = 0; b < BATCH_SIZE; b++)
                for (uint32_t z = 0; z < x.GetSizeZ(); z++)
                    for (uint32_t y = 0; y < x.GetSizeY(); y++)
                        for (uint32_t u = 0; u < x.GetSizeX(); u++)
                        {
                            type_t v = x.GetElement(b, z, y, u);
                            mn = std::min(mn, v);
                            mx = std::max(mx, v);
                            sum += v; sum2 += v * v; n++;
                        }
            printf("DDIM t=%d  x range=[%.3f, %.3f]  mean=%.3f  var=%.3f\n",
                   t, mn, mx, (double)sum / n,
                   (double)sum2 / n - ((double)sum / n) * ((double)sum / n));

            char str[256];
            sprintf(str, "Test/ddim_t%03d.tga", t);
            SaveImage(x, str, 0, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
        }
    }

    // ---- 5. Сохраняем финальное изображение ----
    cTensor_Image = DiffusionNet[0]->GetOutputTensor();

    char str[STRING_BUFFER_SIZE];
    static uint32_t counter = 0;
    for (uint32_t n = 0; n < BATCH_SIZE; n++)
    {
        sprintf(str, "Test/test%05i-%03i.tga", (int)counter, (int)n);
        SaveImage(cTensor_Image, str, n, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
        if (n == 0)
            SaveImage(cTensor_Image, "Test/test-current.tga", n,
                      IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
    }

    // ---- 6. Возвращаем режим обучения ----
    for (uint32_t layer = 0; layer < DiffusionNet.size(); layer++)
    {
        DiffusionNet[layer]->SetUseEMA(false);
        DiffusionNet[layer]->SetInferenceMode(false);
    }

    //throw("");
}

/*
//----------------------------------------------------------------------------------------------------
// Универсальный стабильный сэмплер (Точный DDPM при step=1, DDIM при step>1)
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveRandomImageDDIM(int32_t step_counter)
{
 GetNoiseTensor(DiffusionNet[0]->GetOutputTensor());
 for(uint32_t layer=0;layer<DiffusionNet.size();layer++)
 {
  DiffusionNet[layer]->SetUseEMA(true);
  DiffusionNet[layer]->SetInferenceMode(true);
 }
// Шаг перехода. При 1 работает как точный DDPM, при >1 как DDIM
 int32_t step=std::max(1,(int32_t)(TIME_COUNTER/step_counter));
 //eta=1.0 дает точную дисперсию DDPM на любом шаге
 //для детерминированного DDIM (без шума на промежутках) нужно ставить ноль
 float eta=1.0f;


 int32_t t=TIME_COUNTER-1;
 for(int32_t n=0;n<step_counter;n++,t-=step)
 {
  if (t<1) t=1;
  int32_t t_prev=std::max(0,t-step);
  //задаём время
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->SetTimeStep(b,t);
  }
  //вычисляем сеть
  for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->Forward();
  float alpha_bar_t=sDiffusion.AlphaBar[t];
  //если пришли в 0, принудительно ставим 1.0 для получения чистой картинки
  float alpha_bar_prev=sDiffusion.AlphaBar[t_prev];
  if (t_prev==0) alpha_bar_prev=1.0;
  //коэффициент сохранения текущего состояния
  float s=std::sqrt(alpha_bar_prev/alpha_bar_t);
  //точная дисперсия. (alpha_bar_prev-alpha_bar_t) теперь ВПОЛНЕ ПОЛОЖИТЕЛЬНОЕ
  float sigma=eta*std::sqrt(std::max(0.0f,(1.0f-alpha_bar_prev)/(1.0f-alpha_bar_t)*(alpha_bar_prev-alpha_bar_t)/alpha_bar_prev));
  //коэффициент направления (вычитания шума)
  float dir_xt_coef=std::sqrt(std::max(0.0f,1.0f-alpha_bar_prev-sigma*sigma));
  //итоговый коэффициент для предсказанного сетью шума (уже включает знак минус)
  float c=-s*std::sqrt(std::max(0.0f,1.0f-alpha_bar_t))+dir_xt_coef;

  CTensor<type_t> &input_noise=DiffusionNet[0]->GetOutputTensor();
  CTensor<type_t> &output_noise=DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor();

  //генерируем шум, только если дисперсия больше нуля
  if (sigma>0.0f) GetNoiseTensor(cTensor_Noise);

  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   uint32_t index=0;
   for(uint32_t z=0;z<input_noise.GetSizeZ();z++)
   {
    for(uint32_t y=0;y<input_noise.GetSizeY();y++)
    {
     for(uint32_t x=0;x<input_noise.GetSizeX();x++,index++)
     {
      type_t noise=0;
      if (sigma>0.0f) noise=cTensor_Noise.GetElement(b,z,y,x);
      type_t input=input_noise.GetElement(b,z,y,x);
      type_t output=output_noise.GetElement(b,z,y,x);//предсказанный шум
      type_t prev_noise=(type_t)s*input+(type_t)c*output+(type_t)sigma*noise;
      if (prev_noise>5) prev_noise=5;
      if (prev_noise<-5) prev_noise=-5;
      input_noise.SetElement(b,z,y,x,prev_noise);
     }
    }
   }
  }
  if (t==1) break;
 }

 //сохраняем изображения
 cTensor_Image=DiffusionNet[0]->GetOutputTensor();

 char str[STRING_BUFFER_SIZE];
 static uint32_t counter=0;
 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/test%05i-%03i.tga",static_cast<int>(counter),static_cast<int>(n));
  SaveImage(cTensor_Image,str,n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor_Image,"Test/test-current.tga",n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }

 for(uint32_t layer=0;layer<DiffusionNet.size();layer++)
 {
  DiffusionNet[layer]->SetUseEMA(false);
  DiffusionNet[layer]->SetInferenceMode(false);
 }
 //counter++;
}
*/

//----------------------------------------------------------------------------------------------------
//сохранить случайное изображение с сети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveRandomImageDDPM(void)
{
/*
 //задаём сети начальный шум
 GetNoiseTensor(DiffusionNet[0]->GetOutputTensor());
 for(uint32_t layer=0;layer<DiffusionNet.size();layer++)
 {
  //DiffusionNet[layer]->SetUseEMA(true);
  DiffusionNet[layer]->SetInferenceMode(true);
 }
 CTensor<type_t> &input_noise=DiffusionNet[0]->GetOutputTensor();
 CTensor<type_t> &output_noise=DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor();
 CTensor<type_t> &cTensor_Image=DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor();

 for(size_t t=0;t<500;t+=50)
 {
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->SetTimeStep(b,t);
  }
  for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->Forward();
  char str[STRING_BUFFER_SIZE];
  sprintf(str,"Test/time-%03i.tga",static_cast<int>(t));
  SaveImage(cTensor_Image,str,0,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
 */


 //задаём сети начальный шум
 GetNoiseTensor(DiffusionNet[0]->GetOutputTensor());

 //TODO: ДИАГНОСТИКА
 SaveImage(DiffusionNet[0]->GetOutputTensor(), "Test/step_start.tga", 0, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
 //TODO: ДИАГНОСТИКА


 for(uint32_t layer=0;layer<DiffusionNet.size();layer++)
 {
  //DiffusionNet[layer]->SetUseEMA(true);
  DiffusionNet[layer]->SetInferenceMode(true);
 }


 // Цикл от TIME_COUNTER-1 до 1 ВКЛЮЧИТЕЛЬНО. Шаг t=0 пропускается!
 for(int32_t t=TIME_COUNTER-1;t>=1;t--)
 {

  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->SetTimeStep(b,t);
  }


    //TODO: ДИАГНОСТИКА
  if (t % 100 == 0 || t==TIME_COUNTER-1) {
    char str[256];
    sprintf(str, "Test/step_t%03d.tga", t);
    SaveImage(DiffusionNet[0]->GetOutputTensor(), str, 0, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
    // вывести статистику
    auto &x=DiffusionNet[0]->GetOutputTensor();
    type_t mn=10000000000, mx=-10000000000, sum=0, sum2=0; uint64_t n=0;
    for(uint32_t b=0;b<BATCH_SIZE;b++) for(uint32_t z=0;z<x.GetSizeZ();z++)
     for(uint32_t y=0;y<x.GetSizeY();y++) for(uint32_t u=0;u<x.GetSizeX();u++){
        type_t v=x.GetElement(b,z,y,u);
        mn=std::min(mn,v); mx=std::max(mx,v); sum+=v; sum2+=v*v; n++;
    }
    printf("t=%d  x_t range=[%.3f, %.3f]  mean=%.3f  var=%.3f\n",
           t, mn, mx, sum/n, sum2/n-(sum/n)*(sum/n));
}

  //TODO: ДИАГНОСТИКА


  for(uint32_t layer=0;layer<DiffusionNet.size();layer++) DiffusionNet[layer]->Forward();

  // Забираем коэффициенты
  float alpha=sDiffusion.Alpha[t];
  float beta=sDiffusion.Beta[t];
  float sqrt_alpha=std::sqrt(alpha);
  type_t sqrt_one_minus_alpha_bar=std::sqrt(std::max(0.0f, 1.0f-sDiffusion.AlphaBar[t]));

  type_t k=(1.0/sqrt_alpha);

  CTensor<type_t> &input_noise=DiffusionNet[0]->GetOutputTensor();
  CTensor<type_t> &output_noise=DiffusionNet[DiffusionNet.size()-1]->GetOutputTensor();

  //GetNoiseTensor(cTensor_Noise);


type_t alpha_t        =sDiffusion.Alpha[t];
type_t alpha_bar_t    =sDiffusion.AlphaBar[t];
type_t alpha_bar_prev =(t > 0) ? sDiffusion.AlphaBar[t-1] : 1.0f;
type_t beta_t         =sDiffusion.Beta[t];

type_t sqrt_alpha_t         =std::sqrt(alpha_t);
type_t sqrt_alpha_bar_t     =std::sqrt(alpha_bar_t);
type_t sqrt_alpha_bar_prev  =std::sqrt(alpha_bar_prev);
type_t sqrt_one_minus_alpha_bar_t=std::sqrt(std::max(0.0f, 1.0f-alpha_bar_t));


  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   uint32_t index=0;
   for(uint32_t z=0;z<input_noise.GetSizeZ();z++)
   {
    for(uint32_t y=0;y<input_noise.GetSizeY();y++)
    {
     for(uint32_t x=0;x<input_noise.GetSizeX();x++,index++)
     {

/*
type_t input =input_noise.GetElement(b, z, y, x);
type_t output=output_noise.GetElement(b, z, y, x);

// 1. Оценка x0
type_t x0_pred=(input-sqrt_one_minus_alpha_bar_t * output) / sqrt_alpha_bar_t;

// 2. Клиппирование x0_pred
if (t<(int32_t)(TIME_COUNTER/2))
{

 if (x0_pred >  1.0f) x0_pred= 1.0f;
 if (x0_pred<-1.0f) x0_pred=-1.0f;
}
// 3. Реконструкция x_{t-1} через DDPM-коэффициенты
// Коэффициент при x0_pred: √ᾱ_{t-1}
// Коэффициент при ε_θ:     √α_t · (1-ᾱ_{t-1}) / √(1-ᾱ_t)
type_t coef_x0 =sqrt_alpha_bar_prev;
type_t coef_eps=sqrt_alpha_t * (1.0f-alpha_bar_prev) / sqrt_one_minus_alpha_bar_t;

type_t prev_noise=coef_x0 * x0_pred+coef_eps * output;

// 4. Добавляем шум (только если не последний шаг)
type_t noise=0.0f;  // если детерминированный режим
// type_t noise=cTensor_Noise.GetElement(b, z, y, x);  // если стохастический
if (t > 1) prev_noise += std::sqrt(beta_t) * noise;

input_noise.SetElement(b, z, y, x, prev_noise);
*/

      type_t noise=0;//cTensor_Noise.GetElement(b,z,y,x);

      type_t input=input_noise.GetElement(b,z,y,x);
      type_t output=output_noise.GetElement(b,z,y,x);

      type_t sqrt_alpha_t    =std::sqrt(alpha_t);
      type_t sqrt_one_minus_ab=std::sqrt(std::max(0.0f, 1.0f-alpha_bar_t));

      type_t eps_coef=(1.0f-alpha_t) / sqrt_one_minus_ab;   // β_t/√(1−ᾱ_t)
      type_t prev_noise=(input-eps_coef * output) / sqrt_alpha_t;

      if (prev_noise>5) prev_noise=5;
      if (prev_noise<-5) prev_noise=-5;

      /*
      // Это в точности: (beta / sqrt(1-alpha_bar)) * epsilon
      type_t net_noise=output*(1.0-alpha)/sqrt_one_minus_alpha_bar;

      // Это в точности: (1 / sqrt(alpha)) * (x_t-net_noise)
      type_t prev_noise=k*(input-net_noise);

      // Добавляем случайный шум, если это не самый последний шаг (t=1)
      // При t=1 мы получаем x_0, и добавлять шум нельзя
      if (t>1) prev_noise+=std::sqrt(beta)*noise;
*/
      input_noise.SetElement(b,z,y,x,prev_noise);

     }
    }
   }
  }


 }
 //сохраняем изображения (они на входе сети)
 cTensor_Image=DiffusionNet[0]->GetOutputTensor();



 char str[STRING_BUFFER_SIZE];
 static uint32_t counter=0;
 for(uint32_t n=0;n<BATCH_SIZE;n++)
 {
  sprintf(str,"Test/test%05i-%03i.tga",static_cast<int>(counter),static_cast<int>(n));
  SaveImage(cTensor_Image,str,n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  if (n==0) SaveImage(cTensor_Image,"Test/test-current.tga",n,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
 //cTensor_Image.Print("Image");
 //throw("Стоп");

 for(uint32_t layer=0;layer<DiffusionNet.size();layer++)
 {
  DiffusionNet[layer]->SetUseEMA(false);
  DiffusionNet[layer]->SetInferenceMode(false);
 }
 //counter++;

}


//----------------------------------------------------------------------------------------------------
//сохранить изображение из набора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SaveKitImage(void)
{
 char str[STRING_BUFFER_SIZE];
 static std::vector<uint32_t> time_step_array(BATCH_SIZE);
 //накладываем шум на изображение
 for(uint32_t n=0;n<TIME_COUNTER;n++)
 {
  uint32_t t_index=0;
  uint32_t r_index=TrainingImage[t_index].RealImageIndex;
  uint32_t time_step=n;
  //задаём массив времени
  for(size_t b=0;b<BATCH_SIZE;b++)
  {
   time_step_array[b]=time_step;
   //задаём тензор изображения
   type_t *ptr=&RealImage[r_index][0];
   uint32_t size=RealImage[r_index].size();
   cTensor_Image.CopyItemLayerWToDevice(b,ptr,size);
  }
  SetNoisyImage(cTensor_NoisyImage,cTensor_Noise,cTensor_Image,time_step_array);
  sprintf(str,"Test/kit%03i.tga",static_cast<int>(n));
  SaveImage(cTensor_NoisyImage,str,0,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
 }
}

//----------------------------------------------------------------------------------------------------
//обучение нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::Training(void)
{
 char str_b[STRING_BUFFER_SIZE];

 const double speed=SPEED;
 uint32_t max_iteration=1000000000;//максимальное количество итераций обучения

 uint32_t image_amount=RealImage.size();

 std::string str;

 CCUDATimeSpent cCUDATimeSpent;

 //SaveKitImage();

 while(Iteration<max_iteration)
 {
  SYSTEM::PutMessageToConsole("----------");
  SYSTEM::PutMessageToConsole("Итерация:"+std::to_string(static_cast<long double>(Iteration+1)));

  ExchangeImageIndex(TrainingImageIndex);

  if (Iteration%ITERATION_OF_SAVE_NET==0)
  {
   //DebugVarVsTime();
   DebugVarVsTimeForImage();
   SYSTEM::PutMessageToConsole("Start save net.");
   SaveNet();
   SaveTrainingParam();
   SYSTEM::PutMessageToConsole("Save net done.");
   SYSTEM::PutMessageToConsole("");
  }

  if (Iteration%ITERATION_OF_SAVE_IMAGE==0)
  {
   SYSTEM::PutMessageToConsole("Start save image.");
   SaveRandomImageDDIM(100);
   SYSTEM::PutMessageToConsole("Save image done.");
   //DebugVarVsTime();
   DebugVarVsTimeForImage();
   //SaveRandomImageDDPM();
  }

  double max_cost=0;
  for(uint32_t batch=0;batch<BATCH_AMOUNT;batch++)
  {
   if (IsExit()==true) throw("Стоп");

   str="Итерация:";
   str+=std::to_string(static_cast<long double>(Iteration+1));
   str+=" минипакет:";
   str+=std::to_string(static_cast<long double>(batch+1));
   str+=" из ";
   str+=std::to_string(static_cast<long double>(BATCH_AMOUNT));
   SYSTEM::PutMessageToConsole(str);

   {
    cCUDATimeSpent.Start();
	//обучаем сеть
    double cost=0;

    for(uint32_t n=0;n<DiffusionNet.size();n++) DiffusionNet[n]->TrainingResetDeltaWeight();

    TrainingDiffusionNet(batch,cost);
    //корректируем веса
    {
     CTimeStamp cTimeStamp("Обновление весов:");
     for(uint32_t n=0;n<DiffusionNet.size();n++) DiffusionNet[n]->TrainingUpdateWeight(speed,Iteration);
    }


    str="Ошибка:";
    str+=std::to_string(static_cast<long double>(cost));
    SYSTEM::PutMessageToConsole(str);

    max_cost+=cost;
   }
   float current_max_cost=max_cost/static_cast<long double>(batch+1);

   float gpu_time=cCUDATimeSpent.Stop();
   sprintf(str_b,"На минипакет ушло:%.2f мс.",gpu_time);
   SYSTEM::PutMessageToConsole(str_b);
   sprintf(str_b,"Общая ошибка:%.2f",current_max_cost);
   SYSTEM::PutMessageToConsole(str_b);
   SYSTEM::PutMessageToConsole("");
  }
  max_cost/=static_cast<long double>(BATCH_AMOUNT);
  sprintf(str_b,"Общая ошибка:%.2f",max_cost);
  SYSTEM::PutMessageToConsole(str_b);
  SYSTEM::PutMessageToConsole("");
  FILE *file=fopen("cost.txt","ab");
  fprintf(file,"%f\r\n",max_cost);
  fclose(file);
  Iteration++;

  //if (max_cost<10) break;
 }
}

//----------------------------------------------------------------------------------------------------
//запуск обучения нейросети
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::TrainingNet(bool mnist)
{
 char str[STRING_BUFFER_SIZE];
 SYSTEM::MakeDirectory("Test");

 cTensor_Image=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 cTensor_ImageSource=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 cTensor_NoisyImage=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 cTensor_Error=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 cTensor_Noise=CTensor<type_t>(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 cTensor_SqrtAlphaBar=CTensor<type_t>(BATCH_SIZE,1,1,1);
 cTensor_OneMinusAlphaBar=CTensor<type_t>(BATCH_SIZE,1,1,1);

 CreateDiffusionNet();

 for(uint32_t n=0;n<DiffusionNet.size();n++)
 {
  DiffusionNet[n]->Reset();
  DiffusionNet[n]->EnableEMA(true);//-копируем случайно инициализированные тензоры в средние веса
 }

 LoadNet();//одновременно будут загружены средние веса

 //включаем обучение
 for(uint32_t n=0;n<DiffusionNet.size();n++)
 {
  DiffusionNet[n]->SetInferenceMode(false);
  DiffusionNet[n]->TrainingModeAdam(0.9,0.999);
  DiffusionNet[n]->TrainingStart();
  //DiffusionNet[n]->EnableEMA(true);-это скопирует текущие настройки сети, а не из файла средних
 }

 //загружаем изображения
 //if (LoadMNISTImage("mnist.bin",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,RealImageIndex)==false)
 if (LoadImage("RealImage",IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH,RealImage,TrainingImageIndex)==false)
 {
  SYSTEM::PutMessage("Не удалось загрузить образы изображений!");
  return;
 }
 SYSTEM::PutMessage("Образы изображений загружены.");
 //инициализируем параметры диффузии
 InitDiffusion();

 //создаём обучающий набор
 uint32_t scale=1;//расширение набора
 uint32_t i=0;
 TrainingImage.resize(RealImage.size()*scale);
 TrainingImageIndex.resize(RealImage.size()*scale);
 for(uint32_t n=0;n<RealImage.size();n++)
 {
  for(uint32_t m=0;m<scale;m++,i++)
  {
   TrainingImage[i].RealImageIndex=n;
   TrainingImageIndex[i]=n;
  }
 }

 //добавляем отражённые по горизонтали
 {
  uint32_t size_img=TrainingImage.size();
  TrainingImage.resize(size_img*2);
  TrainingImageIndex.resize(size_img*2);
  for(uint32_t n=0;n<size_img;n++)
  {
   TrainingImage[n+size_img]=TrainingImage[n];
   TrainingImage[n+size_img].sImageTransformation.FlipHorizontal=true;
   TrainingImageIndex[n+size_img]=n+size_img;
  }
 }

 //добавляем с изменённой яркостью
 {
  uint32_t size_img=TrainingImage.size();
  TrainingImage.resize(size_img*2);
  TrainingImageIndex.resize(size_img*2);
  for(uint32_t n=0;n<size_img;n++)
  {
   TrainingImage[n+size_img]=TrainingImage[n];
   TrainingImage[n+size_img].sImageTransformation.ScaleZ[0]=0.9;
   TrainingImageIndex[n+size_img]=n+size_img;
  }
 }

 //добавляем с изменёнными цветами
 {
  uint32_t size_img=TrainingImage.size();
  TrainingImage.resize(size_img*2);
  TrainingImageIndex.resize(size_img*2);
  for(uint32_t n=0;n<size_img;n++)
  {
   TrainingImage[n+size_img]=TrainingImage[n];
   TrainingImage[n+size_img].sImageTransformation.ScaleZ[1]=0.9;
   TrainingImage[n+size_img].sImageTransformation.ScaleZ[2]=0.9;
   TrainingImageIndex[n+size_img]=n+size_img;
  }
 }

 //дополняем набор до кратного размеру пакета
 uint32_t image_amount=TrainingImage.size();
 BATCH_AMOUNT=image_amount/BATCH_SIZE;
 if (BATCH_AMOUNT==0) BATCH_AMOUNT=1;
 if (image_amount%BATCH_SIZE!=0)
 {
  uint32_t index=0;
  for(uint32_t n=image_amount%BATCH_SIZE;n<BATCH_SIZE;n++,index++)
  {
   TrainingImageIndex.push_back(TrainingImageIndex[index%image_amount]);
  }
  image_amount=TrainingImageIndex.size();
  BATCH_AMOUNT=image_amount/BATCH_SIZE;
 }


 sprintf(str,"Исходных изображений:%i Обучающих изображений:%i Минипакетов:%i",static_cast<int>(RealImage.size()),static_cast<int>(image_amount),static_cast<int>(BATCH_AMOUNT));
 SYSTEM::PutMessageToConsole(str);
/*
 //тест сохранения изображений
 static std::vector<typename CModelMain<type_t>::SImageTransformation> sImageTransformation_List(BATCH_SIZE);//список преобразования изображений
 for(uint32_t n=0;n<TrainingImageIndex.size();n+=BATCH_SIZE)
 {
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   uint32_t training_index=TrainingImageIndex[n+b];
   uint32_t real_index=TrainingImage[training_index].RealImageIndex;
   //задаём тензор изображения
   type_t *ptr=&RealImage[real_index][0];
   uint32_t size=RealImage[real_index].size();
   cTensor_ImageSource.CopyItemLayerWToDevice(b,ptr,size);
   sImageTransformation_List[b]=TrainingImage[training_index].sImageTransformation;
  }
  //преобразуем изображения в зависимости от настроек
  TransformationImage(cTensor_Image,cTensor_ImageSource,sImageTransformation_List);
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   char str[STRING_BUFFER_SIZE];
   sprintf(str,"Test/out%04i.tga",static_cast<int>(n+b));
   SaveImage(cTensor_Image,str,b,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH);
  }
 }
 throw("Stop");
*/


 //загружаем параметры обучения
 LoadTrainingParam();
 //запускаем обучение
 Training();
 //отключаем обучение
 for(uint32_t n=0;n<DiffusionNet.size();n++) DiffusionNet[n]->TrainingStop();

 SaveNet();
}


//----------------------------------------------------------------------------------------------------
//!инициализация параметров диффузии
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::InitDiffusion(void)
{
 sDiffusion.Init(TIME_COUNTER);
}

#ifdef USE_GPU_NORMAL_RANDOM_GENERATOR_FOR_DIFFUSION_NET
//----------------------------------------------------------------------------------------------------
//заполнить тензор зашумлённым изображением
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SetNoisyImage(CTensor<type_t> &cTensor_NoisyImage,CTensor<type_t> &cTensor_Noise,const CTensor<type_t> &cTensor_Image,const std::vector<uint32_t> &time_step_array)
{
 //вычисляем коэффициенты для текущего шага и заполняем тензоры
 for(size_t n=0;n<time_step_array.size();n++)
 {
  uint32_t time_step=time_step_array[n];
  type_t sqrt_alpha_bar=std::sqrt(sDiffusion.AlphaBar[time_step]);
  type_t sqrt_one_minus_alpha_bar=std::sqrt(std::max(0.0f, 1.0f-sDiffusion.AlphaBar[time_step]));
  cTensor_SqrtAlphaBar.SetElement(n,0,0,0,sqrt_alpha_bar);
  cTensor_OneMinusAlphaBar.SetElement(n,0,0,0,sqrt_one_minus_alpha_bar);
 }
 CTensorMath<type_t>::GetNoiseImageAndNoise(cTensor_NoisyImage,cTensor_Noise,cTensor_Image,cTensor_SqrtAlphaBar,cTensor_OneMinusAlphaBar);
}
//----------------------------------------------------------------------------------------------------
//!получить тензор шума
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::GetNoiseTensor(CTensor<type_t> &cTensor)
{
 CTensorMath<type_t>::SetNormalNoise(cTensor);
}
#endif

#ifndef USE_GPU_NORMAL_RANDOM_GENERATOR_FOR_DIFFUSION_NET
//----------------------------------------------------------------------------------------------------
//заполнить тензор зашумлённым изображением
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::SetNoisyImage(CTensor<type_t> &cTensor_NoisyImage,CTensor<type_t> &cTensor_Noise,const CTensor<type_t> &cTensor_Image,const std::vector<uint32_t> &time_step_array)
{
 if (time_step_array.size()!=BATCH_SIZE) throw("Размерность вектора временного шага отличается от количества элементов пакета!") ;

 GetNoiseTensor(cTensor_Noise);

 //применяем шум к изображению
 size_t index=0;
 type_t *noise_ptr=cTensor_Noise.GetColumnPtr(0,0,0);
 const type_t *image_ptr=cTensor_Image.GetColumnPtr(0,0,0);
 type_t *noisyimage_ptr=cTensor_NoisyImage.GetColumnPtr(0,0,0);

 for(uint32_t w=0;w<cTensor_Noise.GetSizeW();w++)
 {
  uint32_t time_step=time_step_array[w];
  type_t sqrt_alpha_bar=std::sqrt(sDiffusion.AlphaBar[time_step]);
  type_t sqrt_one_minus_alpha_bar=std::sqrt(std::max(0.0f,1.0f-sDiffusion.AlphaBar[time_step]));
  for(uint32_t z=0;z<cTensor_Noise.GetSizeZ();z++)
  {
   for(uint32_t y=0;y<cTensor_Noise.GetSizeY();y++)
   {
    for(uint32_t x=0;x<cTensor_Noise.GetSizeX();x++,index++,noise_ptr++,noisyimage_ptr++,image_ptr++)
    {
     type_t noise=*noise_ptr;
     type_t image=*image_ptr;
     type_t value=sqrt_alpha_bar*image+sqrt_one_minus_alpha_bar*noise;
     *noisyimage_ptr=value;
    }
   }
  }
 }
}
//----------------------------------------------------------------------------------------------------
//!получить тензор шума
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::GetNoiseTensor(CTensor<type_t> &cTensor)
{
 double max=1;
 double min=-1;
 double average=0;//(max+min)/2.0;
 double log_var=1;//(average-min)/3.0;

 static std::random_device rd;
 std::mt19937_64 gen(rd());//ускоренный генератор _64
 std::normal_distribution<type_t> dist(average,log_var);

 type_t *noise_ptr=cTensor.GetColumnPtr(0,0,0);

 for(uint32_t w=0;w<cTensor.GetSizeW();w++)
 {
  for(uint32_t z=0;z<cTensor.GetSizeZ();z++)
  {
   for(uint32_t y=0;y<cTensor.GetSizeY();y++)
   {
    for(uint32_t x=0;x<cTensor.GetSizeX();x++,noise_ptr++)
    {
     *noise_ptr=dist(gen);
     /*
     type_t value=static_cast<type_t>(CRandom<type_t>::GetGaussRandValue(average,log_var));
     while(value<min || value>max) value=static_cast<type_t>(CRandom<type_t>::GetGaussRandValue(average,log_var));//åñëè ýòî ïðîèçîøëî ãåíåðèðóåì íîâîå ÷èñëî.
     *noise_ptr=value;
     */
    }
   }
  }
 }
}
#endif

//----------------------------------------------------------------------------------------------------
// диагностика: Var(eps_theta) как функция от t
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::DebugVarVsTime(void)
{
 SYSTEM::PutMessageToConsole("=== Var(eps_theta) vs t ===");
 // запоминаем текущий режим и переключаемся в инференс
 uint32_t net_size=static_cast<uint32_t>(DiffusionNet.size());
 for(uint32_t layer=0;layer<net_size;layer++)
 {
  DiffusionNet[layer]->SetUseEMA(false);
  DiffusionNet[layer]->SetInferenceMode(true);
 }
 // фиксированный входной шум x_t
 CTensor<type_t> x_test(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 GetNoiseTensor(x_test);
 x_test.CopyToDevice();
 // массив шагов, на которых проверяем
 std::vector<uint32_t> test_time;
 for(uint32_t t=0;t<TIME_COUNTER;t+=100) test_time.push_back(t);
 test_time.push_back(TIME_COUNTER-1);
 for(uint32_t n=0;n<test_time.size();n++)
 {
  uint32_t t=test_time[n];
  //подаём на вход сети
  DiffusionNet[0]->GetOutputTensor()=x_test;
  //задаём время всем слоям
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   for(uint32_t layer=0;layer<net_size;layer++) DiffusionNet[layer]->SetTimeStep(b,t);
  }
  //вычисляем сеть
  for(uint32_t layer=0;layer<net_size;layer++) DiffusionNet[layer]->Forward();
  //считаем Var(eps_theta)
  CTensor<type_t> &out=DiffusionNet.back()->GetOutputTensor();
  double sum=0;
  double sum2=0;
  uint64_t cnt=0;
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   for(uint32_t z=0;z<out.GetSizeZ();z++)
   {
    for(uint32_t y=0;y<out.GetSizeY();y++)
    {
     for(uint32_t x=0;x<out.GetSizeX();x++)
     {
      double v=static_cast<double>(out.GetElement(b,z,y,x));
      sum+=v;
      sum2+=v*v;
      cnt++;
     }
    }
   }
  }
  double mean=sum/cnt;
  double var=sum2/cnt-mean*mean;
  char str[STRING_BUFFER_SIZE];
  snprintf(str,STRING_BUFFER_SIZE,"t=%4u  mean=%+.5f  var=%.5f",t,mean,var);
  SYSTEM::PutMessageToConsole(str);
 }
 for(uint32_t layer=0;layer<net_size;layer++)
 {
  DiffusionNet[layer]->SetUseEMA(false);
  DiffusionNet[layer]->SetInferenceMode(false);
 }
 SYSTEM::PutMessageToConsole("===========================");
}

//----------------------------------------------------------------------------------------------------
// диагностика: Var(eps_theta) как функция от t для изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CModelBasicDiffusion<type_t>::DebugVarVsTimeForImage(void)
{
 SYSTEM::PutMessageToConsole("=== Var(eps_theta) vs t for image ===");
 //запоминаем текущий режим и переключаемся в инференс, 600, 700, 800, 900, 999
 uint32_t net_size=static_cast<uint32_t>(DiffusionNet.size());
 for(uint32_t layer=0;layer<net_size;layer++)
 {
  DiffusionNet[layer]->SetUseEMA(false);
  DiffusionNet[layer]->SetInferenceMode(true);
 }
 //задаём реальные изображения
 CTensor<type_t> x0(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
 for(uint32_t b=0;b<BATCH_SIZE;b++)
 {
  uint32_t index=b%RealImage.size();
  type_t *ptr=&RealImage[index][0];
  uint32_t size=RealImage[index].size();
  x0.CopyItemLayerWToDevice(index,ptr,size);
 }
 //фиксированный входной шум x_t
 CTensor<type_t> x_test(BATCH_SIZE, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH);
 GetNoiseTensor(x_test);
 x_test.CopyToDevice();
 //массив шагов, на которых проверяем
 std::vector<uint32_t> test_time;
 for(uint32_t t=0;t<TIME_COUNTER;t+=100) test_time.push_back(t);
 test_time.push_back(TIME_COUNTER-1);

 for(uint32_t n=0;n<test_time.size();n++)
 {
  uint32_t t=test_time[n];
  //генерируем эпсилон
  CTensor<type_t> eps(BATCH_SIZE,IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH);
  GetNoiseTensor(eps);
  //вычисляем
  type_t alpha=std::sqrt(sDiffusion.AlphaBar[t]);
  type_t beta=std::sqrt(1.0f-sDiffusion.AlphaBar[t]);
  CTensor<type_t> xt=alpha*x0+beta*eps;
  //вычисляем сеть
  DiffusionNet[0]->GetOutputTensor()=xt;
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   for(uint32_t layer=0;layer<net_size;layer++) DiffusionNet[layer]->SetTimeStep(b,t);
  }
  for(uint32_t layer=0;layer<net_size;layer++) DiffusionNet[layer]->Forward();
  //считаем Var(eps_theta)
  CTensor<type_t> &out=DiffusionNet.back()->GetOutputTensor();
  double sum=0;
  double sum2=0;
  uint64_t cnt=0;
  for(uint32_t b=0;b<BATCH_SIZE;b++)
  {
   for(uint32_t z=0;z<out.GetSizeZ();z++)
   {
    for(uint32_t y=0;y<out.GetSizeY();y++)
    {
     for(uint32_t x=0;x<out.GetSizeX();x++)
     {
      double v=static_cast<double>(out.GetElement(b,z,y,x));
      sum+=v;
      sum2+=v*v;
      cnt++;
     }
    }
   }
  }
  double mean=sum/cnt;
  double var=sum2/cnt-mean*mean;
  char str[STRING_BUFFER_SIZE];
  snprintf(str,STRING_BUFFER_SIZE,"t=%4u  mean=%+.5f  var=%.5f",t,mean,var);
  SYSTEM::PutMessageToConsole(str);
 }
 for(uint32_t layer=0;layer<net_size;layer++)
 {
  DiffusionNet[layer]->SetUseEMA(false);
  DiffusionNet[layer]->SetInferenceMode(false);
 }
 SYSTEM::PutMessageToConsole("===========================");
}


//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//выполнить
//----------------------------------------------------------------------------------------------------

template<class type_t>
void CModelBasicDiffusion<type_t>::Execute(void)
{
 //зададим размер динамической памяти на стороне устройства (1М по-умолчанию)
 //cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*512);
 TrainingNet(true);
}

#endif
