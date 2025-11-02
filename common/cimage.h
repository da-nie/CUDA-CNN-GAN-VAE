#ifndef C_IMAGE_H
#define C_IMAGE_H

//****************************************************************************************************
//класс работы с изображениями
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <math.h>

#include "ccolormodel.h"
#include "tga.h"

//****************************************************************************************************
//макроопределения
//****************************************************************************************************

//****************************************************************************************************
//константы
//****************************************************************************************************

//****************************************************************************************************
//предварительные объявления
//****************************************************************************************************
template <class type_t>
class CTensor;

//****************************************************************************************************
//класс работы с изображениями
//****************************************************************************************************
template <class type_t>
class CImage
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CImage(void);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CImage();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  static bool LoadImage(const std::string &file_name,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector<type_t> &image);///<загрузить изображение
  static bool CreateResamplingImage(uint32_t input_image_width,uint32_t input_image_height,uint32_t input_image_depth,const std::vector<type_t> &image_input,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector<type_t> &image_output);///<создать изображение другого разрешения
  static void SaveImage(CTensor<type_t> &cTensor,const std::string &name,uint32_t w,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth);///<сохранить изображение
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template <class type_t>
CImage<type_t>::CImage(void)
{
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template <class type_t>
CImage<type_t>::~CImage()
{
}

//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//загрузить изображение
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CImage<type_t>::LoadImage(const std::string &file_name,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector<type_t> &image)
{
 int32_t height;
 int32_t width;
 uint32_t *img_ptr=reinterpret_cast<uint32_t*>(LoadTGAFromFile(file_name.c_str(),width,height));//загрузить tga-файл
 if (img_ptr==NULL) return(false);
 image=std::vector<type_t>(output_image_depth*output_image_width*output_image_height);

 double dx=static_cast<double>(width)/static_cast<double>(output_image_width);
 double dy=static_cast<double>(height)/static_cast<double>(output_image_height);

 for(uint32_t y=0;y<output_image_height;y++)
 {
  for(uint32_t x=0;x<output_image_width;x++)
  {
   uint32_t xp=x*dx;
   uint32_t yp=y*dy;

   uint32_t offset=(xp+yp*width);
   uint32_t color=img_ptr[offset];
   float r=(color>>0)&0xff;
   float g=(color>>8)&0xff;
   float b=(color>>16)&0xff;

   float gray=(r+g+b)/3.0f;

   gray/=255.0;
   gray*=2.0;
   gray-=1.0;

   float sl;
   float sa;
   float sb;

   CColorModel::RGB2Lab(r,g,b,sl,sa,sb);

   sl*=2.0;
   sl-=1.0;

   r/=255.0;
   g/=255.0;
   b/=255.0;
   //приведём к диапазону [-1,1]

   r*=2.0;
   r-=1.0;

   g*=2.0;
   g-=1.0;

   b*=2.0;
   b-=1.0;

   if (output_image_depth==1) image[x+y*output_image_width]=gray;
   if (output_image_depth==3)
   {
    r=sl;
    g=sa;
    b=sb;

    if (r<-1) r=-1;
    if (g<-1) g=-1;
    if (b<-1) b=-1;

    if (r>1) r=1;
    if (g>1) g=1;
    if (b>1) b=1;

	image[x+y*output_image_width+0*output_image_width*output_image_height]=r;
	image[x+y*output_image_width+1*output_image_width*output_image_height]=g;
	image[x+y*output_image_width+2*output_image_width*output_image_height]=b;
   }
  }
 }
 delete[](img_ptr);
 return(true);
}
//----------------------------------------------------------------------------------------------------
//создать изображение другого разрешения
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CImage<type_t>::CreateResamplingImage(uint32_t input_image_width,uint32_t input_image_height,uint32_t input_image_depth,const std::vector<type_t> &image_input,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth,std::vector<type_t> &image_output)
{
 uint32_t size=image_input.size();
 image_output=std::vector<type_t>(output_image_depth*output_image_width*output_image_height);
 const type_t *img_ptr=&image_input[0];

 double dx=static_cast<double>(input_image_width)/static_cast<double>(output_image_width);
 double dy=static_cast<double>(input_image_height)/static_cast<double>(output_image_height);

 for(uint32_t y=0;y<output_image_height;y++)
 {
  for(uint32_t x=0;x<output_image_width;x++)
  {
   uint32_t xp=x*dx;
   uint32_t yp=y*dy;
   for(uint32_t z=0;z<output_image_depth;z++)
   {
    uint32_t offset=xp+yp*input_image_width+z*input_image_width*input_image_height;
    type_t color=img_ptr[offset];
    image_output[x+y*output_image_width+z*output_image_width*output_image_height]=color;
   }
  }
 }
 return(true);
}

//----------------------------------------------------------------------------------------------------
//сохранить изображение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CImage<type_t>::SaveImage(CTensor<type_t> &cTensor,const std::string &name,uint32_t w,uint32_t output_image_width,uint32_t output_image_height,uint32_t output_image_depth)
{
 std::vector<uint32_t> image(output_image_width*output_image_height);
 type_t *ptr=cTensor.GetColumnPtr(w,0,0);
 for(uint32_t y=0;y<output_image_height;y++)
 {
  for(uint32_t x=0;x<output_image_width;x++)
  {
   float r=0;
   float g=0;
   float b=0;

   float ir=0;
   float ig=0;
   float ib=0;

   if (output_image_depth==3)
   {
    ir=ptr[x+y*output_image_width+(output_image_height*output_image_width)*0];
    ig=ptr[x+y*output_image_width+(output_image_height*output_image_width)*1];
    ib=ptr[x+y*output_image_width+(output_image_height*output_image_width)*2];

    //восстановление из RGB
    {
     r=ir;
     g=ig;
     b=ib;

     r+=1.0;
     r/=2.0;

     g+=1.0;
     g/=2.0;

     b+=1.0;
     b/=2.0;

     r*=255.0;
     g*=255.0;
     b*=255.0;
    }

    //восстановление из Lab
    {
     float sl=ir;
     float sa=ig;
     float sb=ib;

	 sl+=1.0;
	 sl/=2.0;

	 CColorModel::Lab2RGB(sl,sa,sb,r,g,b);
    }


    if (r<0) r=0;
    if (r>255) r=255;

    if (g<0) g=0;
    if (g>255) g=255;

    if (b<0) b=0;
    if (b>255) b=255;

   }
   if (output_image_depth==1)
   {
    double gray=ptr[x+y*output_image_width+(output_image_height*output_image_width)*0];
   gray+=1.0;
    gray/=2.0;

    gray*=255.0;

    if (gray<0) gray=0;
    if (gray>255) gray=255;

	r=gray;
	g=gray;
	b=gray;
   }
   uint32_t offset=x+y*output_image_width;
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

   image[offset]=color;
  }
 }
 SaveTGA(name.c_str(),output_image_width,output_image_height,reinterpret_cast<uint8_t*>(&image[0]));
}

#endif
