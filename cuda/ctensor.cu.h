#ifndef C_TENSOR_H
#define C_TENSOR_H

//****************************************************************************************************
//Класс тензоров произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <vector>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../common/idatastream.h"
#include "../system/system.h"
#include "ccudadevicevector.cu.h"

//****************************************************************************************************
//макроопределения
//****************************************************************************************************

//****************************************************************************************************
//константы
//****************************************************************************************************

//****************************************************************************************************
//предварительные объявления
//****************************************************************************************************
template<class type_t>
class CTensor;

//****************************************************************************************************
//прототипы функций
//****************************************************************************************************

template<class type_t>
CTensor<type_t> operator+(const CTensor<type_t>& cTensor_Left,const CTensor<type_t>& cTensor_Right);//оператор "+"
template<class type_t>
CTensor<type_t> operator-(const CTensor<type_t>& cTensor_Left,const CTensor<type_t>& cTensor_Right);//оператор "-"
template<class type_t>
CTensor<type_t> operator*(const CTensor<type_t>& cTensor_Left,const CTensor<type_t>& cTensor_Right);//оператор "*"
template<class type_t>
CTensor<type_t> operator*(const CTensor<type_t>& cTensor_Left,const type_t& value_right);//оператор "*"
template<class type_t>
CTensor<type_t> operator*(const type_t& value_left,const CTensor<type_t>& cTensor_Right);//оператор "*"
template<class type_t>
CTensor<type_t> operator*(const CTensor<type_t>& cTensor_Left,const CTensor<type_t>& cTensor_Right);//оператор "*"

//****************************************************************************************************
//Класс тензоров произвольной размерности
//****************************************************************************************************
template<class type_t>
class CTensor
{
 template<class new_type_t>
 friend struct STensorKernel;

 template<class new_type_t>
 friend class CTensorMath;

 template<class new_type_t>
 friend class CTensorConv;

 template<class new_type_t>
 friend class CTensorApplyFunc;

 template<class new_type_t>
 friend class CTensorTest;

 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  std::vector<type_t> Item;///<массив компонентов тензора

  size_t Size_X;///<размер по X
  size_t Size_Y;///<размер по Y
  size_t Size_Z;///<размер по Z

  size_t BasedSize_X;///<исходный размер по X
  size_t BasedSize_Y;///<исходный размер по Y
  size_t BasedSize_Z;///<исходный размер по Z

  CCUDADeviceVector<type_t> DeviceItem;///<данные на устройстве
  mutable bool OnChange;///<изменились ли данные
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CTensor<type_t>(size_t size_z=1,size_t size_y=1,size_t size_x=1);
  //-конструктор копирования----------------------------------------------------------------------------
  CTensor<type_t>(const CTensor<type_t> &cTensor);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CTensor<type_t>();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  size_t GetSizeX(void) const;///<получить размер по x
  size_t GetSizeY(void) const;///<получить размер по y
  size_t GetSizeZ(void) const;///<получить размер по z
  type_t GetElement(size_t z,size_t y,size_t x) const;///<получить элемент тензора
  void SetElement(size_t z,size_t y,size_t x,type_t value);///<задать элемент тензора
  type_t* GetColumnPtr(size_t z,size_t y);///<получить указатель на строку тензора
  const type_t* GetColumnPtr(size_t z,size_t y) const;///<получить указатель на строку тензора
  void Unitary(void);///<привести к единичному виду
  void Zero(void);///<обнулить тензор
  void Move(CTensor<type_t> &cTensor);///<переместить тензор
  void CopyItem(CTensor<type_t> &cTensor);///<скопировать только элементы

  CTensor<type_t>& operator=(const CTensor<type_t> &cTensor);///<оператор "="

  friend CTensor<type_t> operator+<type_t>(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<оператор "+"
  friend CTensor<type_t> operator-<type_t>(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<оператор "-"

  friend CTensor<type_t> operator*<type_t>(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<оператор "*"

  friend CTensor<type_t> operator*<type_t>(const CTensor<type_t> &cTensor_Left,const type_t &value_right);///<оператор "*"
  friend CTensor<type_t> operator*<type_t>(const type_t &value_left,const CTensor<type_t> &cTensor_Right);///<оператор "*"

  bool Save(IDataStream *iDataStream_Ptr);///<сохранить тензор
  bool Load(IDataStream *iDataStream_Ptr);///<загрузить тензор

  void ExchangeSizeXY(void);///<обменять размеры по X и по Y местами
  void ReinterpretSize(size_t size_z,size_t size_y,size_t size_x);///<интерпретировать размер по-новому
  void RestoreSize(void);///<восстановить первоначальную интерпретацию размеров

  void Normalize(void);///<нормировка тензора
  type_t GetNorma(size_t z) const;///<получить норму тензора

  void CopyToDevice(bool force=false) const;///<скопировать тензор на устройство
  void CopyFromDevice(bool force=false);///<скопировать тензор с устройства
  void SetOnChange(void);///<установить, что данные изменились

  void Print(const std::string &name,bool print_value=true) const;///<вывод тензора на экран
  bool Compare(const CTensor<type_t> &cTensor_Control,const std::string &name="") const;///<сравнение тензоров
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};

//****************************************************************************************************
//константы
//****************************************************************************************************

static const double CTENSOR_EPS=0.0000000001;

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>::CTensor(size_t size_z,size_t size_y,size_t size_x)
{
 Size_X=size_x;
 Size_Y=size_y;
 Size_Z=size_z;

 BasedSize_X=Size_X;
 BasedSize_Y=Size_Y;
 BasedSize_Z=Size_Z;

 Item.resize(Size_X*Size_Y*Size_Z);
 DeviceItem.resize(Size_X*Size_Y*Size_Z);
 SetOnChange();
}
//----------------------------------------------------------------------------------------------------
//конструктор копирования
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>::CTensor(const CTensor<type_t> &cTensor)
{
 if (&cTensor==this) return;
 Item=cTensor.Item;
 DeviceItem.resize(Item.size());
 Size_X=cTensor.Size_X;
 Size_Y=cTensor.Size_Y;
 Size_Z=cTensor.Size_Z;
 BasedSize_X=Size_X;
 BasedSize_Y=Size_Y;
 BasedSize_Z=Size_Z;
 SetOnChange();
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>::~CTensor()
{
 Item.clear();
 DeviceItem.clear();
 Size_X=0;
 Size_Y=0;
 Size_Z=0;
 BasedSize_X=Size_X;
 BasedSize_Y=Size_Y;
 BasedSize_Z=Size_Z;
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------------------------

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//получить размер по x
//----------------------------------------------------------------------------------------------------
template<class type_t>
size_t CTensor<type_t>::GetSizeX(void) const
{
 return(Size_X);
}
//----------------------------------------------------------------------------------------------------
//получить размер по y
//----------------------------------------------------------------------------------------------------
template<class type_t>
size_t CTensor<type_t>::GetSizeY(void) const
{
 return(Size_Y);
}
//----------------------------------------------------------------------------------------------------
//получить размер по z
//----------------------------------------------------------------------------------------------------
template<class type_t>
size_t CTensor<type_t>::GetSizeZ(void) const
{
 return(Size_Z);
}
//----------------------------------------------------------------------------------------------------
//получить элемент тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensor<type_t>::GetElement(size_t z,size_t y,size_t x) const
{
 if (x>=Size_X || y>=Size_Y || z>=Size_Z) throw("Ошибка доступа к элементу тензора для чтения!");
 return(Item[Size_X*y+x+Size_X*Size_Y*z]);
}
//----------------------------------------------------------------------------------------------------
//задать элемент тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::SetElement(size_t z,size_t y,size_t x,type_t value)
{
 if (x>=Size_X || y>=Size_Y || z>=Size_Z) throw("Ошибка доступа к элементу тензора для записи!");
 Item[Size_X*y+x+ Size_X*Size_Y*z]=value;
 SetOnChange();
}

//----------------------------------------------------------------------------------------------------
//получить указатель на строку тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t* CTensor<type_t>::GetColumnPtr(size_t z,size_t y)
{
 if (y>=Size_Y || z>=Size_Z) throw("Ошибка получения указателя на строку тензора!");
 return(&Item[Size_X*y+Size_X*Size_Y*z]);
}


//----------------------------------------------------------------------------------------------------
//получить указатель на строку тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
const type_t* CTensor<type_t>::GetColumnPtr(size_t z,size_t y) const
{
 if (y>=Size_Y || z>=Size_Z) throw("Ошибка получения указателя на строку тензора!");
 return(&Item[Size_X*y+Size_X*Size_Y*z]);
}
//----------------------------------------------------------------------------------------------------
//привести к единичному виду
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Unitary(void)
{
 type_t *o_ptr=&Item[0];
 for(size_t z=0;z<Size_Z;z++)
 {
  for(size_t y=0;y<Size_Y;y++)
  {
   for(size_t x=0;x<Size_X;x++,o_ptr++)
   {
    if (x==y) *o_ptr=1;
         else *o_ptr=0;
   }
  }
 }
 SetOnChange();
}
//----------------------------------------------------------------------------------------------------
//обнулить тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Zero(void)
{
 type_t *o_ptr=&Item[0];
 for(size_t z=0;z<Size_Z;z++)
 {
  for(size_t y=0;y<Size_Y;y++)
  {
   for(size_t x=0;x<Size_X;x++,o_ptr++)
   {
    *o_ptr=0;
   }
  }
 }
 SetOnChange();
}

//----------------------------------------------------------------------------------------------------
//переместить тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Move(CTensor<type_t> &cTensor)
{
 if (this==&cTensor) return;
 Item=std::move(cTensor.Item);
 DeviceItem.swap(cTensor.DeviceItem);

 Size_X=cTensor.Size_X;
 Size_Y=cTensor.Size_Y;
 Size_Z=cTensor.Size_Z;
 BasedSize_X=cTensor.BasedSize_X;
 BasedSize_Y=cTensor.BasedSize_Y;
 BasedSize_Z=cTensor.BasedSize_Z;

 cTensor.Size_X=0;
 cTensor.Size_Y=0;
 cTensor.Size_Z=0;

 cTensor.BasedSize_X=0;
 cTensor.BasedSize_Y=0;
 cTensor.BasedSize_Z=0;

 SetOnChange();
 cTensor.SetOnChange();
}
//----------------------------------------------------------------------------------------------------
//скопировать только элементы
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyItem(CTensor<type_t> &cTensor)
{
 if (Size_X*Size_Y*Size_Z!=cTensor.GetSizeX()*cTensor.GetSizeY()*cTensor.GetSizeZ()) throw("Нельзя копировать элементы тензора, если их количество различно.");
 Item=cTensor.Item;
 SetOnChange();
}
//----------------------------------------------------------------------------------------------------
//оператор "="
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>& CTensor<type_t>::operator=(const CTensor<type_t> &cTensor)
{
 if (this!=&cTensor)
 {
  Item=cTensor.Item;
  DeviceItem=cTensor.DeviceItem;
  Size_X=cTensor.Size_X;
  Size_Y=cTensor.Size_Y;
  Size_Z=cTensor.Size_Z;

  BasedSize_X=cTensor.BasedSize_X;
  BasedSize_Y=cTensor.BasedSize_Y;
  BasedSize_Z=cTensor.BasedSize_Z;
  SetOnChange();
 }
 return(*this);
}
//----------------------------------------------------------------------------------------------------
//сохранить тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensor<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 //сохраняем размерность тензора
 iDataStream_Ptr->SaveUInt32(Size_Z);
 iDataStream_Ptr->SaveUInt32(Size_Y);
 iDataStream_Ptr->SaveUInt32(Size_X);
 //сохраняем данные тензора
 for(size_t n=0;n<Size_X*Size_Y*Size_Z;n++) iDataStream_Ptr->SaveDouble(Item[n]);
 return(true);
}
//----------------------------------------------------------------------------------------------------
//загрузить тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensor<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 //загружаем размерность тензора
 Size_Z=iDataStream_Ptr->LoadUInt32();
 Size_Y=iDataStream_Ptr->LoadUInt32();
 Size_X=iDataStream_Ptr->LoadUInt32();

 std::vector<type_t> item(Size_X*Size_Y*Size_Z);
 Item.clear();
 std::swap(Item,item);
 DeviceItem.resize(Item.size());

 //загружаем данные тензора
 for(size_t n=0;n<Size_X*Size_Y*Size_Z;n++) Item[n]=static_cast<type_t>(iDataStream_Ptr->LoadDouble());

 SetOnChange();
 return(true);
}

//----------------------------------------------------------------------------------------------------
//обменять размеры по X и по Y местами
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::ExchangeSizeXY(void)
{
 size_t tmp=Size_X;
 Size_X=Size_Y;
 Size_Y=tmp;
}
//----------------------------------------------------------------------------------------------------
//интерпретировать размер по-новому
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::ReinterpretSize(size_t size_z,size_t size_y,size_t size_x)
{
 if (Size_X*Size_Y*Size_Z!=size_x*size_y*size_z) throw("Новая интерпретация размеров невозможна из-за различного количества элементов.");
 Size_X=size_x;
 Size_Y=size_y;
 Size_Z=size_z;
}
//----------------------------------------------------------------------------------------------------
//восстановить первоначальную интерпретацию размеров
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::RestoreSize(void)
{
 Size_X=BasedSize_X;
 Size_Y=BasedSize_Y;
 Size_Z=BasedSize_Z;
}


//----------------------------------------------------------------------------------------------------
//нормировка тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Normalize(void)
{
 for(size_t z=0;z<Size_Z;z++)
 {
  type_t norma=GetNorma(z);
  if (norma<CTENSOR_EPS) continue;
  for(size_t y=0;y<Size_Y;y++)
  {
   for(size_t x=0;x<Size_X;x++)
   {
    Item[Size_X*y+x+z*Size_X*Size_Y]/=norma;
   }
  }
 }
 SetOnChange();
}
//----------------------------------------------------------------------------------------------------
//получить норму тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensor<type_t>::GetNorma(size_t z) const
{
 type_t norma=0;
 for(size_t y=0;y<Size_Y;y++)
 {
  for(size_t x=0;x<Size_X;x++)
  {
   norma+=Item[Size_X*y+x+z*Size_X*Size_Y]*Item[Size_X*y+x+z*Size_X*Size_Y];
  }
 }
 norma=sqrt(norma);
 return(norma);
}

//----------------------------------------------------------------------------------------------------
///!скопировать тензор на устройство
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyToDevice(bool force) const
{
 if (OnChange==false && force==false) return;
 DeviceItem.copy_host_to_device(&Item[0],Item.size());
 OnChange=false;
}
//----------------------------------------------------------------------------------------------------
///!скопировать тензор с устройства
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyFromDevice(bool force)
{
 if (OnChange==false && force==false) return;
 DeviceItem.copy_device_to_host(&Item[0],Item.size());
 OnChange=false;
}

//----------------------------------------------------------------------------------------------------
///!установить, что данные изменились
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::SetOnChange(void)
{
 OnChange=true;
}
//----------------------------------------------------------------------------------------------------
///!вывод тензора на экран
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Print(const std::string &name,bool print_value) const
{
 printf("***** %s *****\r\n",name.c_str());
 printf("Z:%i Y:%i X:%i\r\n",Size_Z,Size_Y,Size_X);
 if (print_value==false) return;
 for(size_t z=0;z<GetSizeZ();z++)
 {
  printf("-----%i-----\r\n",z);
  for(size_t y=0;y<GetSizeY();y++)
  {
   for(size_t x=0;x<GetSizeX();x++)
   {
    type_t e1=GetElement(z,y,x);
    printf("%f ",e1);
   }
   printf("\r\n");
  }
 }
 printf("\r\n");
}
//----------------------------------------------------------------------------------------------------
///!сравнение тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensor<type_t>::Compare(const CTensor<type_t> &cTensor_Control,const std::string &name) const
{
 static const double EPS=0.00001;

 printf("***** %s *****\r\n",name.c_str());
 for(size_t z=0;z<GetSizeZ();z++)
 {
  printf("-----%i-----\r\n",z);
  for(size_t y=0;y<GetSizeY();y++)
  {
   for(size_t x=0;x<GetSizeX();x++)
   {
    type_t e1=GetElement(z,y,x);
    type_t e2=cTensor_Control.GetElement(z,y,x);
    printf("%f[%f] ",e1,e2);
    if (fabs(e1-e2)>EPS) return(false);
   }
   printf("\r\n");
  }
 }
 printf("\r\n");
 return(true);
}


#endif
