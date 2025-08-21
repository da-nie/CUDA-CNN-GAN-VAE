#ifndef C_TENSOR_CU_H
#define C_TENSOR_CU_H

#include "../settings.h"

#ifdef USE_CPU

#include "../cpu/ctensor.h"

#endif


#ifndef USE_CPU

//****************************************************************************************************
//Класс тензоров произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <vector>
#include <math.h>
#include <memory>

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
 friend struct STensorTransponseKernel;

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
  mutable uint32_t Size_X;///<размер по X
  mutable uint32_t Size_Y;///<размер по Y
  mutable uint32_t Size_Z;///<размер по Z
  mutable uint32_t Size_W;///<размер по W

  mutable uint32_t BasedSize_X;///<исходный размер по X
  mutable uint32_t BasedSize_Y;///<исходный размер по Y
  mutable uint32_t BasedSize_Z;///<исходный размер по Z
  mutable uint32_t BasedSize_W;///<исходный размер по W

  mutable std::vector<type_t> Item;///<массив компонентов тензора
  mutable CCUDADeviceVector<type_t> DeviceItem;///<данные на устройстве
  mutable bool HostOnChange;///<изменились данные на хосте
  mutable bool DeviceOnChange;///<изменились данные на устройстве

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CTensor<type_t>(uint32_t size_w=1,uint32_t size_z=1,uint32_t size_y=1,uint32_t size_x=1);
  //-конструктор копирования----------------------------------------------------------------------------
  CTensor<type_t>(const CTensor<type_t> &cTensor);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CTensor<type_t>();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ __device__ uint32_t GetSizeX(void) const;///<получить размер по x
  __host__ __device__ uint32_t GetSizeY(void) const;///<получить размер по y
  __host__ __device__ uint32_t GetSizeZ(void) const;///<получить размер по z
  __host__ __device__ uint32_t GetSizeW(void) const;///<получить размер по w
  type_t GetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x) const;///<получить элемент тензора
  void SetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x,type_t value);///<задать элемент тензора
  type_t* GetColumnPtr(uint32_t w,uint32_t z,uint32_t y);///<получить указатель на строку тензора
  const type_t* GetColumnPtr(uint32_t w,uint32_t z,uint32_t y) const;///<получить указатель на строку тензора
  CCUDADeviceVector<type_t>& GetDeviceVector(void) const;///<получить класс хранения данных на устройстве
  void Unitary(void);///<привести к единичному виду
  void Zero(void);///<обнулить тензор
  void Fill(type_t value);///<задать тензор числом
  void Move(CTensor<type_t> &cTensor);///<переместить тензор
  void CopyItem(CTensor<type_t> &cTensor);///<скопировать только элементы
  void CopyItemToHost(type_t *item_array,uint32_t size);///<скопировать элементы из массива в хост
  void CopyItemToDevice(type_t *item_array,uint32_t size);///<скопировать элементы из массива в устройство

  void CopyItemLayerWToHost(uint32_t w,type_t *item_array,uint32_t size);///<скопировать элементы из массива в хост слоя w
  void CopyItemLayerWToDevice(uint32_t w,type_t *item_array,uint32_t size);///<скопировать элементы из массива в устройство слоя w
  CTensor<type_t>& operator=(const CTensor<type_t> &cTensor);///<оператор "="

  friend CTensor<type_t> operator+<type_t>(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<оператор "+"
  friend CTensor<type_t> operator-<type_t>(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<оператор "-"

  friend CTensor<type_t> operator*<type_t>(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<оператор "*"

  friend CTensor<type_t> operator*<type_t>(const CTensor<type_t> &cTensor_Left,const type_t &value_right);///<оператор "*"
  friend CTensor<type_t> operator*<type_t>(const type_t &value_left,const CTensor<type_t> &cTensor_Right);///<оператор "*"

  bool Save(IDataStream *iDataStream_Ptr);///<сохранить тензор
  bool Load(IDataStream *iDataStream_Ptr,bool check_size=false);///<загрузить тензор

  void ExchangeSizeXY(void) const;///<обменять размеры по X и по Y местами
  void ReinterpretSize(uint32_t size_w,uint32_t size_z,uint32_t size_y,uint32_t size_x) const;///<интерпретировать размер по-новому
  void RestoreSize(void) const;///<восстановить первоначальную интерпретацию размеров

  void Normalize(void);///<нормировка тензора
  type_t GetNorma(uint32_t w,uint32_t z) const;///<получить норму тензора

  void CopyToDevice(bool force=false) const;///<скопировать тензор на устройство
  void CopyFromDevice(bool force=false) const;///<скопировать тензор с устройства
  void SetDeviceOnChange(void) const;///<установить, что данные на устройстве изменились
  void SetHostOnChange(void) const;///<установить, что данные изменились

  void Print(const std::string &name,bool print_value=true) const;///<вывод тензора на экран
  void PrintToFile(const std::string &file_name,const std::string &name,bool print_value=true) const;///<вывод тензора в файл
  bool Compare(const CTensor<type_t> &cTensor_Control,const std::string &name="",bool print_value=true) const;///<сравнение тензоров
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
CTensor<type_t>::CTensor(uint32_t size_w,uint32_t size_z,uint32_t size_y,uint32_t size_x)
{
 Size_X=size_x;
 Size_Y=size_y;
 Size_Z=size_z;
 Size_W=size_w;

 BasedSize_X=Size_X;
 BasedSize_Y=Size_Y;
 BasedSize_Z=Size_Z;
 BasedSize_W=Size_W;

 Item.resize(Size_X*Size_Y*Size_Z*Size_W);
 DeviceItem.resize(Size_X*Size_Y*Size_Z*Size_W);

 HostOnChange=false;
 DeviceOnChange=false;
}
//----------------------------------------------------------------------------------------------------
//конструктор копирования
//----------------------------------------------------------------------------------------------------
template<class type_t>
CTensor<type_t>::CTensor(const CTensor<type_t> &cTensor)
{
 if (&cTensor==this) return;
 Item=cTensor.Item;
 DeviceItem=cTensor.DeviceItem;

 Size_X=cTensor.Size_X;
 Size_Y=cTensor.Size_Y;
 Size_Z=cTensor.Size_Z;
 Size_W=cTensor.Size_W;
 BasedSize_X=Size_X;
 BasedSize_Y=Size_Y;
 BasedSize_Z=Size_Z;
 BasedSize_W=Size_W;

 HostOnChange=cTensor.HostOnChange;
 DeviceOnChange=cTensor.DeviceOnChange;
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
 Size_W=0;
 BasedSize_X=Size_X;
 BasedSize_Y=Size_Y;
 BasedSize_Z=Size_Z;
 BasedSize_W=Size_W;
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
__host__ __device__ uint32_t CTensor<type_t>::GetSizeX(void) const
{
 return(Size_X);
}
//----------------------------------------------------------------------------------------------------
//получить размер по y
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ uint32_t CTensor<type_t>::GetSizeY(void) const
{
 return(Size_Y);
}
//----------------------------------------------------------------------------------------------------
//получить размер по z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ uint32_t CTensor<type_t>::GetSizeZ(void) const
{
 return(Size_Z);
}
//----------------------------------------------------------------------------------------------------
//получить размер по w
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ uint32_t CTensor<type_t>::GetSizeW(void) const
{
 return(Size_W);
}
//----------------------------------------------------------------------------------------------------
//получить элемент тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensor<type_t>::GetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x) const
{
 if (x>=Size_X || y>=Size_Y || z>=Size_Z || w>=Size_W) throw("Ошибка доступа к элементу тензора для чтения!");
 CopyFromDevice();
 return(Item[Size_X*y+x+Size_X*Size_Y*z+Size_X*Size_Y*Size_Z*w]);
}
//----------------------------------------------------------------------------------------------------
//задать элемент тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::SetElement(uint32_t w,uint32_t z,uint32_t y,uint32_t x,type_t value)
{
 if (x>=Size_X || y>=Size_Y || z>=Size_Z || w>=Size_W) throw("Ошибка доступа к элементу тензора для записи!");
 CopyFromDevice();
 Item[Size_X*y+x+Size_X*Size_Y*z+Size_X*Size_Y*Size_Z*w]=value;
 SetHostOnChange();
}

//----------------------------------------------------------------------------------------------------
//получить указатель на строку тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t* CTensor<type_t>::GetColumnPtr(uint32_t w,uint32_t z,uint32_t y)
{
 if (y>=Size_Y || z>=Size_Z || w>=Size_W) throw("Ошибка получения указателя на строку тензора!");
 CopyFromDevice();
 SetHostOnChange();
 return(&Item[Size_X*y+Size_X*Size_Y*z+Size_X*Size_Y*Size_Z*w]);
}
//----------------------------------------------------------------------------------------------------
//получить указатель на строку тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
const type_t* CTensor<type_t>::GetColumnPtr(uint32_t w,uint32_t z,uint32_t y) const
{
 if (y>=Size_Y || z>=Size_Z || w>=Size_W) throw("Ошибка получения указателя на строку тензора!");
 CopyFromDevice();
 SetHostOnChange();
 return(&Item[Size_X*y+Size_X*Size_Y*z+Size_X*Size_Y*Size_Z*w]);
}
//----------------------------------------------------------------------------------------------------
///!получить класс хранения данных на устройстве
//----------------------------------------------------------------------------------------------------
template<class type_t>
CCUDADeviceVector<type_t>& CTensor<type_t>::GetDeviceVector(void) const
{
 return(DeviceItem);
}

//----------------------------------------------------------------------------------------------------
//привести к единичному виду
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Unitary(void)
{
 type_t *o_ptr=&Item[0];
 for(uint32_t w=0;w<Size_W;w++)
 {
  for(uint32_t z=0;z<Size_Z;z++)
  {
   for(uint32_t y=0;y<Size_Y;y++)
   {
    for(uint32_t x=0;x<Size_X;x++,o_ptr++)
    {
     if (x==y) *o_ptr=1;
          else *o_ptr=0;
    }
   }
  }
 }
 SetHostOnChange();
}
//----------------------------------------------------------------------------------------------------
//обнулить тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Zero(void)
{
 type_t *o_ptr=&Item[0];
 for(uint32_t w=0;w<Size_W;w++)
 {
  for(uint32_t z=0;z<Size_Z;z++)
  {
   for(uint32_t y=0;y<Size_Y;y++)
   {
    for(uint32_t x=0;x<Size_X;x++,o_ptr++)
    {
     *o_ptr=0;
    }
   }
  }
 }
 SetHostOnChange();
}

//----------------------------------------------------------------------------------------------------
//задать тензор числом
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Fill(type_t value)
{
 type_t *o_ptr=&Item[0];
 for(uint32_t w=0;w<Size_W;w++)
 {
  for(uint32_t z=0;z<Size_Z;z++)
  {
   for(uint32_t y=0;y<Size_Y;y++)
   {
    for(uint32_t x=0;x<Size_X;x++,o_ptr++)
    {
     *o_ptr=value;
    }
   }
  }
 }
 SetHostOnChange();
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
 Size_W=cTensor.Size_W;
 BasedSize_X=cTensor.BasedSize_X;
 BasedSize_Y=cTensor.BasedSize_Y;
 BasedSize_Z=cTensor.BasedSize_Z;
 BasedSize_W=cTensor.BasedSize_W;

 HostOnChange=cTensor.HostOnChange;
 DeviceOnChange=cTensor.DeviceOnChange;

 cTensor.Size_X=0;
 cTensor.Size_Y=0;
 cTensor.Size_Z=0;
 cTensor.Size_W=0;

 cTensor.BasedSize_X=0;
 cTensor.BasedSize_Y=0;
 cTensor.BasedSize_Z=0;
 cTensor.BasedSize_W=0;

 cTensor.HostOnChange=false;
 cTensor.DeviceOnChange=false;
}
//----------------------------------------------------------------------------------------------------
//скопировать только элементы
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyItem(CTensor<type_t> &cTensor)
{
 if (Size_X*Size_Y*Size_Z*Size_W!=cTensor.GetSizeX()*cTensor.GetSizeY()*cTensor.GetSizeZ()*cTensor.GetSizeW()) throw("Нельзя копировать элементы тензора, если их количество различно.");
 cTensor.CopyFromDevice();
 Item=cTensor.Item;
 SetHostOnChange();
}

//----------------------------------------------------------------------------------------------------
//скопировать элементы из массива в хост
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyItemToHost(type_t *item_array,uint32_t size)
{
 if (Item.size()<size) throw("Слишком большой массив для копирования на хост.");
 memcpy(&Item[0],item_array,size*sizeof(type_t));
 SetHostOnChange();
}
//----------------------------------------------------------------------------------------------------
//скопировать элементы из массива в устройство
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyItemToDevice(type_t *item_array,uint32_t size)
{
 if (Item.size()<size) throw("Слишком большой массив для копирования на устройство.");
 DeviceItem.copy_host_to_device(item_array,size);
 SetDeviceOnChange();
}


//----------------------------------------------------------------------------------------------------
//скопировать элементы из массива в хост слоя w
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyItemLayerWToHost(uint32_t w,type_t *item_array,uint32_t size)
{
 if (Item.size()<size+w*Size_X*Size_Y*Size_Z) throw("Слишком большой массив для копирования на хост.");
 memcpy(&Item[w*Size_X*Size_Y*Size_Z],item_array,size*sizeof(type_t));
 SetHostOnChange();
}
//----------------------------------------------------------------------------------------------------
//скопировать элементы из массива в устройство слоя w
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyItemLayerWToDevice(uint32_t w,type_t *item_array,uint32_t size)
{
 if (Item.size()<size+w*Size_X*Size_Y*Size_Z) throw("Слишком большой массив для копирования на устройство.");
 DeviceItem.copy_host_to_device(item_array,size,w*Size_X*Size_Y*Size_Z);
 SetDeviceOnChange();
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
  Size_W=cTensor.Size_W;

  BasedSize_X=cTensor.BasedSize_X;
  BasedSize_Y=cTensor.BasedSize_Y;
  BasedSize_Z=cTensor.BasedSize_Z;
  BasedSize_W=cTensor.BasedSize_W;

  HostOnChange=cTensor.HostOnChange;
  DeviceOnChange=cTensor.DeviceOnChange;
 }
 return(*this);
}
//----------------------------------------------------------------------------------------------------
//сохранить тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensor<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 CopyFromDevice();

 //сохраняем размерность тензора
 //iDataStream_Ptr->SaveUInt32(Size_W);//TODO: тест
 iDataStream_Ptr->SaveUInt32(Size_Z);
 iDataStream_Ptr->SaveUInt32(Size_Y);
 iDataStream_Ptr->SaveUInt32(Size_X);
 //сохраняем данные тензора
 for(uint32_t n=0;n<Size_X*Size_Y*Size_Z*Size_W;n++) iDataStream_Ptr->SaveDouble(Item[n]);
 return(true);
}
//----------------------------------------------------------------------------------------------------
//загрузить тензор
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensor<type_t>::Load(IDataStream *iDataStream_Ptr,bool check_size)
{
 //загружаем размерность тензора

 if (check_size==true)
 {
  //if (Size_W!=iDataStream_Ptr->LoadUInt32()) throw("Ошибка загрузки тензора: неверный размер W.");//TODO: тест
  if (Size_Z!=iDataStream_Ptr->LoadUInt32()) throw("Ошибка загрузки тензора: неверный размер Z.");
  if (Size_Y!=iDataStream_Ptr->LoadUInt32()) throw("Ошибка загрузки тензора: неверный размер Y.");
  if (Size_X!=iDataStream_Ptr->LoadUInt32()) throw("Ошибка загрузки тензора: неверный размер X.");
 }
 else
 {
  //Size_W=iDataStream_Ptr->LoadUInt32();//TODO: тест
  Size_Z=iDataStream_Ptr->LoadUInt32();
  Size_Y=iDataStream_Ptr->LoadUInt32();
  Size_X=iDataStream_Ptr->LoadUInt32();
 }

 std::vector<type_t> item(Size_X*Size_Y*Size_Z*Size_W);
 Item.clear();
 std::swap(Item,item);
 DeviceItem.resize(Item.size());

 //загружаем данные тензора
 for(uint32_t n=0;n<Size_X*Size_Y*Size_Z*Size_W;n++) Item[n]=static_cast<type_t>(iDataStream_Ptr->LoadDouble());

 SetHostOnChange();
 return(true);
}

//----------------------------------------------------------------------------------------------------
//обменять размеры по X и по Y местами
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::ExchangeSizeXY(void) const
{
 uint32_t tmp=Size_X;
 Size_X=Size_Y;
 Size_Y=tmp;
}
//----------------------------------------------------------------------------------------------------
//интерпретировать размер по-новому
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::ReinterpretSize(uint32_t size_w,uint32_t size_z,uint32_t size_y,uint32_t size_x) const
{
 static const uint32_t STRING_BUFFER_SIZE=1024;
 if (Size_X*Size_Y*Size_Z*Size_W!=size_x*size_y*size_z*size_w)
 {
  char str[STRING_BUFFER_SIZE];
  sprintf(str,"%ix%ix%ix%i->%ix%ix%ix%i",static_cast<int>(Size_W),static_cast<int>(Size_Z),static_cast<int>(Size_Y),static_cast<int>(Size_X),static_cast<int>(size_w),static_cast<int>(size_z),static_cast<int>(size_y),static_cast<int>(size_x));
  SYSTEM::PutMessageToConsole(str);
  throw("Новая интерпретация размеров невозможна из-за различного количества элементов.");
 }
 Size_X=size_x;
 Size_Y=size_y;
 Size_Z=size_z;
 Size_W=size_w;
}
//----------------------------------------------------------------------------------------------------
//восстановить первоначальную интерпретацию размеров
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::RestoreSize(void) const
{
 Size_X=BasedSize_X;
 Size_Y=BasedSize_Y;
 Size_Z=BasedSize_Z;
 Size_W=BasedSize_W;
}


//----------------------------------------------------------------------------------------------------
//нормировка тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Normalize(void)
{
 CopyFromDevice();

 for(uint32_t w=0;w<Size_W;w++)
 {
  for(uint32_t z=0;z<Size_Z;z++)
  {
   type_t norma=GetNorma(w,z);
   if (norma<CTENSOR_EPS) continue;
   for(uint32_t y=0;y<Size_Y;y++)
   {
    for(uint32_t x=0;x<Size_X;x++)
    {
     Item[Size_X*y+x+z*Size_X*Size_Y+w*Size_X*Size_Y*Size_Z]/=norma;
    }
   }
  }
 }
 SetHostOnChange();
}
//----------------------------------------------------------------------------------------------------
//получить норму тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensor<type_t>::GetNorma(uint32_t w,uint32_t z) const
{
 CopyFromDevice();
 type_t norma=0;
 for(uint32_t y=0;y<Size_Y;y++)
 {
  for(uint32_t x=0;x<Size_X;x++)
  {
   type_t v=Item[Size_X*y+x+z*Size_X*Size_Y+w*Size_X*Size_Y*Size_Z];
   norma+=v*v;
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
 //копируем данные хоста на устройство
 if (HostOnChange==false && force==false) return;
 DeviceItem.copy_host_to_device(&Item[0],Item.size());
 //указываем, что копии идентичные
 HostOnChange=false;
 DeviceOnChange=false;
}
//----------------------------------------------------------------------------------------------------
///!скопировать тензор с устройства
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyFromDevice(bool force) const
{
 //копируем данные с устройства на хост
 if (DeviceOnChange==false && force==false) return;
 DeviceItem.copy_device_to_host(&Item[0],Item.size());
 //указываем, что копии идентичные
 HostOnChange=false;
 DeviceOnChange=false;
}

//----------------------------------------------------------------------------------------------------
///!установить, что данные хоста изменились
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::SetHostOnChange(void) const
{
 DeviceOnChange=false;
 HostOnChange=true;
}
//----------------------------------------------------------------------------------------------------
///!установить, что данные устройства изменились
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::SetDeviceOnChange(void) const
{
 DeviceOnChange=true;
 HostOnChange=false;
}

//----------------------------------------------------------------------------------------------------
///!вывод тензора на экран
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Print(const std::string &name,bool print_value) const
{
 char str[255];

 CopyFromDevice();

 sprintf(str,"***** %s *****",name.c_str());
 SYSTEM::PutMessageToConsole(str);
 sprintf(str,"W:%i Z:%i Y:%i X:%i",static_cast<int>(Size_W),static_cast<int>(Size_Z),static_cast<int>(Size_Y),static_cast<int>(Size_X));
 SYSTEM::PutMessageToConsole(str);
 if (print_value==false) return;
 for(uint32_t w=0;w<GetSizeW();w++)
 {
  sprintf(str,"==========%i==========",static_cast<int>(w));
  SYSTEM::PutMessageToConsole(str);
  for(uint32_t z=0;z<GetSizeZ();z++)
  {
   sprintf(str,"-----%i-----",static_cast<int>(z));
   SYSTEM::PutMessageToConsole(str);
   for(uint32_t y=0;y<GetSizeY();y++)
   {
    std::string line;
    for(uint32_t x=0;x<GetSizeX();x++)
    {
     type_t e1=GetElement(w,z,y,x);
     sprintf(str,"%f ",e1);
 	line+=str;
    }
    SYSTEM::PutMessageToConsole(line);
   }
  }
 }
 SYSTEM::PutMessageToConsole("");
}
//----------------------------------------------------------------------------------------------------
///!вывод тензора в файл
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::PrintToFile(const std::string &file_name,const std::string &name,bool print_value) const
{
 CopyFromDevice();
 FILE *file=fopen(file_name.c_str(),"wb");
 if (file==NULL) return;

 fprintf(file,"***** %s *****\r\n",name.c_str());
 fprintf(file,"W:%i Z:%i Y:%i X:%i\r\n",static_cast<int>(Size_W),static_cast<int>(Size_Z),static_cast<int>(Size_Y),static_cast<int>(Size_X));
 if (print_value==false) return;
 for(uint32_t w=0;w<GetSizeW();w++)
 {
  fprintf(file,"==========%i==========",static_cast<int>(w));
  for(uint32_t z=0;z<GetSizeZ();z++)
  {
   fprintf(file,"-----%i-----\r\n",static_cast<int>(z));
   for(uint32_t y=0;y<GetSizeY();y++)
   {
    for(uint32_t x=0;x<GetSizeX();x++)
    {
     type_t e1=GetElement(w,z,y,x);
     fprintf(file,"%f ",e1);
    }
    fprintf(file,"\r\n");
   }
  }
 }
 fclose(file);
}
//----------------------------------------------------------------------------------------------------
///!сравнение тензоров
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CTensor<type_t>::Compare(const CTensor<type_t> &cTensor_Control,const std::string &name,bool print_value) const
{
 char str[255];

 CopyFromDevice();
 cTensor_Control.CopyFromDevice();

 static const double EPS=0.00001;

 bool ret=true;

 sprintf(str,"***** %s *****",name.c_str());
 if (print_value==true) SYSTEM::PutMessageToConsole(str);
 sprintf(str,"W:%i Z:%i Y:%i X:%i",static_cast<int>(Size_W),static_cast<int>(Size_Z),static_cast<int>(Size_Y),static_cast<int>(Size_X));
 if (print_value==true) SYSTEM::PutMessageToConsole(str);
 for(uint32_t w=0;w<GetSizeW();w++)
 {
  sprintf(str,"==========%i==========",static_cast<int>(w));
  if (print_value==true) SYSTEM::PutMessageToConsole(str);
  for(uint32_t z=0;z<GetSizeZ();z++)
  {
   sprintf(str,"-----%i-----",static_cast<int>(z));
   if (print_value==true) SYSTEM::PutMessageToConsole(str);
   for(uint32_t y=0;y<GetSizeY();y++)
   {
    std::string line;
    for(uint32_t x=0;x<GetSizeX();x++)
    {
     type_t e1=GetElement(w,z,y,x);
     type_t e2=cTensor_Control.GetElement(w,z,y,x);
     sprintf(str,"%f[%f] ",e1,e2);
	 line+=str;
     if (fabs(e1-e2)>EPS) ret=false;
    }
    if (print_value==true) SYSTEM::PutMessageToConsole(line);
   }
  }
 }
 if (print_value==true) SYSTEM::PutMessageToConsole("");
 return(ret);
}

#endif

#endif
