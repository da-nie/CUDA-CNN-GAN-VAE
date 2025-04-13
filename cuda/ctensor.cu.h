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
  mutable size_t Size_X;///<размер по X
  mutable size_t Size_Y;///<размер по Y
  mutable size_t Size_Z;///<размер по Z

  mutable size_t BasedSize_X;///<исходный размер по X
  mutable size_t BasedSize_Y;///<исходный размер по Y
  mutable size_t BasedSize_Z;///<исходный размер по Z

  mutable std::vector<type_t> Item;///<массив компонентов тензора
  mutable CCUDADeviceVector<type_t> DeviceItem;///<данные на устройстве
  mutable bool HostOnChange;///<изменились данные на хосте
  mutable bool DeviceOnChange;///<изменились данные на устройстве

 public:
  //-конструктор----------------------------------------------------------------------------------------
  CTensor<type_t>(size_t size_z=1,size_t size_y=1,size_t size_x=1);
  //-конструктор копирования----------------------------------------------------------------------------
  CTensor<type_t>(const CTensor<type_t> &cTensor);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CTensor<type_t>();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ __device__ size_t GetSizeX(void) const;///<получить размер по x
  __host__ __device__ size_t GetSizeY(void) const;///<получить размер по y
  __host__ __device__ size_t GetSizeZ(void) const;///<получить размер по z
  type_t GetElement(size_t z,size_t y,size_t x) const;///<получить элемент тензора
  void SetElement(size_t z,size_t y,size_t x,type_t value);///<задать элемент тензора
  type_t* GetColumnPtr(size_t z,size_t y);///<получить указатель на строку тензора
  const type_t* GetColumnPtr(size_t z,size_t y) const;///<получить указатель на строку тензора
  CCUDADeviceVector<type_t>& GetDeviceVector(void) const;///<получить класс хранения данных на устройстве
  void Unitary(void);///<привести к единичному виду
  void Zero(void);///<обнулить тензор
  void Fill(type_t value);///<задать тензор числом
  void Move(CTensor<type_t> &cTensor);///<переместить тензор
  void CopyItem(CTensor<type_t> &cTensor);///<скопировать только элементы
  void CopyItemToHost(type_t *item_array,size_t size);///<скопировать элементы из массива в хост
  void CopyItemToDevice(type_t *item_array,size_t size);///<скопировать элементы из массива в устройство

  CTensor<type_t>& operator=(const CTensor<type_t> &cTensor);///<оператор "="

  friend CTensor<type_t> operator+<type_t>(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<оператор "+"
  friend CTensor<type_t> operator-<type_t>(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<оператор "-"

  friend CTensor<type_t> operator*<type_t>(const CTensor<type_t> &cTensor_Left,const CTensor<type_t> &cTensor_Right);///<оператор "*"

  friend CTensor<type_t> operator*<type_t>(const CTensor<type_t> &cTensor_Left,const type_t &value_right);///<оператор "*"
  friend CTensor<type_t> operator*<type_t>(const type_t &value_left,const CTensor<type_t> &cTensor_Right);///<оператор "*"

  bool Save(IDataStream *iDataStream_Ptr);///<сохранить тензор
  bool Load(IDataStream *iDataStream_Ptr);///<загрузить тензор

  void ExchangeSizeXY(void) const;///<обменять размеры по X и по Y местами
  void ReinterpretSize(size_t size_z,size_t size_y,size_t size_x) const;///<интерпретировать размер по-новому
  void RestoreSize(void) const;///<восстановить первоначальную интерпретацию размеров

  void Normalize(void);///<нормировка тензора
  type_t GetNorma(size_t z) const;///<получить норму тензора

  void CopyToDevice(bool force=false) const;///<скопировать тензор на устройство
  void CopyFromDevice(bool force=false) const;///<скопировать тензор с устройства
  void SetDeviceOnChange(void) const;///<установить, что данные на устройстве изменились
  void SetHostOnChange(void) const;///<установить, что данные изменились

  void Print(const std::string &name,bool print_value=true) const;///<вывод тензора на экран
  void PrintToFile(const std::string &file_name,const std::string &name,bool print_value=true) const;///<вывод тензора в файл
  bool Compare(const CTensor<type_t> &cTensor_Control,const std::string &name="",bool print_value=true) const;///<сравнение тензоровzz
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
 BasedSize_X=Size_X;
 BasedSize_Y=Size_Y;
 BasedSize_Z=Size_Z;

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
__host__ __device__ size_t CTensor<type_t>::GetSizeX(void) const
{
 return(Size_X);
}
//----------------------------------------------------------------------------------------------------
//получить размер по y
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ size_t CTensor<type_t>::GetSizeY(void) const
{
 return(Size_Y);
}
//----------------------------------------------------------------------------------------------------
//получить размер по z
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ size_t CTensor<type_t>::GetSizeZ(void) const
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
 CopyFromDevice();
 return(Item[Size_X*y+x+Size_X*Size_Y*z]);
}
//----------------------------------------------------------------------------------------------------
//задать элемент тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::SetElement(size_t z,size_t y,size_t x,type_t value)
{
 if (x>=Size_X || y>=Size_Y || z>=Size_Z) throw("Ошибка доступа к элементу тензора для записи!");
 CopyFromDevice();
 Item[Size_X*y+x+ Size_X*Size_Y*z]=value;
 SetHostOnChange();
}

//----------------------------------------------------------------------------------------------------
//получить указатель на строку тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t* CTensor<type_t>::GetColumnPtr(size_t z,size_t y)
{
 if (y>=Size_Y || z>=Size_Z) throw("Ошибка получения указателя на строку тензора!");
 CopyFromDevice();
 SetHostOnChange();
 return(&Item[Size_X*y+Size_X*Size_Y*z]);
}
//----------------------------------------------------------------------------------------------------
//получить указатель на строку тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
const type_t* CTensor<type_t>::GetColumnPtr(size_t z,size_t y) const
{
 if (y>=Size_Y || z>=Size_Z) throw("Ошибка получения указателя на строку тензора!");
 CopyFromDevice();
 SetHostOnChange();
 return(&Item[Size_X*y+Size_X*Size_Y*z]);
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
 SetHostOnChange();
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
 SetHostOnChange();
}

//----------------------------------------------------------------------------------------------------
//задать тензор числом
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Fill(type_t value)
{
 type_t *o_ptr=&Item[0];
 for(size_t z=0;z<Size_Z;z++)
 {
  for(size_t y=0;y<Size_Y;y++)
  {
   for(size_t x=0;x<Size_X;x++,o_ptr++)
   {
    *o_ptr=value;
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
 BasedSize_X=cTensor.BasedSize_X;
 BasedSize_Y=cTensor.BasedSize_Y;
 BasedSize_Z=cTensor.BasedSize_Z;

 HostOnChange=cTensor.HostOnChange;
 DeviceOnChange=cTensor.DeviceOnChange;

 cTensor.Size_X=0;
 cTensor.Size_Y=0;
 cTensor.Size_Z=0;

 cTensor.BasedSize_X=0;
 cTensor.BasedSize_Y=0;
 cTensor.BasedSize_Z=0;

 cTensor.HostOnChange=false;
 cTensor.DeviceOnChange=false;
}
//----------------------------------------------------------------------------------------------------
//скопировать только элементы
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyItem(CTensor<type_t> &cTensor)
{
 if (Size_X*Size_Y*Size_Z!=cTensor.GetSizeX()*cTensor.GetSizeY()*cTensor.GetSizeZ()) throw("Нельзя копировать элементы тензора, если их количество различно.");
 cTensor.CopyFromDevice();
 Item=cTensor.Item;
 SetHostOnChange();
}

//----------------------------------------------------------------------------------------------------
//скопировать элементы из массива в хост
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyItemToHost(type_t *item_array,size_t size)
{
 if (Item.size()<size) throw("Слишком большой массив для копирования на хост.");
 memcpy(&Item[0],item_array,size*sizeof(type_t));
 SetHostOnChange();
}
//----------------------------------------------------------------------------------------------------
//скопировать элементы из массива в устройство
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::CopyItemToDevice(type_t *item_array,size_t size)
{
 if (Item.size()<size) throw("Слишком большой массив для копирования на устройство.");
 DeviceItem.copy_host_to_device(item_array,size);
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

  BasedSize_X=cTensor.BasedSize_X;
  BasedSize_Y=cTensor.BasedSize_Y;
  BasedSize_Z=cTensor.BasedSize_Z;

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

 SetHostOnChange();
 return(true);
}

//----------------------------------------------------------------------------------------------------
//обменять размеры по X и по Y местами
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::ExchangeSizeXY(void) const
{
 size_t tmp=Size_X;
 Size_X=Size_Y;
 Size_Y=tmp;
}
//----------------------------------------------------------------------------------------------------
//интерпретировать размер по-новому
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::ReinterpretSize(size_t size_z,size_t size_y,size_t size_x) const
{
 static const size_t STRING_BUFFER_SIZE=1024;

 if (Size_X*Size_Y*Size_Z!=size_x*size_y*size_z)
 {
  char str[STRING_BUFFER_SIZE];
  sprintf(str,"%ix%ix%i->%ix%ix%i",static_cast<int>(Size_Z),static_cast<int>(Size_Y),static_cast<int>(Size_X),static_cast<int>(size_z),static_cast<int>(size_y),static_cast<int>(size_x));
  SYSTEM::PutMessageToConsole(str);
  throw("Новая интерпретация размеров невозможна из-за различного количества элементов.");
 }
 Size_X=size_x;
 Size_Y=size_y;
 Size_Z=size_z;
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
}


//----------------------------------------------------------------------------------------------------
//нормировка тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CTensor<type_t>::Normalize(void)
{
 CopyFromDevice();

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
 SetHostOnChange();
}
//----------------------------------------------------------------------------------------------------
//получить норму тензора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CTensor<type_t>::GetNorma(size_t z) const
{
 CopyFromDevice();
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
 sprintf(str,"Z:%i Y:%i X:%i",static_cast<int>(Size_Z),static_cast<int>(Size_Y),static_cast<int>(Size_X));
 SYSTEM::PutMessageToConsole(str);
 if (print_value==false) return;
 for(size_t z=0;z<GetSizeZ();z++)
 {
  sprintf(str,"-----%i-----",static_cast<int>(z));
  SYSTEM::PutMessageToConsole(str);
  for(size_t y=0;y<GetSizeY();y++)
  {
   std::string line;
   for(size_t x=0;x<GetSizeX();x++)
   {
    type_t e1=GetElement(z,y,x);
    sprintf(str,"%f ",e1);
	line+=str;
   }
   SYSTEM::PutMessageToConsole(line);
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
 fprintf(file,"Z:%i Y:%i X:%i\r\n",static_cast<int>(Size_Z),static_cast<int>(Size_Y),static_cast<int>(Size_X));
 if (print_value==false) return;
 for(size_t z=0;z<GetSizeZ();z++)
 {
  fprintf(file,"-----%i-----\r\n",static_cast<int>(z));
  for(size_t y=0;y<GetSizeY();y++)
  {
   for(size_t x=0;x<GetSizeX();x++)
   {
    type_t e1=GetElement(z,y,x);
    fprintf(file,"%f ",e1);
   }
   fprintf(file,"\r\n");
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
 sprintf(str,"Z:%i Y:%i X:%i",static_cast<int>(Size_Z),static_cast<int>(Size_Y),static_cast<int>(Size_X));
 if (print_value==true) SYSTEM::PutMessageToConsole(str);
 for(size_t z=0;z<GetSizeZ();z++)
 {
  sprintf(str,"-----%i-----",static_cast<int>(z));
  if (print_value==true) SYSTEM::PutMessageToConsole(str);
  for(size_t y=0;y<GetSizeY();y++)
  {
   std::string line;
   for(size_t x=0;x<GetSizeX();x++)
   {
    type_t e1=GetElement(z,y,x);
    type_t e2=cTensor_Control.GetElement(z,y,x);
    sprintf(str,"%f[%f] ",e1,e2);
	line+=str;
    if (fabs(e1-e2)>EPS) ret=false;
   }
   if (print_value==true) SYSTEM::PutMessageToConsole(line);
  }
 }
 if (print_value==true) SYSTEM::PutMessageToConsole("");
 return(ret);
}

#endif

#endif
