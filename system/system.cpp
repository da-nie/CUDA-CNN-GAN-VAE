//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "system.h"

#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/times.h>
#include <unistd.h>

//****************************************************************************************************
//реализация функций
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//создать список файлов директории
//----------------------------------------------------------------------------------------------------
void SYSTEM::CreateFileList(const std::string &path,std::vector<std::string> &vector_file_name)
{
 vector_file_name.clear();
 //сканируем файлы каталога
 DIR *dir;
 dirent *d_ptr;
 dir=opendir(path.c_str());
 if (dir!=NULL)
 {
  while(1)
  {
   d_ptr=readdir(dir);
   if (d_ptr==NULL) break;
   long len=strlen(d_ptr->d_name);
   if (len==1)
   {
    if (d_ptr->d_name[len-1]=='.') continue;
   }
   if (len==2)
   {
    if (d_ptr->d_name[len-1]=='.' && d_ptr->d_name[len-2]=='.') continue;
   }
   std::string filename=path;
   filename+=GetPathDivider();
   filename+=d_ptr->d_name;

   DIR *dir_tmp;
   dirent *d_tmp_ptr;
   dir_tmp=opendir(filename.c_str());
   if (dir_tmp!=NULL)//это каталог, а не файл
   {
    closedir(dir_tmp);
    continue;
   }
   //vector_file_name.push_back(filename);
   vector_file_name.push_back(d_ptr->d_name);
  }
  closedir(dir);
 }
}

//----------------------------------------------------------------------------------------------------
//создать список каталогов директории
//----------------------------------------------------------------------------------------------------
void SYSTEM::CreateDirectoryList(const std::string &path,std::vector<std::string> &vector_directory_name)
{
 vector_directory_name.clear();
 //сканируем директории каталога
 DIR *dir;
 dirent *d_ptr;
 dir=opendir(path.c_str());
 if (dir!=NULL)
 {
  while(1)
  {
   d_ptr=readdir(dir);
   if (d_ptr==NULL) break;
   long len=strlen(d_ptr->d_name);
   if (len==1)
   {
    if (d_ptr->d_name[len-1]=='.') continue;
   }
   if (len==2)
   {
    if (d_ptr->d_name[len-1]=='.' && d_ptr->d_name[len-2]=='.') continue;
   }
   std::string filename=path;
   filename+=GetPathDivider();
   filename+=d_ptr->d_name;

   DIR *dir_tmp;
   dirent *d_tmp_ptr;
   dir_tmp=opendir(filename.c_str());
   if (dir_tmp==NULL) continue;//это файл,а не каталог
   closedir(dir_tmp);
   //vector_directory_name.push_back(filename);
   vector_directory_name.push_back(d_ptr->d_name);
  }
  closedir(dir);
 }
}

//----------------------------------------------------------------------------------------------------
//создать каталог
//----------------------------------------------------------------------------------------------------
void SYSTEM::MakeDirectory(const std::string &directory_name)
{
 mkdir(directory_name.c_str(),S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH |S_IXOTH);
}

//----------------------------------------------------------------------------------------------------
//получить прошедшее время в секундах
//----------------------------------------------------------------------------------------------------
long double SYSTEM::GetSecondCounter(void)
{
 const long double ticks=static_cast<long double>(sysconf(_SC_CLK_TCK));
 struct tms tms;
 if (times(&tms)!=(clock_t)-1) return(static_cast<long double>(tms.tms_utime)/ticks);
 return(0);
}
//----------------------------------------------------------------------------------------------------
//пауза в миллисекундах
//----------------------------------------------------------------------------------------------------
void SYSTEM::PauseInMs(size_t ms)
{
 usleep(ms*1000);
}
//----------------------------------------------------------------------------------------------------
//пауза в микросекундах
//----------------------------------------------------------------------------------------------------
void SYSTEM::PauseInUs(size_t us)
{
 usleep(us);
}
//----------------------------------------------------------------------------------------------------
//получить текущую директорию
//----------------------------------------------------------------------------------------------------
std::string SYSTEM::GetCurrentPath(void)
{
 return(std::string("."));
}
//----------------------------------------------------------------------------------------------------
//получить разделитель каталого
//----------------------------------------------------------------------------------------------------
std::string SYSTEM::GetPathDivider(void)
{
 return(std::string("/"));
}
//----------------------------------------------------------------------------------------------------
//вывести сообщение
//----------------------------------------------------------------------------------------------------
void SYSTEM::PutMessage(const std::string &message)
{
 printf("%s\r\n",message.c_str());
}
//----------------------------------------------------------------------------------------------------
//вывести сообщение в консоль
//----------------------------------------------------------------------------------------------------
void SYSTEM::PutMessageToConsole(const std::string &message)
{
 printf("%s\r\n",message.c_str());
}
//----------------------------------------------------------------------------------------------------
//скопировать файл
//----------------------------------------------------------------------------------------------------
void SYSTEM::CopyFile(const std::string &source_file,const std::string &target_file)
{
 /*
 std::string new_source_file=ConvertFileName(source_file);
 std::string new_target_file=ConvertFileName(target_file);

 std::string command="cp -p \"";
 command+=new_source_file;
 command+="\" \"";
 command+=new_target_file;
 command+="\"";
 printf("command:%s\r\n",command.c_str());
 system(command.c_str());
 */

 FILE *file_one=fopen(source_file.c_str(),"rb");
 if (file_one==NULL) return;
 FILE *file_two=fopen(target_file.c_str(),"wb");
 if (file_two==NULL)
 {
  fclose(file_one);
  return;
 }
 printf("Copy file:%s -> %s\r\n",source_file.c_str(),target_file.c_str());

 uint8_t buffer[65536];
 while(1)
 {
  size_t size=fread(buffer,sizeof(uint8_t),65535,file_one);
  if (size==0) break;
  fwrite(buffer,sizeof(uint8_t),size,file_two);
 }
 fclose(file_one);
 fclose(file_two);
}

//----------------------------------------------------------------------------------------------------
//перенести файл
//----------------------------------------------------------------------------------------------------
void SYSTEM::MoveFileTo(const std::string &source_file,const std::string &target_file)
{
 FILE *file_one=fopen(source_file.c_str(),"rb");
 if (file_one==NULL) return;
 FILE *file_two=fopen(target_file.c_str(),"wb");
 if (file_two==NULL)
 {
  fclose(file_one);
  return;
 }
 printf("Move file:%s -> %s\r\n",source_file.c_str(),target_file.c_str());

 uint8_t buffer[65536];
 while(1)
 {
  size_t size=fread(buffer,sizeof(uint8_t),65535,file_one);
  if (size==0) break;
  fwrite(buffer,sizeof(uint8_t),size,file_two);
 }
 fclose(file_one);
 fclose(file_two);
 //удаляем файл
 //remove(source_file.c_str());
}
