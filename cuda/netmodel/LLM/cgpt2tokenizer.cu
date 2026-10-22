//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "cgpt2tokenizer.cu.h"

//****************************************************************************************************
//глобальные переменные
//****************************************************************************************************

//****************************************************************************************************
//константы
//****************************************************************************************************

//****************************************************************************************************
//макроопределения
//****************************************************************************************************

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//!конструктор
//----------------------------------------------------------------------------------------------------
CGPT2Tokenizer::CGPT2Tokenizer(void)
{
 Specials.push_back(std::string("'s"));
 Specials.push_back(std::string("'t"));
 Specials.push_back(std::string("'re"));
 Specials.push_back(std::string("'ve"));
 Specials.push_back(std::string("'m"));
 Specials.push_back(std::string("'ll"));
 Specials.push_back(std::string("'d"));
}
//----------------------------------------------------------------------------------------------------
//!деструктор
//----------------------------------------------------------------------------------------------------
CGPT2Tokenizer::~CGPT2Tokenizer()
{
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
//конвертация Unicode code point в UTF-8 строку
//----------------------------------------------------------------------------------------------------
std::string CGPT2Tokenizer::UnicodeToUTF8(int32_t cp)
{
 std::string out;
 if (cp<=0x7F) out+=(char)cp;
 else
 {
  if (cp<=0x7FF)
  {
   out+=(char)(0xC0|(cp>>6));
   out+=(char)(0x80|(cp&0x3F));
  }
  else
  {
   if (cp<=0xFFFF)
   {
    out+=(char)(0xE0|(cp>>12));
    out+=(char)(0x80|((cp>>6)&0x3F));
    out+=(char)(0x80|(cp&0x3F));
   }
   else
   {
    if (cp<=0x10FFFF)
    {
     out+=(char)(0xF0|(cp>>18));
     out+=(char)(0x80|((cp>>12)&0x3F));
     out+=(char)(0x80|((cp>>6)&0x3F));
     out+=(char)(0x80|(cp&0x3F));
    }
   }
  }
 }
 return(out);
}
//----------------------------------------------------------------------------------------------------
//инициализация byte_encoder как в оригинале OpenAI
//----------------------------------------------------------------------------------------------------
void CGPT2Tokenizer::InitByteEncoder(void)
{
 //создаем список "печатных" символов Latin-1
 std::vector<int32_t> bs;
 for(int32_t i=33;i<=126;i++) bs.push_back(i);
 for(int32_t i=161;i<=172;i++) bs.push_back(i);
 for(int32_t i=174;i<=255;i++) bs.push_back(i);
 std::set<int32_t> bs_set(bs.begin(),bs.end());
 //кодируем неподдерживаемые байты в дополнительный диапазон Unicode
 int32_t n=0;
 for(int32_t b=0;b<256;b++)
 {
  if (bs_set.count(b)) byte_encoder_[b]=UnicodeToUTF8(b);
  else
  {
   byte_encoder_[b]=UnicodeToUTF8(256+n);
   n++;
  }
  byte_decoder_[byte_encoder_[b]]=b;
 }
}
//----------------------------------------------------------------------------------------------------
//декодирование Unicode escape последовательностей в JSON
//----------------------------------------------------------------------------------------------------
std::string CGPT2Tokenizer::DecodeJSONString(const std::string& str)
{
 std::string result;
 for(size_t i=0;i<str.length();i++)
 {
  if (str[i]=='\\' && (i+1)<str.length())
  {
   i++;
   switch (str[i])
   {
    case 'n': result += '\n';break;
    case 'r': result += '\r';break;
    case 't': result += '\t';break;
    case '\\': result += '\\';break;
    case '"': result += '"';break;
    case 'u':
    {
     if (i+4<str.length())
     {
      std::string hex=str.substr(i+1,4);
      int cp=strtol(hex.c_str(),nullptr,16);
      result+=UnicodeToUTF8(cp);
      i+=4;
     }
     break;
    }
    default: result+=str[i];break;
   }
  }
  else result+=str[i];
 }
 return(result);
}
//----------------------------------------------------------------------------------------------------
//загрузка encoder.json и vocab.bpe
//----------------------------------------------------------------------------------------------------
bool CGPT2Tokenizer::Load(const std::string& encoder_path,const std::string &vocab_bpe_path)
{
 InitByteEncoder();
 //Загрузка encoder.json
 FILE *fp=fopen(encoder_path.c_str(),"rb");
 if (fp==NULL) return(false);
 fseek(fp,0,SEEK_END);
 size_t fsize=ftell(fp);
 fseek(fp,0,SEEK_SET);
 std::vector<char> data(fsize+1);
 fread(data.data(),1,fsize,fp);
 data[fsize]=0;
 fclose(fp);
 std::string content(data.data());
 //простой парсер JSON для этого формата
 size_t pos=content.find('{')+1;
 while (pos<content.length())
 {
  while (pos<content.length() && content[pos]!='"' && content[pos]!='}')
  {
   pos++;
  }
  if (pos>=content.length() || content[pos]=='}') break;
  pos++;
  std::string key;
  while (pos<content.length() && content[pos]!='"')
  {
   if (content[pos]=='\\' && pos+1<content.length())
   {
    pos++;
    if (content[pos]=='u' && pos+4<content.length())
    {
     std::string hex=content.substr(pos+1,4);
     int32_t cp=strtol(hex.c_str(),nullptr,16);
     key+=UnicodeToUTF8(cp);
     pos += 4;
    }
    else
    {
     switch (content[pos])
     {
      case 'n': key+='\n';break;
      case 'r': key+='\r';break;
      case 't': key+='\t';break;
      case '\\': key+='\\';break;
      case '"': key+='"';break;
      default: key+=content[pos];break;
     }
    }
   }
   else key+=content[pos];
   pos++;
  }
  if (pos>=content.length()) break;
  pos++;
  while (pos<content.length() && content[pos]!=':')
  {
   pos++;
  }
  if (pos>=content.length()) break;
  pos++;
  while (pos<content.length() && (content[pos]==' ' || content[pos]=='\n')) pos++;
  std::string num_str;
  while (pos<content.length() && content[pos]>='0' && content[pos]<='9')
  {
   num_str+=content[pos];
   pos++;
  }
  if (!key.empty() && !num_str.empty())
  {
   int32_t value=atoi(num_str.c_str());
   encoder_[key] = value;
   decoder_[value] = key;
  }
  while (pos<content.length() && content[pos]!=',' && content[pos]!='}')
  {
   pos++;
  }
  if (pos<content.length() && content[pos]==',') pos++;
 }
 //printf("Готово. Загружено %zu токенов.\n", encoder_.size());

 //Загрузка vocab.bpe
 fp=fopen(vocab_bpe_path.c_str(),"rb");
 if (fp==NULL) return(false);

 char line[2048];
 int32_t rank=0;
 bool first_line=true;
 while (fgets(line,sizeof(line),fp))
 {
  size_t len=strlen(line);
  while (len>0 && (line[len-1]=='\n' || line[len-1]=='\r')) line[--len]=0;
  if (len==0) continue;
  if (first_line&&line[0]=='#')
  {
   first_line=false;
   continue;
  }
  first_line=false;
  char* space=strchr(line,' ');
  if (space==NULL) continue;
  *space=0;
  std::string first(line);
  std::string second(space+1);
  if (!first.empty() && !second.empty())
  {
   first=DecodeJSONString(first);
   second=DecodeJSONString(second);
   std::pair<std::string,std::string> item(first,second);
   bpe_ranks_[item]=rank++;
  }
 }
 fclose(fp);
 //printf("Готово. Загружено %d BPE-пар.\n", rank);
 return(true);
}
//----------------------------------------------------------------------------------------------------
//преобразование слова в последовательность байтов, закодированных через byte_encoder
//----------------------------------------------------------------------------------------------------
std::vector<std::string> CGPT2Tokenizer::EncodeWordToBytes(const std::string &word)
{
 std::vector<std::string> result;
 for(size_t n=0;n<word.size();n++)
 {
  char c=word[n];
  uint8_t byte=static_cast<uint8_t>(c);
  result.push_back(byte_encoder_[byte]);
 }
 return(result);
}
//----------------------------------------------------------------------------------------------------
//byte-Pair Encoding
//----------------------------------------------------------------------------------------------------
std::vector<std::string> CGPT2Tokenizer::BPE(const std::string &token)
{
 auto cache_it=bpe_cache_.find(token);
 if (cache_it!=bpe_cache_.end()) return(cache_it->second);

 std::vector<std::string> word=EncodeWordToBytes(token);
 if (word.size()<2)
 {
  bpe_cache_[token]=word;
  return(word);
 }
 while (true)
 {
  int32_t best_rank=1000000000;
  int32_t best_idx=-1;
  for (size_t i= 0;i<word.size()-1;i++)
  {
   auto item=std::make_pair(word[i],word[i+1]);
   auto it=bpe_ranks_.find(item);
   if (it!=bpe_ranks_.end() && it->second<best_rank)
   {
    best_rank=it->second;
    best_idx=i;
   }
  }
  if (best_idx==-1) break;
  word[best_idx]=word[best_idx]+word[best_idx+1];
  word.erase(word.begin()+best_idx+1);
 }
 bpe_cache_[token]=word;
 return(word);
}
//----------------------------------------------------------------------------------------------------
//разбиение текста на токены по паттерну GPT-2
//----------------------------------------------------------------------------------------------------
std::vector<std::string> CGPT2Tokenizer::SplitToTokens(const std::string &text)
{
 std::vector<std::string> tokens;
 size_t i=0;
 while (i<text.length())
 {
  //проверяем специальные последовательности: 's, 't, 're, 've, 'm, 'll, 'd
  bool found_special=false;
  for(size_t n=0;n<Specials.size();n++)
  {
   const char *s=Specials[n].c_str();
   size_t len=Specials[n].length();
   if (i+len<=text.length() && text.substr(i,len)==s)
   {
    tokens.push_back(s);
    i+=len;
    found_special=true;
    break;
   }
  }
  if (found_special==true) continue;
  unsigned char c=text[i];
  //пробелы: группируем все подряд идущие пробелы вместе со следующим за ними словом
  if (c==' ')
  {
   size_t space_start=i;
   while (i<text.length() && text[i]==' ') i++;
   std::string spaces=text.substr(space_start,i-space_start);
   if (i<text.length())
   {
    //определяем тип следующего токена и захватываем его вместе с пробелами
    unsigned char next_c=text[i];
    size_t token_start=i;
    if ((next_c>='a' && next_c<='z') || (next_c>='A' && next_c<='Z') || next_c>=0x80)
    {
     //буквы (включая Unicode)
     while (i<text.length())
     {
      c=text[i];
      if (c==' ' || c=='\n' || c=='\r') break;
      if ((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' &&c<='9') || c>=0x80 || c=='\'' || c=='-') i++;
                                                                                                    else break;
     }
    }
    else
    {
     if (next_c >= '0' && next_c <= '9')
     {
      //цифры
      while (i < text.length() && text[i] >= '0' && text[i] <= '9') i++;
     }
     else
     {
      if (next_c == '\n' || next_c == '\r') i++;//перевод строки
                                       else i++;//пунктуация и другие одиночные символы
     }
    }
    std::string word = text.substr(token_start, i - token_start);
    tokens.push_back(spaces + word);
   }
   else tokens.push_back(spaces);//строка заканчивается пробелами — добавляем их как отдельный токен
   continue;
  }
  //буквы без пробела в начале
  if ((c>='a' && c<='z') || (c>='A' && c<='Z') || c>=0x80)
  {
   size_t start=i;
   while (i<text.length())
   {
    c=text[i];
    if (c==' ' || c=='\n' || c=='\r') break;
    if ((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c>=0x80 || c=='\'' || c=='-') i++;
                                                                                                   else break;
   }
   tokens.push_back(text.substr(start,i-start));
   continue;
  }
  //цифры без пробела
  if (c>='0' && c<='9')
  {
   size_t start=i;
   while (i<text.length() && text[i]>='0' && text[i]<='9') i++;
   tokens.push_back(text.substr(start,i-start));
   continue;
  }
  //переводы строк
  if (c=='\n' || c=='\r')
  {
   tokens.push_back(text.substr(i,1));
   i++;
   continue;
  }
  //остальные символы (пунктуация и т.д.)
  tokens.push_back(text.substr(i,1));
  i++;
 }
 return(tokens);
}
//----------------------------------------------------------------------------------------------------
//основная функция токенизации
//----------------------------------------------------------------------------------------------------
std::vector<int32_t> CGPT2Tokenizer::Encode(const std::string &text)
{
 std::vector<int32_t> result;
 std::vector<std::string> tokens=SplitToTokens(text);
 for(size_t n=0;n<tokens.size();n++)
 {
  const auto &token=tokens[n];
  if (!token.empty() && token[0]==' ')
  {
   //токен начинается с пробела: берём один пробел + остаток слова
   size_t non_space=0;
   while (non_space<token.length() && token[non_space]==' ') non_space++;
   if (non_space<token.length())
   {
    std::string processed=" "+token.substr(non_space);//один пробел (0x20) + слово
    std::vector<std::string> bpe_tokens=BPE(processed);
    for(size_t m=0;m<bpe_tokens.size();m++)
    {
     const auto &bt=bpe_tokens[m];
     auto it=encoder_.find(bt);
     if (it!=encoder_.end()) result.push_back(it->second);
    }
   }
   //иначе только пробелы — игнорируем
  }
  else
  {
   //токен без пробела
   std::vector<std::string> bpe_tokens=BPE(token);
   for(size_t m=0;m<bpe_tokens.size();m++)
   {
    const auto &bt=bpe_tokens[m];
    auto it=encoder_.find(bt);
    if (it!=encoder_.end()) result.push_back(it->second);
   }
  }
 }
return(result);
}
//----------------------------------------------------------------------------------------------------
//декодирование последовательности токенов обратно в текст
//----------------------------------------------------------------------------------------------------
std::string CGPT2Tokenizer::Decode(const std::vector<int32_t> &tokens)
{
 std::string text;
 for(size_t n=0;n<tokens.size();n++)
 {
  int32_t token=tokens[n];
  auto it=decoder_.find(token);
  if (it!=decoder_.end()) text+=it->second;
 }
 std::string result;
 for (size_t i=0;i<text.length();)
 {
  bool found=false;
  //ищем самую длинную известную последовательность byte_encoder
  for (size_t len=std::min<size_t>(text.length()-i,size_t(4));len > 0;len--)
  {
   std::string sub=text.substr(i,len);
   auto it=byte_decoder_.find(sub);
   if (it!=byte_decoder_.end())
   {
    result+=(char)(it->second);
    i+=len;
    found=true;
    break;
   }
  }
  if (found==false)
  {
   result+=text[i];
   i++;
  }
 }
 return(result);
}
//----------------------------------------------------------------------------------------------------
//получить декодер
//----------------------------------------------------------------------------------------------------
const std::map<int32_t,std::string>& CGPT2Tokenizer::GetDecoder(void) const
{
 return(decoder_);
}


//----------------------------------------------------------------------------------------------------
//декодировать токен
//----------------------------------------------------------------------------------------------------
std::string CGPT2Tokenizer::DecodeToken(int32_t token_id)
{
 auto it=decoder_.find(token_id);
 if (it!=decoder_.end()) return(it->second);
 return("");
}

//----------------------------------------------------------------------------------------------------
//тест текста
//----------------------------------------------------------------------------------------------------
void CGPT2Tokenizer::TestText(const std::string &text)
{
 printf("\n=== Тест токенизации ===\n");
 printf("Вход: '%s'\n", text.c_str());
 std::vector<int32_t> tokens=Encode(text);
 printf("Токены (%zu):",tokens.size());
 for(size_t n=0;n<tokens.size();n++)
 {
  int32_t id=tokens[n];
  printf("%d ",id);
 }
 printf("\n");
 printf("Декодировано: ");
 for(size_t n=0;n<tokens.size();n++)
 {
  int32_t id=tokens[n];
  printf("[%s]",DecodeToken(id).c_str());
 }
 printf("\n");
 std::string decoded=Decode(tokens);
 printf("Восстановлено: '%s'\n",decoded.c_str());
 printf("=== Конец теста ===\n\n");
}
//----------------------------------------------------------------------------------------------------
//тест
//----------------------------------------------------------------------------------------------------
void CGPT2Tokenizer::Test(void)
{
 printf("\n=== ПРОВЕРКА ТОКЕНИЗАТОРА ===\n");
 TestText("Hello!");
 TestText("Hello world");
 TestText("hello world");
 TestText("the quick brown fox");
 printf("=== ПРОВЕРКА ЗАВЕРШЕНА ===\n\n");
}

