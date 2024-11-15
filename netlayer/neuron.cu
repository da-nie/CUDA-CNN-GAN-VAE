//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include "neuron.cu.h"
#include <math.h>
#include <stdio.h>
//****************************************************************************************************
//глобальные переменные
//****************************************************************************************************

//****************************************************************************************************
//константы
//****************************************************************************************************

//****************************************************************************************************
//макроопределения
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//сигмоид
//----------------------------------------------------------------------------------------------------
double NNeuron::Sigmoid(double v)
{
 if (v>20) v=19.99999;
 if (v<-20) v=-19.99999;
 return(1.0/(1.0+exp(-v)));
}
//----------------------------------------------------------------------------------------------------
//ReLU
//----------------------------------------------------------------------------------------------------
double NNeuron::ReLU(double v)
{
 if (v>0) return(v);
 return(0);
}
//----------------------------------------------------------------------------------------------------
//LeakyReLU
//----------------------------------------------------------------------------------------------------
double NNeuron::LeakyReLU(double v)
{
 if (v>0) return(v);
 return(0.1*v);
}
//----------------------------------------------------------------------------------------------------
//линейная
//----------------------------------------------------------------------------------------------------
double NNeuron::Linear(double v)
{
 return(v);
}
//----------------------------------------------------------------------------------------------------
//гиперболический тангенс
//----------------------------------------------------------------------------------------------------
double NNeuron::Tangence(double v)
{
 if (v>20) v=19.99999;
 if (v<-20) v=-19.99999;
 double ep=exp(2*v);
 double en=exp(-2*v);
 return((ep-en)/(ep+en));
}
//----------------------------------------------------------------------------------------------------
//производная сигмоида
//----------------------------------------------------------------------------------------------------
double NNeuron::dSigmoid(double v)
{
 double s=Sigmoid(v);
 return((1.0-s)*s);
}
//----------------------------------------------------------------------------------------------------
//производная ReLU
//----------------------------------------------------------------------------------------------------
double NNeuron::dReLU(double v)
{
 if (v>=0) return(1);
 return(0);
}
//----------------------------------------------------------------------------------------------------
//производная LeakyReLU
//----------------------------------------------------------------------------------------------------
double NNeuron::dLeakyReLU(double v)
{
 if (v>=0) return(1);
 return(0.1);
}
//----------------------------------------------------------------------------------------------------
//производная линейной функции
//----------------------------------------------------------------------------------------------------
double NNeuron::dLinear(double v)
{
 return(1);
}
//----------------------------------------------------------------------------------------------------
//производная гипербилического тангенса
//----------------------------------------------------------------------------------------------------
double NNeuron::dTangence(double v)
{
 double t=Tangence(v);
 return(1-t*t);
}
//----------------------------------------------------------------------------------------------------
//получить указатель на функцию нейрона
//----------------------------------------------------------------------------------------------------
NNeuron::neuron_function_ptr_t NNeuron::GetNeuronFunctionPtr(NEURON_FUNCTION neuron_function)
{
 if (neuron_function==NEURON_FUNCTION_SIGMOID) return(&Sigmoid);
 if (neuron_function==NEURON_FUNCTION_RELU) return(&ReLU);
 if (neuron_function==NEURON_FUNCTION_LEAKY_RELU) return(&LeakyReLU);
 if (neuron_function==NEURON_FUNCTION_LINEAR) return(&Linear);
 if (neuron_function==NEURON_FUNCTION_TANGENCE) return(&Tangence);
 return(NULL);
}
//----------------------------------------------------------------------------------------------------
//получить указатель на производную функции нейрона
//----------------------------------------------------------------------------------------------------
NNeuron::neuron_function_ptr_t NNeuron::GetNeuronFunctionDifferencialPtr(NEURON_FUNCTION neuron_function)
{
 if (neuron_function==NEURON_FUNCTION_SIGMOID) return(&dSigmoid);
 if (neuron_function==NEURON_FUNCTION_RELU) return(&dReLU);
 if (neuron_function==NEURON_FUNCTION_LEAKY_RELU) return(&dLeakyReLU);
 if (neuron_function==NEURON_FUNCTION_LINEAR) return(&dLinear);
 if (neuron_function==NEURON_FUNCTION_TANGENCE) return(&dTangence);
 return(NULL);
}
