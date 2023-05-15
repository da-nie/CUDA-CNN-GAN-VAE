#ifndef NEURON_H
#define NEURON_H

//****************************************************************************************************
//функции нейрона
//****************************************************************************************************

#include <stdio.h>
#include <stdint.h>

//****************************************************************************************************
//функции нейрона
//****************************************************************************************************

namespace NNeuron
{
 typedef double (*neuron_function_ptr_t)(double);

 typedef uint32_t NEURON_FUNCTION;

 static const NEURON_FUNCTION NEURON_FUNCTION_SIGMOID=0;
 static const NEURON_FUNCTION NEURON_FUNCTION_RELU=1;
 static const NEURON_FUNCTION NEURON_FUNCTION_LEAKY_RELU=2;
 static const NEURON_FUNCTION NEURON_FUNCTION_LINEAR=3;
 static const NEURON_FUNCTION NEURON_FUNCTION_TANGENCE=4;

 double Sigmoid(double v);//сигмоид
 double ReLU(double v);//ReLU
 double LeakyReLU(double v);//LeakyReLU
 double Linear(double v);//линейная
 double Tangence(double v);//гиперболический тангенс

 double dSigmoid(double v);//производная сигмоида
 double dReLU(double v);//производная ReLU
 double dLeakyReLU(double v);//производная LeakyReLU
 double dLinear(double v);//производная линейной функции
 double dTangence(double v);//производная гиперболического тангенса

 neuron_function_ptr_t GetNeuronFunctionPtr(NEURON_FUNCTION neuron_function);//получить указатель на функцию нейрона
 neuron_function_ptr_t GetNeuronFunctionDifferencialPtr(NEURON_FUNCTION neuron_function);//получить указатель на производную функции нейрона

};


#endif
