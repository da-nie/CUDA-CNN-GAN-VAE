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
 static const NEURON_FUNCTION NEURON_FUNCTION_GELU=5;
};


#endif
