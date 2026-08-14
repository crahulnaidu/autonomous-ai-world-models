# Day 9

## Activation Functions.
1 Step Function.
2 Sigmoid
3 Tanh 
4 ReLU
5 Leakly ReLU


## Saturation.
When the gradients become really large because of repeated multiplication for tanh and sigmoid ,it leads to constant and saturated gradients leading to unstabilitiy in learning.


## Vanishing Gradients.
When the gradient values are really small for the top layers ,they become smaller and smaller for the lower layers for sigmoid and tanh functions leading to almost null gradient values which leads to almost non-existant updates for the lower layers.


## Quantum Linear Operators.
Unitary operators have a condition that they must unitary i.e their product with their conjugate transpose must be one.


## Hardware Considerations.
ReLU is a common choice for many deep DNN's as it is mathematically easy to compute and has a constant derivative.It also adds non-linearity to the network.Since it is a simple compare and select operation ,the hardware for it would be really simple.


## Research Reflection.


