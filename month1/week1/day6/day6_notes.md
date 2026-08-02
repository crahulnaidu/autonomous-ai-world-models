# Day 6

## Why Gradient Descent Is Slow.

Gradient descent calculates the gradient of a function at all points at each epoch.If the loss function has plateaus,ridges and valley's then it can get really slow as the updates would only incrementally decrease the loss function value.

## Momentum.

It add's up previous gradient's to speed up convergence in case of a smooth surface and average's out oscillations in case of platueaus and ridges.

## Velocity.

It dictate's by how much should the parameter value change.It is a part of momentum.

## Hyperparameters.

They are the parameters which remain the same or are not part of the model training.ex-no of epochs,learning rate.etc

## Quantum Gates.

The X gate reverses the qubit in the vertical direction in the bloch sphere.The Y gate does the same but adds a phase which is only relevant when a gate operation is performed.The Z gate adds a phase of -1 to the ket 1 qubit.

## Tensor Basics.

A tensor is a high dimensional representation of data optimized for neural network operations.

## Research Reflection.

1.Sometimes forgetting may help smooth out the parameter update or prevent it from diverging in optimization.

2.Remebering only useful information from the past like momentum helps an autonomous robot make smooth decisions as only a handful of features are useful in making future predictions.

3.Yes a world model can benefit from remembering compressed summaries ,because memory is finite and constraiend,remembering every observation would eventually lead to memory overflow.And also in many situations only recent observations have the most weight in predicting the future leading to waster of memory space in the latter case.

4.These decisions influence the design of memory in future autonomous ai systems as memory is finite as the ai systems need to filter which observations are useful and which are not.



## Questions I Still Have.

1.What is bias correction in momentum and why do we need it?
