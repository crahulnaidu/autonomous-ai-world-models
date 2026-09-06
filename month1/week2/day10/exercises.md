1.Forward propogation is the flow of input through an MLP,where it is multiplied by the weight matrix at each layer and then an activation function is applied.

2.An activation is a mathematical function that helps the MLP to model complex relationships within the data.

3.A parameter is a trainable value of a DNN that changes during training to reduce the loss function.An activation is usually the value obtained after a layer applies the activation function to the weighed sum.

4.A bias helps in adding a shift to the function and helping it model a wider class of functions.

5.The value at each point in the computational graph is the output of a particular subexpression formed by the graph till that point.

6.4 → 8 → 8 → 1

Matrix dimensions.
# Assuming bias.

8x5,
8x9,
1x8

7.X : (32, 100)
W : (100, 64)
b : (64,)

shape of XW-(32,64),
XW+b-(32,65)
ReLU(XW+b)-(32,65)

