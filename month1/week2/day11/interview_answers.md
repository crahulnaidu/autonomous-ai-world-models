1. Backpropogation is the flow of gradients backward through a computational graph.Chain rule is used to decide the value of the gradient of the current node,as the current gradient * local derivative.

2. F=(a-y)
L=F^2

dL=2*F*dF

dF=da

da=(1-sigmoid(z))(sigmoid(z))*dz

dz=x*dw

dL=2*(a-y)*(1-sigmoid(z))*(sigmoid(z))*x*dw

3. Z=W*X+b

dZ=X*dW.T+dX.T*W+db

Since the dimensions of both sides have to match
dW=(256,128)
dX=(128,32)
db=(1,256)

4. During backpropagation the gradients for the inner layers are computed using the activations of those layers ,along with the weights and gradients of the previously calculated layers.This leads to faster calculation of the gradients but increase in memory requirements.

5. Training a neural network involves 2 phases forward and backward pass.In the forward pass the inputs are passed through each layer where they are multiplied by the weights and biases and then passed through an activation function.In backpropagation ,each layer has to compute its gradient using its own input activations and the gradients of its successor layers.For this the intermediate activations and inputs have to be cached leading to high memory requirements for really deep neural nets.This process of forward and backpropagation has to be repeated for many epochs leading to the high computational and memory bandwidth.

