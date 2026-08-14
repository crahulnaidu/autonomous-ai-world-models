## Theory

1.step function has a zero gradient everywhere and is not differentiable at zero making it ill-suited for gradient descent ,as the gradients would not change the weights/activations leaving the network completely random.

2.The derivative of sigmoid is sigmoid(x)*(1-sigmoid(x)),for large values of sigmoid(x),the derivative gets multiplies for the lower layers leading to exploding and saturated gradients.

3.Since tanh is centered at 0 and outputs values between -1 and 1 it produces different range of values for the gradient leading to a much smoother gradient flow.

4.ReLU produce's a constant gradient which doen't change much even for deeper layers,leadint to a more stable gradient.

5.It is possible that a large number of neurons have thier inputs and weights such that they only output zero ,leading to dead/off neurons.It leads to the neural network having reduced capacity.


## Mathematics

1.sigmoid derivative=sigmoid(x)*(1-sigmoid(x))

2.tanh derivative=1-tanh(x)**2

3.derivative of ReLU(3x+2)= 3,for x>-2/3,0 for x<-2/3

4.z=wx+b,a=sigmoid(z)

da_dw=sigmoid(z)*(1-sigmoid(z))*w

5.Since H is a unitary matrix HH*=H*H=I

