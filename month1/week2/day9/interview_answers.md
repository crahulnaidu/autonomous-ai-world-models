
1. A neural network consisting of only linear layers produces only a linear transformation of the input.For ex-suppose layer 1 outputs W1x+b1,where x-input,W1-layer 1 weights,b1=layer 1 biases,layer 2 outputs W2a+b2,
where a is the output of the first layer,then output2=w2(w1x+b1)+b2
=w2*w1x+w2*b1+b2=W'x+b'

Hence a stack of linear layers can be replaced by a single neuron with appropriate weight and bias.This means a stack of linear layers have the same expressive power of a single neuron.This is a problem and to overcome it non-linear activations are used which introduces some tiny bumps and Hinge's at each neuron which when combined across neurons can be used to model non-linear functions.


2. z=wTx+b
a=sigmoid(z)

using chain rule da_dw=da_dz*dz_dw

da_dz=sigmoid(z)*(1-sigmoid(z))

dz_dw=xT

therefore da_dw=sigmoid(z)*(1-sigmoid(z))*xT


3. class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.weights=np.random.randn(self.input_dim,self.output_dim)
        self.biases=np.random.randn(self.output_dim,1)

    def forward(self, X):
        return self.weights.T.dot(X)+self.biases


4. Since the average gradient is 0.2,by the time is reaches layer 1 its value would be (0.2)**20,which is really small and negligible.This means the weights of layer 1 would not change much leading to them not learning any useful features about the input. They would just ouput random values. 


5. I would choose option B ReLU/GELU.Their derivative is constant ,which leads to really simple gradient calculations.Also the gradient would not explode or saturate like that of sigmoid or have the vanishing gradient problem as the gradient is constant in each layer.In terms of expressiveness ReLU/GELU can express different types of non-linear functions as they fit the curve using piecewise linear transformations with appropriate number of neurons.Sigmoid also has similar expressive power ,but its gradient calculation is expensive than ReLU/GELU because of the exponent and inverse.In terms of hardware implementation ReLU/GERU is simpler as it only involves comparison and selection ,while for sigmoid ,it is exponent calculation and division,which is a lot costlier.


## Research Lab Challenge.

Firstly an object's position is a non-linear function of its velocity and acceleration,from newtons laws-
2*a*s=v^2-u^2

where a-acceleration,s=displacement,v-final velocity,u-initial velocity.

Since the neuron must model a non-linear function it should an activation function suited for this.

I would choose tanh for this, as  it's derivative is smooth at all points and it can model non-linear functions well.

If the activation saturates,then the model wont be able to learn the curves,and bends of the function properly.It would make the model more linear.

When it comes to hardware implementation ,choosing ReLU over tanh would be a practical choice as it is computationally easier to handle and also has similar expressive power as that of tanh.

The neuron should operate on a learned representation ,as it helps identify and also reduce the number of important features.It leads to a faster and computationally efficient implementation.For ex-from the realtionship between the values it should learn that,greater the force,greater is the acceleration.
        