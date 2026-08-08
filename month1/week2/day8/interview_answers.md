1.Perceptron was a revolutionary idea because it allowed for the representation of different boolean and mathematical functions making it a general information processing system imitating that of biological neurons.Its limitation was that it could only represent functions with linearly separable data.It could not model for ex-the XOR gate because of its non-linearity.

2.The ouput of a single layer is a Affine transformation of the input,which when fed into another layer would also result in a Affine transformation but with different coefficients.For ex-suppose the first layer of a neural net models 2*x+5,and the second layer models 3*y+6,where y is the input of the first layer ,the output of the second layer would be 3(2*x+5)+6=6*x+21,which is also a liner/Affine transformation of the input.


3.class Perceptron:
  def __init__(self,input_dim:int,lr:float=0.1):
    self.weights=np.random.rand(input_dim+1,1) # Adding 1 for the bias.
    self.lr=lr
    self.history=[] # stores weight snapshots after every epoch.


# Outputs a prediction for an input X
  def predict(self,X:np.ndarray)->np.ndarray:
    # X:(N,D)
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    linear_output=X_b@self.weights

    return (linear_output>=0).astype(int)

# Trains the perceptron for input X,output Y for some epochs.
  def fit(self,X:np.ndarray,Y:np.ndarray,epochs:int=20):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    self.history=[self.weights.copy()]

    for epoch in range(epochs):
      prediction=X_b@self.weights
      error=prediction-Y
      gradient=X_b.T@(error)*1/len(Y)

      self.weights-=self.lr*gradient    
      self.history.append(self.weights.copy())

    return self  



4.For XOR the data points when plotted in graph are not lineary separable i.e there is no line that can perfectly separate the 2 classes i.e 0 and 1.
Algebraiclly if we try to solve the equations for the inputs and outputs ,we get F(w1*1+w2*1)=0,F(w1*0+w2*0)=0,F(w1*1+w2*0)=1,F(w1*0+w2*1)=1,

where F is the activation function and w1,w2 are the activation weights for the inputs to the perceptron.The above equations are not satisfied for any value's of w1,w2 and the step function F ,Hence a single perceptron cannot model XOR.


5.I would choose combining a learned latent representation with a graph of interacting objects.For ex-a TV and a remote will have a similar latent representation and will have a graph between them as they interact with each other, so a household robot will know they must be in same place or nearby.This would generalize well to other objects as well because objects have a unique latent representation regardless of how it's features were stored or recorded and their relationship with other objects also dont change.For ex-for a vase with flowers, the robot might learn that the vase has flower's in it and will recognize it regardless of its shape or how its pixels are recorded.This representation is also compuationally efficient as it only stores the necessary features and relationships between them by filtering the unnecessary ones.It would also be able to reason physically as it knows the relationship between different objects. for ex- suppose the flowers and the vase are in different positions because of a prior situation,the robot knows that the flower must go into the vase because of the relationship graph.This representation would also help the robot in long term planning as it knows how the action of one object would affect the other.All these considerations would mean the AI Accelerator must store compressed latent information and also selectively retain and forget past information depending upon the current situation.
