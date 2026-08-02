1. The momentum update equation is given by 

m<-beta*m+gradient.

theta<-theta-lr*m

Here m-momentum operator,beta-hyperparameter for momentum operator(friction parameter),
gradient-gradient of the cost function w.r.t its parameters,lr-learning rate.

this equation works because for a smooth gradient like in linear regression , the previous gradients add up ,leading to a speedy convergence to the global minimum.

It is the exponential moving average of gradient's.For irregular loss surface's , it average's out the oscillations leading to stable convergence.For ex-for a regression problem with 2 features with vastly different scales,the cost function would be elongated in the direction of the larger feature ,leading to a zig-zag and really slow convergence for gradient descent.On the other hand for momentum optimizer, the oscillations in the elongated direction get averaged out leading to a stable convergence towards the minimum, and on the horizontal side the gradients get added up leading to a faster convergence.


2. For beta=0, it is a case of pure gradient descent.For beta=1,it may lead to oscillations as the gradient's in the same direction keep adding up without converging (1+beta+beta^2+beta^3..) leading to uncontrolled oscillations.


3. def momentum_optimizer(theta,lr,beta,epochs):
  np.random.seed(42) # For reproducibility.

  x=np.random.rand(100,1) # Input feature
  y=3*x+np.random.randn(100,1) # Output 

  m=np.random.rand() # Initializing the momentum operator.

  theta=np.random.rand(2,1) # Initializing the model parameters.

  x_b=np.c_[x,np.ones((100,1))] # Additing the bias term to the input feature.

  loss_history=[]

  for epoch in range(epochs):
    loss_history.append(np.mean((x_b.dot(theta)-y)**2))
    grad=(2/100)*x_b.T.dot(x_b.dot(theta)-y)

    m=m*beta+grad

    theta-=lr*m


  return theta,loss_history 


4. A quantum gate must transform every quantum state vector to a quantum state vector ,since a quantum state vector must have a euclidian norm of 1, the quantum gate must be unitary to allow that i.e UU*=U*U,

where U*=conjugate transpose of the matrix U.

let q be a quantum state vector with euclidian norm =1 i.e |q|=1.
If a quantum gate U acts on q ,the resultant state would be Uq.
For the resultant state to be a valid quantum state vector
|Uq|=1 => |U|=1 => U is unitary.


5. The robot should gradually forget older information and emphasize recent information as in many situations only recent information carries the most weight useful for predicting the future.For ex- for a house cleaning robot, the state of the room at the current instance is the most important and required for it to clean the room rather than its past state.

It's similar to EVA in momentum as in it the recent observations get more weight compared to older observations.

For ex-A humanoid robot is trying to climb a mountain for a rescue operation.The robot judges on how to climb the mountain based on the current terrain in its vicinity.If it tries to remember the terrain from the starting point till now , then it wont be able to judge properly on how much force should be applied in its joint's and where should it be more careful.
