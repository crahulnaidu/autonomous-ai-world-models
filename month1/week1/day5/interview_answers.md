1. The main goal of Linear Regression is to best fit a line to input data.
It does this by initializing a set of parameters and then slowly optimizing them by nudging them in the direction of their opposite gradient. The gradient is the derivative of the loss function with respect to the parameters.It gives the direction in which the loss increase's the most and going in the opposite way gives the direction in which the loss decrease's the most.Repeating this continuously for some iterations eventually leads to parameters that minize the loss functin. 
The mathematical update rule is theta=theta-lr*gradient.

Here lr is the step size ,i.e by how much should the parameters move in the opposite direction of the gradient.It should be chosen carefully as it dictate's whether or not the global minimum can be reached or how fast.

2.Due to the widely different range's in the feature scale's ,the loss function become's elongated in the direction of the feature with the larger range.This leads to gradient descent happening really slowly in the direction of the larger feature because a greater step size is required to cause a noticable change in the loss function for this feature.

3.def train_linear_regression(X,y,lr,epochs):
loss_hitory=[] #Stores the loss at each epoch.
m=X.shape[0] #Total no of samples.

X_b=np.c_[X,np.ones((100,1)) #Adding a bias term to the input.

theta=np.random.rand(2,1) #Initializing weight vector.

for epoch in range(epochs):
gradient=(2/m)*X_b.T.dot(X_b.dot(theta)-Y)

loss=np.mean((X_b.dot(theta)-Y)**2)

loss_history.append(loss)

theta-=lr*gradient


return loss_history,theta


4.A 2 qubit state can have 4 possible state's 00,01,10,11. Any 2 qubit state can be formed by a linear combination of these 4 states such that their euclidian norm is one.Hence the no of basis states required is 4.But every 2 qubit state cannot be represented as 2 independent qubits,for ex-the bell state 1/root(2)(ket 00+ket 11), is an entangled state and hence cannot be written as the product of 2 individual qubits.

5.I would train it to predict only abstract properites such as object positions,velocities and interactions.Since there is a lot of noise or unnessary information in a frame,removing it and only retaining the important properites really reduces the computational cost as the number important features are less. It would generalize well because learning the latent reprsentation would allow it to identify properties for objects even under different conditions.For ex- if the model learns that in the latent representation for a book,it is usually rectangular in shape,has pages in it,then whenever it sees a new book it identifies it and then places it in its correct position.Since the latent representations are independent of the lighting or camera changes it would work well in these conditions as well.For ex- a robot might recognize a table even if it is poorly reprsented in the frame.Understanding abstract properties is important to understand the physics of the world as it influence's how an object would move under a force.For ex a robot might learn that bigger a table is the harder it is to push. 
