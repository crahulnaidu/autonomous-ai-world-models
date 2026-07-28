## Theory

1.parameters help a model represent a wide range of function's so that they can learn from different types of inputs.

2.Optimizing parameters helps the model to generalize to other inputs never seen during training.On the other hand optimizing inputs doesn't help the model, because the input's can be of wide variety and it would not be possible to optimize all of them.

3.Features with different scale's have a really rugged cost/loss function which really slows down gradient descent.In order to have a smoother cost function scaling is required.

4.In Batch GD all the training instances are used to calculate the gradient for every epoch,in contrast for Stochastic GD only 1 random example is chosen for the gradient.Mini Batch GD is in the middle, it choose's a subset of data from the input for the gradient.

5.No gradient descent can't converge without feature scaling all times, because there may be really wide valley's,plateaus which may really slow down gradient descent and even prevent it from reaching the global minimum.

## Coding

Linear Regression Training.

X=np.random.rand(100,1)
Y=3*X+4+np.random.randn(100,1)

theta=np.random.rand(2,1)

X_b=np.c_[X,np.ones((100,1))]

lr=0.1

iterations=40

loss_history=[]

loss_init=np.mean((X_b.dot(theta)-Y)**2)

loss_history.append(loss_init)

for epoch in range(iterations):
  grad=(2/100)*X_b.T.dot(X_b.dot(theta)-Y)

  theta-=lr*grad

  loss=np.mean((X_b.dot(theta)-Y)**2)

  loss_history.append(loss)


plt.plot(loss_history,range(iterations+1))
plt.title("loss vs iterations")

## Quantum.

1.Since a qubit can be in state 0 or 1,adding one more qubits increases the number of states 2 times as the number of states increase exponentially by each qubit,i.e no of states=2^qubits.

2.A tensor product is combination of each element of one side with the other.

3.Entangled states are non-product states and can't be written as a tensor product of 2 qubits as their measurement's are correlated to each other.


