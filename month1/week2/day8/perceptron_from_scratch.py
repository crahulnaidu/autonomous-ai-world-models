import numpy as np

# Implementing a single perceptron.

X=np.random.rand(20,2)

# Creating a line separable input features.
Y=(X[:,0]+X[:,1]>=0.5).astype(int).reshape(-1,1)

# Appending a bias term for the perceptron.
X_b=np.c_[np.ones((X.shape[0],1)),X]

weight=np.random.rand(3,1)

# Using a binary step function.
def step(x):
  return (x>=0).astype(int)


epochs=100
lr=0.1

for epoch in range(epochs):
  prediction=step(X_b@weight)
  error=prediction-Y
  gradient=(X_b.T@(error))*1/len(Y)

  weight-=0.1*gradient


input=np.random.rand(1,2)
input=np.c_[np.ones((1,1)),input]

print(step(input@weight))