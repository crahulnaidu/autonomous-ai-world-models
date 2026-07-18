import matplotlib.pyplot as plt
import numpy as np

X=np.random.rand(100,1)
Y=4*X+5


iterations=300
lr=0.1

#Initializing the parameters.
theta=np.random.randn(2,1)

X_b=np.c_[X,np.ones((100,1))]

loss_history=[]

for epoch in range(iterations):
  #hypothesis=X*theta
  y=np.matmul(X_b,theta)

  loss=np.mean((y-Y)**2)
  
  loss_history.append(loss)

  theta-=lr*1/(X.shape[0])*(np.matmul(X_b.T,y-Y))


plt.plot(X,Y,label="Real line",color='red')
plt.plot(X,np.matmul(X_b,theta),label="predicted line",color='blue',linestyle='dashed')
plt.legend()
