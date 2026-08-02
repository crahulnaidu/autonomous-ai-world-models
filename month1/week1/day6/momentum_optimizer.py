import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

new_val=np.random.uniform(-10,10)

x=new_val

lr=0.1

epochs=25

losses=[]

for epoch in range(epochs):
  losses.append(np.mean((x)**2))
  grad=2*x

  x-=lr*grad



lr=0.001

beta=0.9

m=np.random.rand()


losses_momentum=[]

x=new_val

epochs=25

diff=1e3

for epoch in range(epochs):
  losses_momentum.append(np.mean((x)**2))
  grad=2*x

  m=beta*m+grad
  x-=lr*m


plt.plot(losses,range(epochs),marker='o',color='red',label="convergence for GD")
plt.plot(losses_momentum,range(epochs),marker='o',color='blue',label="convergence for momentum")
plt.title("convergence of GD vs momentum")    