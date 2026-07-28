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

plt.plot(X,Y,'b.')
plt.plot(X,X_b.dot(theta),'r-')
plt.title("Predicted vs real data")


