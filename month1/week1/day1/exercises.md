1.AI is the science of teaching machine's to think,reason,plan,perform action's like human's without explicit programming.

2.Machine learning involve's training an algorithm to recognize pattern's and insight's from data.It is a subset of AI as it doen't involve interaction with a dynamic environment and respond accordingly.

3.A model is a function that tries to approximate some input by tuning it's parameters and then using them to predict output for unseen data.

4.Optimization is needed to tune the parameter's of a model so that it can learn the pattern in the data and reduce the loss function.

5.Training involve's tuning a models's parameters to best fit the input data.Inference involve's using the tuned parameter's to output a value based on some unseen data.

## Math.

[17,39]

gradient(x^2)=2*x

gradient(3*x^2)=6*x

## Coding.

x=8.0

lr_arr=[0.01,0.1,1.0]
color=['red','blue','green']




for ind,lr in enumerate(lr_arr):
  val=[]

  x=8.0

  for epoch in range(25):
    val.append(x)
    x=x-lr*2*x

  plt.plot(val,color=color[ind],label=f"lr={lr}")
  plt.xlabel("x value")
  plt.ylabel('no of iterations')
  plt.title("convergence vs learning rate")
  plt.legend()  



## Quantum.

1.Superposition is the linear combination of quantum states such that the resulting vector has a euclidian norm of one.

2.Hadamard gate rotates the quantum state such that it is exactly between ket 0 and ket 1 so that the probability of each state become's 0.5.

3.Since a qubit can be in a superpostion of state's ,each state has a probability associated with it given by the square of its amplitude.So measuring a qubit result's in a state given by its probability,hence measurement is probabilistic.


