1.Gradient is the direction of steepest ascent for a multivariate function.
It is the direction in which the rate of change of the function is maximum with respect to its inputs(>1).Ex-f(x)=2x^2+3y^2,grad(f(x))=4*x*dx+6*y*dy.
i.e for a point let's say (2,3),the direction of the gradient is (4*2,6*3)=(8,18).

Derivative is usually defined for a univariate function and is the ratio of the change in the function output to the change in input.for ex-f(x)=x^2,the derivative is 2*x*dx,i.e for a change in the input dx the output changes by 2x*dx,where x is a point at which the change is measured.

Partial derivative is the derivative of a multivariate function with respect to a single variable assuming the other variable's constant.It gives the direction of maximum change for a particular variable. ex-2x^2+3y,partial derivative wrt x is 4x,i.e changing x by dx changes f(x) by 4x*dx in the x direction.

2.For a large learning rate the step size become's large such that the value oscillates up and down without settling down.It leads to the global minimum being overshot and leading to oscillation.

3.def gradient_descent(x0,learning_rate,iterations):
"""x0-initial value of x.
learning_rate-step size,
iterations-number of epochs.
"""

history=[]
loss=[]
x=x0

history.append(x)
loss.append(x^2)
for epoch in range(iterations):
""" for the function x^2,the gradient is 2x."""
gradient=2*x

x=x-learning_rate*gradient

history.append(x)
loss.append(x^2)


return history,loss


4.Since in the bell state the qubits are entangled,measuring one automatically determines the other.But this does not violate relativity because in order to know the state of the second qubit the measurement of the first qubit has to be transferred via classical communication which is bound by the speed of light.


5.To understand gravity from video's gradient descent alone would not be enough firstly because of the really high dimensionality of the cost function which would make gradient descent very difficult,secondly each frame contains a lot of unnessary information which is not required for gravity like the sunlight,or people or other background objects.The robot needs a good representation of the world to only work on compressed and essential information required to study gravity.For ex-while throwing a ball up ,the size of the ball, the maximum height reached by the ball are meaningful features that are vital for understanding gravity.
Representation learning involve's extracting only relevant and useful features from the video frame to predict the future.Optimizing the number of relevant features is necessary to strike a good balancing betweend accuracy of the prediction and computational co
5.To understand gravity from video's gradient descent alone would not be enough firstly because of the really high dimensionality of the cost function which would make gradient descent very difficult,secondly each frame contains a lot of unnessary information which is not required for gravity like the sunlight,or people or other background objects.The robot needs a good representation of the world to only work on compressed and essential information required to study gravity.For ex-while throwing a ball up ,the size of the ball, the maximum height reached by the ball are meaningful features that are vital for understanding gravity.
Representation learning involve's extracting only relevant and useful features from the video frame to predict the future.Optimizing the number of relevant features is necessary to strike a good balancing betweend accuracy of the prediction and computational cost.
