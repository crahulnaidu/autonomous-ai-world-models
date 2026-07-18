## Theory

1.A feature is an attirbute of data that help's in distinguishing different sample's.

2.feature engineering is important because it allow's the model to learn only the important feature's and discard the other's.

3.Splitting dataset's is done so that the model can be tested on data it has not seen before and prevent it from over fitting the data.

4.train data is the one from which the model learn's the patterns from and test data is the one which the model makes the predictions.

5.testing data can be used for training but is not recommended because the model would could memorize the noise or pattern in that data and not generalize well to unsee data.

## Mathematics.

1.slope between (2,3) and (5,9)
=(9-3)/(5-2)=2

## Coding

import numpy 
import matplotlib.pyplot as plt

X=np.random.rand(100,1)+0.3*np.random.rand(100,1)
Y=4*X+3+np.random.rand(100,1)

lr=0.1
iterations=700

X_b=np.c_[X,np.ones((100,1))]
theta=np.random.randn(2,1)

for epoch in range(iterations):
y=np.matmul(X_b,theta)

theta-=lr*1/(X.shape[0])*np.matmul(X_b,y-Y)



plt.plot(X,np.matmul(X_b,theta))


## Quantum.

1.During measurement the quantum state of a system collapses and the resulting state is one of the possible classical states of the system.

2.With more shots the probabilities tend to become their original values according to the strong law of large numbers.

3.Measurement destroys a quantum state and the resultant state is a classial state which doen't change upon measurement.


