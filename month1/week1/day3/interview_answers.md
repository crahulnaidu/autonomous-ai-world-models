1.
Prediction is the value that a model output's for a input that it has never seen before.

Error is the difference between prediction and ground truth.

Loss is the measure of similarity between the prediction and ground truth for a single instance.

Cost is the average of the Loss for the entire dataset.

for ex-in predicting the average number of death per year based on a country's GDP ,average annual income,happyness index.etc ,the value that a model output's for a specific country is its' prediction. Erros is the no of death by which the model is wrong.Loss is the error for a single instance and cost is the average of the loss for the entire dataset.

2.
MSE has a squared term because of which an outlier would give a much larger value compared ot MAE.

for ex-suppose a model output's value's between 0 to 1 and there is an outlier with value 20 ,then MSE would output (20-some value b/w 0 and 1)**2, which is a large value compared to the other points leading to the model pivoting towards that outlier.

3.

import numpy as np

def mse(y_true,y_pred):
"""y_true:the ground truth or the actual value associated with the input.
y_pred:the value that a model output's for a input.
"""

return np.mean((y_true-y_pred)**2)

4.In an ideal simulation the state's other than 00 and 11 do not exist because of the perfect application of gate's and the absence of noise.

5.Minimizing pixel level MSE would not necessarliy produce an intelligent world model because ,the model would only learn what the pixel value's are now and what they should be 1 s later.If the same action is performed on differnet quality camera or some other background conditions it might predict a different frame even though the actions are the same.Instead it should learn what the frame is actually representing by removing all the unnecessary features and then using them as a reference to predict what happens next.

For ex-suppose a world model is predicting the next frame in a video in which a player is shooting a football.If the model wants to predict the next frame just before the player touches the ball, then it should first identify what the frame is about ,like there is a person who is trying to kick a ball,so the next frame should bring his leg closer to the ball,and then predict the frame based on that understanding.
