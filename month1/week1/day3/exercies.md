## Theory

1.A loss function is a measure of  how much a model's output deviate's from the ground truth for a single instance.

2.Cost function is the average of the loss for the entire dataset.

3.Squaring the error help's make the cost function easier to differentiate and find the global minimum.

4.MAE is more resistant to outliers compared ot MSE.

5.yes when it overfits the data.

## Math.

1.MSE=1/(Total no of samples)*(predicted-ground_truth)**2

2.MAE=1/(Total no of samples0*abs(predicted-ground_truth)

3.MSE is larger after adding and outlier.

## Coding.

1.

def MSE(predict,ground_truth):
  """
  predict:model's output on new input.

  ground_truth:The actual value associated with the input.

  MSE is the mean of the sum of residuals squared,
  where residuals=predict-ground_truth
  """
  return np.mean((predict-ground_truth)**2)

2.

def MAE(predict,ground_truth):
   """
  predict:model's output on new input.

  ground_truth:The actual value associated with the input.

  MAE is the mean of the sum of absolute residuals,
  where residuals=predict-ground_truth
  """
  return np.mean(abs(predict-ground_truth))


3.MAE is more immune to outlier's comapred to MSE.

## Quantum

1.A bell state is a 2 qubit non-product state that forms one of the basis for all 2 qubit quantum states.

2.Measurement outcomes are correleated because the bell state is entangled i.e the measurement of 1 qubit depends on the other.

3.No correleation does not imply classical communication.
