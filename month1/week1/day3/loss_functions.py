

def MSE(predict,ground_truth):
  """
  predict:model's output on new input.

  ground_truth:The actual value associated with the input.

  MSE is the mean of the sum of residuals squared,
  where residuals=predict-ground_truth
  """
  return np.mean((predict-ground_truth)**2


def MAE(predict,ground_truth):
   """
  predict:model's output on new input.

  ground_truth:The actual value associated with the input.

  MAE is the mean of the sum of absolute residuals,
  where residuals=predict-ground_truth
  """
  return np.mean(abs(predict-ground_truth))
)
