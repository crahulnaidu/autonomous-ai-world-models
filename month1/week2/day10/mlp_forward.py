import numpy as np

class MLP:
  def __init__(self,input_dim:int,hidden_layers:int,hidden_layer_dim:int,output_dim:int):
    self.input_dim=input_dim
    self.hidden_layers=hidden_layers
    self.hidden_layer_dim=hidden_layer_dim
    self.output_dim=output_dim
    self.weight_matrix=[]
    self.output_activations=[]

    total_matrices=1+self.hidden_layers

    for i in range(total_matrices):
      if i==0:
        self.weight_matrix.append(np.random.randn(self.hidden_layer_dim,self.input_dim+1))
      elif i<total_matrices-1:
        self.weight_matrix.append(np.random.randn(self.hidden_layer_dim,self.hidden_layer_dim+1))
      else:
        self.weight_matrix.append(np.random.randn(self.output_dim,self.hidden_layer_dim+1))

  def activation(self,x:np.ndarray)->np.ndarray:
    return np.maximum(0,x)      

  def forward_propogate(self,X:np.ndarray)->np.ndarray:
    X_b=np.c_[X,np.ones((X.shape[0],1))]
    Z=X_b.T

    for i in range(len(self.weight_matrix)):
      Z=self.weight_matrix[i]@Z

      if i<len(self.weight_matrix)-1:
        Z=self.activation(Z)
        self.output_activations.append(Z)
        Z=np.r_[Z,np.ones((1,Z.shape[1]))]

    return Z.T  
               



instance=MLP(2,2,4,2)

X=np.random.rand(50,2)

instance.forward_propogate(X)


