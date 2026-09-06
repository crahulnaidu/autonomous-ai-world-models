1.Assume an MLP has the following layer dimensions-input=2,hidden_layer1=2,hidden_layer_2=2,output=2
# Assuming bias.

Assuming the input is X with shape-(3,1)

the first weight matrix (with bias) would be of shape-(2,3)-W1

the second weigh matrix would be-(2,3)-W2

# Assuming no bias at the output.
the last would be -(2,2)-W3

Assuming the activation at each layer to be ReLU.

the tensor dimensions at each layer would be 

layer-1

Z1=W2*X
Z1=ReLU(Z1)

(2,1)

layer-2

Z2=W3*Z1
Z2=ReLU(Z2)

(2,1)

output

O=W3*Z2
O=ReLU(O)

(2,1)


Here Z1,Z2,Z3 are the outputs of the respective layers.

So Forward propogation is matrix multiplication followed by an activation function for an MLP.

2.Z=XW+b

XW shape=(128,256)

adding a bias to this would make the resultant shape (128,257).

This is the forward propogation step for a layer in a MLP without the activation function.

3.

(x-->*<--w)(-->+<--b)-->(z)-->Sigmoid-->a-->(-->-<--y)-->I-->^2-->L

Here z,a,I are the intermediate quantities that need to be cached for backpropogation

4.Using ReLU introduces non-linearity by removing negative output neurons and only firing neurons with a positive output.This introduces a hinge or bend to the linear weighed sum function making it linearly separable for non-linear data.

5.The early layers should extract important individual features like the ball,table.etc,the later layers should extract or represent the relationships among the individual features ,like the cup is on the table or the ball is round in shape etc.Non-linearities are useful to model the relationship between different features.For ex- if the table is moved,so is the cup,and if it is moved fast enough the cup might fall,so non-linearties help model this kind of relationship via the latent representation.The model should only remember the import features and forget the rest ,for ex-the background and other lightening conditions are unnecessary for prediction.The hypothetical pipeline should contain early CNN layers that extract the latent representation of the frame,the later layers moodel the feature relationship using non-linearities and the next layers use this to predict what will happen next.


# Research Lab Challenge.

I would say Network B is more suitable for a physical world model as it captures the latent representation of the world by ignoring useless features like background,pixel to pixel variation due to shadow or lightening conditions.Not only is this computationally efficient than network A as the latent representation as less number of feature,it can generalize well to other unseen world instances as it captures the relationship or dynamics between important features.For ex-lets say for a robot which is doing construction work by laying bricks on a adjustable ladder.The robot should automatically increase the height of the ladder when it is unable to reach the top edge of the wall.It can only achieve this if it understands that more the number of brick layers more the height ,and hence more should be the height of the ladder ,this relation between the ladder and brick wall can be understood a lot better by Network B ,as it places them closer together in the latent space ,and predicts that the height of the ladder should increase in response to the brick wall.