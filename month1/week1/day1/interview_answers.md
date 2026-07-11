1.AI is the science of teaching macine's to think,plan,reason,action like human's without explicit programming.For ex-a garbage collecting robot that roams around detecting garbage and collecting it.

ML is the subset of AI that involve's training an algorithm to recognize pattern's and insights from data without explicit programming.
ex-House pricing predictor.

DL is subset of ML and involve's teaching a machine to classify,predict,give answers to inputs not seen during training.It involve's building layers of neurons which are mathematical functions interconnected with each other and tuning them to learn about the data. For ex-Teaching a neural net to recognize haandwritten digits.

RL is the subsset of AI that deal's with an agent,environment interaction in which an agent has a set of goals it has to achieve and a set of actions it has to perform.It optimize's its actions to maximize some reward.
For ex-teaching a robot how to walk,the reward could be the amount of time the robot can walk properly.

2.
f(x)=x*2

grad(f(x))=2*x

first iteration x=8.0-0.1*2*8.0=6.4

second iteration x=6.4-0.1*2*6.4=6.12

final value after 2 iterations=6.12


3.
def gradient_descent():
  x=8.0
  lr=0.1
  iterations=50

  for epoch in range(50):
    gradient=2*x

    x=x-lr*gradient


  return x

  pass  

4.
Applying a hadamard gate to ket 0 essentially rotates it halfway between ket 0 and ket 1.This means the probability of the qubit in either state is 50%.The state of a qubit can be in superposition of many state's,which means it is can be in any of the states with the probability given by the square of the amplitude.

5.
The robot should record the inclination of the ramp,the speed of the ball,the shape and size of the ramp and the height at which the ball is released.

It could predict future motion by combining the observation's from previous ramps and using them to predict the speed of the ball at each point.

It doen't necessarliy need newton laws ,as it could predict the motion of the ball from experience based on the different parameters of the new system.For ex-if it saw that the ball has a higher speed upon being released from further up the ramp or for a really steep ramp then it could use that to learn that the speed of the ball is proportional to the inclination of the lamp.Similarlly it learns the relationship between its motion and other parameter's from experience.


