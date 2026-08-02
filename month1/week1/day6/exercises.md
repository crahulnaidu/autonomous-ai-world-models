## Theory

1.When features are differently scaled gradient descent oscillate's because one feature is more elongated than the other in the loss contour,which leads to really slow convergence speed.

2.Momentum is the accumulation or exponential moving average of gradients.

3.EMA is the average of values from past to present,such that more preference is given to present values than past.

4.Beta close to 1 helps in speeding up convergency for a smooth loss surface but also smoothing out oscillations in case of uneven terrain.

5.Yes momentum can hurt optimization depending upon the initial value's or the loss landscape.

## Mathematics.

1.vt=beta*vt-1+(1-beta)*gt

vt=beta^t*v0+beta^(t-1)*(1-beta)*g1+....(1-beta)*gt

2.Once expanded the EMA has higher power's for past observations , and since beta is <1, it leads to past values getting less importance than recent ones.

## Coding.

Already done.

## Quantum.

1.X gate inverts a qubit i.e ket 0 to ket 1 and ket 1 to ket 0.Z gate on the other hand adds a phase of -1 to the ket 1 state.

2.H creates a superpostion of 2 qubits and hence cannot be obtained by rotation about any single computational axis.

3.Since quantum state vectors are eculidian any unitary matrix that acts on a quantum state vector must lead to another state vecotor.Since it has to be euclidian, this puts a constraint on the matrix that it has to be unitary.