from Apps.Neural_Net.module import Module
from Apps.Neural_Net.linear_layer import Linear


class NeuralNet(Module):
    def __init__(self,activation,input_size=1,hidden_size=5,output_size=1):
        self.fc1=Linear(input_size,hidden_size)
        self.activation=activation()
        self.fc2=Linear(hidden_size,output_size)
    def forward(self,x):
        l1=self.fc1(x)
        l2=self.activation(l1)
        l3=self.fc2(l2)
        return l3
    def backward(self,dout):
        d2=self.fc2.backward(dout)
        d1=self.activation.backward(d2)
        d0=self.fc1.backward(d1)
        return d0
