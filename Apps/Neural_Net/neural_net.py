from module import Module
from linear_layer import Linear
from RELU import RELU
# from sigmoid import sigmoid
from softmax import softmax
import numpy as np
class NeuralNet(Module):
    def __init__(self,input_size=784,hidden_size=256,output_size=10):
        super().__init__()
        self.fc1=Linear(input_size,hidden_size)
        self.RELU=RELU()
        self.fc2=Linear(hidden_size,output_size)
        self.softmax=softmax()
        
    def forward(self,x):
        l1=self.fc1(x)  
        l2=self.RELU(l1)
        out=self.fc2(l2)
        return self.softmax(out)
    
    def backward(self,dout):
        
        
        l2_back=self.fc2.backward(dout)
        relu_back=self.RELU.backward(l2_back)
        return self.fc1.backward(relu_back)
        
    


        
        
    


