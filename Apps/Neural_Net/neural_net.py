from module import Module
from linear_layer import Linear

from softmax import softmax
import numpy as np
class NeuralNet(Module):
    def __init__(self,activation,input_size=6,hidden_size=64,output_size=4):
        super().__init__()
        self.fc1=Linear(input_size,64)
        self.activation_1=activation()
        self.fc2=Linear(64,32)
        self.activation_2=activation()
        self.fc3=Linear(32,16)
        self.activation_3=activation()
        self.fc4=Linear(16,output_size)
        self.softmax=softmax()    
        
    def forward(self,x):
        l1=self.fc1(x)  
        l2=self.activation_1(l1)
        l3=self.fc2(l2)
        l4=self.activation_2(l3)
        l5=self.fc3(l4)
        l6=self.activation_3(l5)
        out=self.fc4(l6)
     
        return self.softmax(out)
    
    def backward(self,dout):
        
        dout_loss = self.softmax.backward(dout)
        l4_back=self.fc4.backward(dout_loss)
        dout_l4=self.activation_3.backward(l4_back)
        l3_back=self.fc3.backward(dout_l4)
        dout_l3=self.activation_2.backward(l3_back)
        l2_back=self.fc2.backward(dout_l3)
        dout_l2=self.activation_1.backward(l2_back)
       
        
        return self.fc1.backward(dout_l2)
        
    


        
        
    


