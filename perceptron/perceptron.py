import numpy as np
class perceptron:
    def __init__(self,eta,epsilon):
        self.eta=eta
        self.b=None
        self.w=None
        self.epsilon=epsilon
    def sign(self,z):
        #wrong ! i needed to do elementwise comparison
        # if z>=0:
        #     return 1
        # return -1
        return np.where(z>=0,1,-1) #did this avoid explicit for looping.

    def train(self,X,y):
        m,n=X.shape
        self.w=np.random.randint(0,1,size=(m,n),dtype=float)
        self.b=np.zeros((n,),dtype=float)
        
        while(y-y_hat>self.epsilon):
            z=np.matmul(self.w.T,X)+self.b
            y_hat=self.sign(z)
            if(y*y_hat<=0):
                self.w+=self.eta*y*X
                self.b+=self.eta*y

    def predict(self,X_test):
        out=np.matmul(self.w.T,X_test)+self.b
        return self.sign(out)
        


            


