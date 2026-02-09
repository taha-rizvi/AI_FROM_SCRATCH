import numpy as np
class perceptron:
    def __init__(self,eta,max_iters):
        self.eta=eta
        self.b=None
        self.w=None
        # self.epsilon=epsilon
        self.max_iters=max_iters
    def sign(self,z):
        #wrong ! i needed to do elementwise comparison
        # if z>=0:
        #     return 1
        # return -1
        return np.where(z>=0,1,-1) #did this avoid explicit for looping.
    
    def train(self,X,y):
        m,n=X.shape
        self.w=np.random.randn(n)*0.01
        self.b=0.0
        # z=np.matmul(X,self.w)+self.b
        # y_hat=self.sign(z)
        # for i in range(m):
        #     while(y[i]-y_hat[i]>self.epsilon):
        #         if(y[i]*y_hat[i]<=0):
        #             self.w+=self.eta*y[i]*X[i]
        #             self.b+=self.eta*y[i]
        for _ in range(self.max_iters):
            n_misclassified=0
            for i in range(m):
            
                z=np.dot(X[i],self.w)+self.b
                y_hat=self.sign(z)

                if y[i]*y_hat<=0:
            
                    self.w+=self.eta*y[i]*X[i]
                    self.b+=self.eta*y[i]
                    n_misclassified+=1
            if n_misclassified==0:
                break


    def predict(self,X_test):
        out = np.dot(X_test, self.w) + self.b
        return self.sign(out)
        


            


