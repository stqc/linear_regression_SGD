
class Linear_regression_one_variable_with_SGD:
    '''
    data_x  = Independent variable
    data_y = Dependent variable
    alpha = Learning Rate
    '''
    def __init__(self,data_x,data_y,alpha):
        self.data_x = data_x
        self.data_y = data_y
        self.theta1 = 0
        self.theta2 = 0
        self.alpha = alpha
        
    def predict(self, x):
        return x*self.theta1 + self.theta2
    
        
    def update_model(self,j):
        total_error =0
        for i in range(len(self.data_x)):
            error = self.predict(self.data_x[i]) - self.data_y[i]
            self.theta1 -= (self.alpha*error*self.data_x[i])
            self.theta2 -= (self.alpha*error)
            total_error = error**2
        print(f'Epoch: {j} T1: {self.theta1} T2: {self.theta2} Error: {total_error}')    
        
        
    def train(self,iterations=10):
        for i in range(iterations):
            self.update_model(i)
    
    def __repr__(self):
        return f'{self.theta1}, {self.theta2}'        
