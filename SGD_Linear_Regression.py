class SGD_Linear_Regression:
    '''
    x = Independent Vairable
    y = Dependent Variable
    alpha = Learning Rate
    
    predict(x)  
    -----------------------------------------------------------
    If sending a single row for prediction enclose it as a list, 
    The predict method will return a list of predicted values
    
    Example:
    Obj.perdict([single_row])
    
    Output: [some_Value(s)]

    '''
    
    def __init__(self,x,y,alpha):
        self.x = x
        self.y = y
        self.cols = len(x[0])
        self.theta1 = 0
        self.theta2 = [0 for i in range(self.cols)]
        self.alpha = alpha
        
    def predict(self,x):
        out =[]
        mx = 0
        for i in range(len(x)):
            for j in range(self.cols):
                mx = mx + x[i][j] * self.theta2[j]
            out.append( mx + self.theta1)
        return out
    
    def update_model(self):
        total_error = 0
        for row in range(len(self.x)):
            error = self.predict([self.x[row]])[0] - self.y[row]
            self.theta1 = self.theta1 - self.alpha*error
            for c in range(self.cols):
                self.theta2[c] = self.theta2[c] - self.alpha*error*self.x[row][c]
            total_error=error**2
        print(f'{row}: T1: {self.theta1} T2: {self.theta2} Error: {total_error}')
        
    
    def train(self,iters =10):
        for i in range(iters):
            self.update_model()
            
    def __repr__(self):
        return f'{self.theta1} , {self.theta2}'
