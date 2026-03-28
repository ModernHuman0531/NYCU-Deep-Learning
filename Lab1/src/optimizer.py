import numpy as np
class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """
        Update the parameters using their gradients and learning rate.
        """
        for param in self.parameters:
            param.data -= self.lr * param.grad
    
    def zero_grad(self):
        """
        Reset the gradients of all parameters to zero.
        Used at the beginning of each training iteration to prevent accumulation of gradients from previous iterations. 
        """
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)
    
    def step(self):
        """
        Update the parameters using their gradients and learning rate.
        The formula for updating the parameters is: param.data -= lr*param.grad
        """
        for param in self.parameters:
            param.data -= self.lr * param.grad
    
    def zero_grad(self):
        """
        Reset the gradient of all parameters to zero.
        Used at the beginning of each training iteration to prevent accumulation of gradients from previous iterations. 
        """
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)
        