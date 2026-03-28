"""
Implementation of loss functions for training the model.
Choose MSE as the loss function.
"""
import numpy as np

class Loss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        pass

    def backward(self, y_pred, y_true):
        pass

class MSELoss(Loss):
    def __init__(self):
        """Inherit from the base Loss class"""
        super().__init__()
    
    def forward(self, y_pred, y_true):
        """
       Calculate the Mean Squared Error loss between the predicted and true labels. 
        """
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred, y_true):
        """
        Calculate the gradient of the MSE loss with respect to the pred_y values.
        This gradient will be used to update the model parameters during backpropagation.
        """
        return 2 * (y_pred - y_true) / len(self.y_true)