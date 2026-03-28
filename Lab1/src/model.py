import numpy as np
#------- Construct a basic class for neural network modules and parameters -------#
class NNModule:
    """
    Construct a basic class for neural network modules
    """
    
    def forward(self, x):
        NotImplementedError
    
    def backward(self, grad):
        """
        Need to pass the grad into the backward function, because calculate gradient not only depends on the input x, but also the gradient from the downstream layers. 
        """
        NotImplementedError
    def parameters(self):
        """
        Return a list of parameters in the module.
        """
        NotImplementedError


class Parameter:
    """
    Construct a basic class for parameters in neural networks.
    Contains two values: 
    1. data: the actual value of the parameter, which is a numpy array.
    2. grad: the gradients of each parameter, which is also a numpy array of the same shape as data. Will be used in optimizer to update the parameters.
    
    """
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)


# ------- Construct activation functions classes -------#

class Sigmoid(NNModule):
    def __init__(self):
        self.sigmoid = None
    
    def forward(self, x):
        """
       Define the forward pass of the sigmoid activation function.  
        """
        self.sigmoid = 1.0 / (1.0 + np.exp(-x))
        return self.sigmoid
    
    def backward(self, grad):
        """
        Define the backward pass of the sigmoid activation function.
        The gradient of the sigmoid function can be calculated as sigmoid(x) * (1 - sigmoid(x)) * grad,
        where grad is the gradient from the downstream layers.
        """
        return grad * self.sigmoid * (1 - self.sigmoid)
    def parameters(self):
        """
        Since the sigmoid function doesn't have any parameters, we return an empty list.
        """
        return []

class ReLU(NNModule):
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        """
        Define the forward pass of the ReLU activation function when input is x. 
        """
        self.input = x
        # Since x can be a numpy array, we use np.maximum instead of max(0, x) to apply the ReLU function element-wise.
        return np.maximum(0, x)
    
    def backward(self, grad):
        """
       Define the backward pass of the ReLU activation function.
       The gradient of the ReLU function can be calculated as grad * (input > 0), where grad is the gradient from the downstream layers and input is the original input to the ReLU function.  
        """
        return grad * (self.input > 0)

    def parameters(self):
        """
        Since the ReLU function doesn't have any parameters, we return an empty list.
        """
        return []

# ------ Construct a class for the linear layer -------#
class Linear(NNModule):
    def __init__(self, in_features, out_features, bias=True):
        """
       Initialize the Linear layer with the given input and output features.
       The weights and biases are initialized randomly using a normal distribution. (np.random.randn)
        """
        self.in_features = in_features
        self.out_features = out_features

        # Convert the weights and biases to Parameter objects.
        self.weight = Parameter(np.random.randn(out_features, in_features))
        self.bias = Parameter(np.random.randn(out_features)) if bias else None
    
    def forward(self, x):
        """
        Define the forward pass of the linear layer when input is x(size is in_features).
        The output is calculated as np.dot(x, self.weight.data) + self.bias.data (if bias is not None).
        """
        self.input = x
        if self.bias is not None:
            return np.dot(self.input, self.weight.data.T) + self.bias.data
        return np.dot(self.input, self.weight.data.T)

    def backward(self, grad):
        """
        Define the backward pass of the linear layer. 
        The gradients of the weights and biases can be calculated as follows:
        1. self.weight.grad = np.dot(grad.T, self.input)
        2. self.bias.grad = np.sum(grad, axis=0)
        3. The gradient to be passed to the upstream layers can be calculated as np.dot(grad, self.weight.data)
        """
        self.weight.grad = np.dot(grad.T, self.input)
        if self.bias is not None:
            # grad is (batch_size, out_features)
            self.bias.grad += np.sum(grad, axis=0) # Sum the gradients across the batch dimension to get the gradient for each bias term.
        return np.dot(grad, self.weight.data)
    
    def parameters(self):
        """
       Return a list of parameters in the linear layer, which include the weights and biases (if bias is not None). 
        """
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

# ------- Construct a class for the MLP -------#
class MLP(NNModule):
    """
    Implement a simple MLP using two hidden layers.(Sigmoid or ReLU)
    linear1 -> activation1 -> linear2 -> activation2 
    """
    def __init__(self, in_features, hidden_features, out_features, activation=[Sigmoid(), ReLU()]):
        """
       Define the layer and activation function in the MLP.  
        """
        self.linear1 = Linear(in_features, hidden_features)
        self.linear2 = Linear(hidden_features, out_features)
        self.activation1, self.activation2 = ReLU(), Sigmoid()
    def forward(self, x):
        """
        Define the forward pass of the MLP when input is x.
        The output is calculated as follows:
        1. x' = self.linear1.forward(x)
        2. z = self.activation1.forward(x')
        3. x'' = self.linear2.forward(z)
        4. y = self.activation2.forward(x'')
        """
        x = self.linear1.forward(x)
        if self.activation1 is not None:
           x = self.activation1.forward(x)
        x = self.linear2.forward(x)
        if self.activation2 is not None:
            x = self.activation2.forward(x)
        return x

    def backward(self, grad):
        """
        Define the backward pass of the MLP
        1. grad = self.activation2.backward(grad)
        2. grad = self.linear2.backward(grad)
        3. grad = self.activation1.backward(grad)
        4. grad = self.linear1.backward(grad)
        """
        if self.activation2 is not None:
            grad = self.activation2.backward(grad)
        grad = self.linear2.backward(grad)
        if self.activation1 is not None:
            grad = self.activation1.backward(grad)
        grad = self.linear1.backward(grad)
        return grad
    
    def parameters(self):
        """
        Use the parameters function of the linear layersto get all the parameters in the MLP.
        Return a list of parameters in the MLP for the optimizer to update.
        """
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params

