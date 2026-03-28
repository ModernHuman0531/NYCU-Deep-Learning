# Lab1-Backpropagation
## File structure
```text
Lab 1/
├── src/
│   ├── data.py      # Contains data generation utilities
│   ├── loss.py      # Implement loss function
│   ├── model.py     # Defines architecture components required for constructing MLP
│   ├── optimizer.py # Implement optimizer
│   ├── train.py     # Implement main training loop
│   └── utils.py     # Implement visualization and plotting functions
├── spec/
├── images/
└── README.md
```

## Experimental results
Conduct experiments on two datasets: linear and XOR problem. For the experiment the MLP was trained using the following parameters:
* Learning rate=1.0
* Number of epochs=100000
* NUmber of neurons per hidden layer: 16
* activation function: Sigmoid, Sigmoid
### Linear Data

### XOR data

## Discussion
### Learning rate
* Number of epochs=100000
* NUmber of neurons per hidden layer: 16
* activation function: Sigmoid, Sigmoid
--- (Linear/XOR)
- 10:100/100
- 1: 100/100
- 0.1: 100/100

### Number of Neurons
* Learning rate=1.0
* Number of epochs=100000
* activation function: Sigmoid, Sigmoid
---
- 4:  100/100
- 8:  100/100
- 16: 100/100

### Different activation function
* Learning rate=0.01
* Number of epochs=100000
* NUmber of neurons per hidden layer: 16
---
- Sigmoid+ReLU(Why don't use pure ReLU):100/100
- Sigmoid: 100/100
- None: 95.00/90.48