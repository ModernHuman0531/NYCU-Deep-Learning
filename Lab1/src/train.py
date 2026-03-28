import argparse
import numpy as np
import time

from data import generate_linear, generate_XOR_easy
from model import MLP, Sigmoid, ReLU
from loss import MSELoss
from optimizer import SGD
from utils import show_result, show_loss

def train(model, loss_fn, optimizer, x_train, y_train, args):
    """
    Apply main training loop to train the MLP on the generated dataset. 
    """
    losses = []
    start_time = time.time()
    # Start training loop
    for epoch in range(args.epochs):
        # clean the grdients of all parameters before every epoch
        optimizer.zero_grad()
        # foward pass: compute the model output
        pred_y = model.forward(x_train)
        # compute the loss
        loss = loss_fn.forward(pred_y, y_train)
        # backward pass
        grad_loss = loss_fn.backward(pred_y, y_train)
        model.backward(grad_loss)
        # update the parameters
        optimizer.step()
        # store the loss for visualization
        losses.append(loss)
        # print the loss every 100 epochs
        if (epoch+1)%100 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss:.4f}")
    # print training is finish
    print(f"Training finished")
    # show the loss curve
    if args.output_path:
        show_loss(losses, args, f"{args.output_path}/{args.data}_lr={args.lr}_hidden={args.hidden_size}_activation=ReLU_loss_curve.png")

def test(model, x_test, y_test):
    """
    Test the trained MLP on the test set and visualize the results. 
    """
    pred_y = model.forward(x_test)
    # Pred_y is a continuous value between 0 and 1, we can set a threshold of 0.5 to convert it into binary output (0 or 1).
    binary_output = np.where(pred_y>=0.5, 1 ,0)
    # Calculate the accuracy of the model by comparing the binary output with the true labels.
    accuracy = np.mean(binary_output == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    # Visulaize the results
    if args.output_path:
        show_result(x_test, y_test, binary_output, f"{args.output_path}/{args.data}_lr={args.lr}_hidden={args.hidden_size}_activation=ReLU_result.png")

def main(args):
    # Generate the dataset according to the argument
    if args.data == "linear":
        x, y = generate_linear()
    elif args.data == "XOR":
        x, y = generate_XOR_easy()
    else:
        raise ValueError("Invalid dataset type. Choose either 'linear' or 'XOR'.")
    
    # Initialize the model
    model = MLP(in_features=2, hidden_features=args.hidden_size, out_features=1, activation=Sigmoid())
    # Initialze the loss function and the optimizer
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=args.lr)
    # Train the model
    train(model, loss_fn, optimizer, x, y, args)
    # Test the model
    test(model, x, y)




if __name__ == "__main__":
    """
    Build the argument parser to store or recieve the arguments from the command line or default values.
    - data: the type of dataset to be generated, which can be either "linear" or "XOR" (default: "linear").
    - epochs: the number of epochs to train the model (default: 1000).
    - lr: the learning rate for the optimizer (default: 0.01).
    - hidden_size: the number of neurons in the hidden layer of MLP (defaault: 16).
    - output_path: the path to savet the result's plot (default: None, which means the plot will not be saved).
    """
    # Build argprase object
    parser = argparse.ArgumentParser(description="Train a MLP on generated datasets.")
    # Add arguments to the parser
    parser.add_argument("--data", type=str, default="linear", choices=["linear", "XOR"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--output_path", type=str, default="../images")
    # Parse the arguments
    args = parser.parse_args()
    main(args)

    