import matplotlib.pyplot as plt
def show_result(x, y, pred_y, output_path=None):
    plt.subplot(1,2,1)
    plt.title('Ground Truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1,2,2)
    plt.title('Prediction result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    if output_path:
        plt.savefig(output_path) 
    plt.show()
    

def show_loss(losses, args, output_path=None):
    """
    Plot the loss curve with the given losses and arguments. If output is provided, save the plot to the specified file. 
    """
    plt.title(f"Training Loss Curve with lr={args.lr}, epochs={args.epochs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses)
    if output_path:
        plt.savefig(output_path)
        
    plt.show()