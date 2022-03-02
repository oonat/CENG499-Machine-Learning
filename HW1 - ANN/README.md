
# Descriptions of Files

- model.py: contains the PyTorch model named ANN.
- train.py: contains functions for model training, validation set accuracy, and test set accuracy calculation.
- run.py: contains functions for train-test set split, validation loss, and accuracy graph plotting. It has the main function, which iterates over different hyperparameter configurations for parameter tuning.

# How to use

## Requirements
I do not use any libraries other than PyTorch to build my model. However, I used the matplotlib library to draw the required graphs. To install the libraries;

    pip install -r requirements.txt 


## Training

To train the model and tune the hyperparameters, use this command;

    python run.py > logs.txt

This command will create a folder named "savings" inside the current directory and save the best models for each hyperparameter configuration. Additionally, training loss and validation loss graphs for each configuration will be saved into this folder in the "png" format. One can view the validation loss, validation accuracy, and best epoch data from the logs.txt file.


Format of the savings : savings/**{layer num}**\_**{neuron num}**\_**{learning rate}**\_**{activation_function}**\_BEST

Format of the plots : savings/**{layer num}**\_**{neuron num}**\_**{learning rate}**\_**{activation_function}**\_PLOT.png


## Testing

I do not have an automated system to obtain the test set accuracy results of the best hyperparameters. To get the test set accuracy results of the best configurations for each k-layer network;

1. Determine the hyperparameter configurations of the models with the least validation loss using the log.txt file.
2. Use the following code in the **run.py** to load the model state and calculate the test set accuracy;

```
model = ANN({layer_num}, {feature_list}, {act_func}).to(device) 
model.load_state_dict(torch.load("savings/{layer_num}\_{neuron_num_first_layer}\_{lr}\_{activation_function}\_BEST")) 
test(model, device, test_loader)  
```

For instance, to calculate the test set accuracy of the model with 2 layers and 1024 neurons in the hidden layer, lr=0.0001 and the activation function is relu, we should use;

    model = ANN(2, [32 * 32, 1024, 10], "relu").to(device)
    model.load_state_dict(torch.load("savings/2_1024_0.0001_relu_BEST"))
    test(model, device, test_loader)