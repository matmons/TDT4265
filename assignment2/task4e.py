import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    mean = X_train.mean()
    sd = X_train.std()
    X_train = pre_process_images(X_train, mean, sd)
    X_val = pre_process_images(X_val, mean, sd)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    
    #two hidden layers 
    neurons_per_layer = [59, 59, 10]
    model_two_hidden_layers = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_two_hidden_layers = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_two_hidden_layers, val_history_two_hidden_layers = trainer_two_hidden_layers.train(num_epochs)

    #ten hidden layers 
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    model_ten_hidden_layers = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_ten_hidden_layers = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_ten_hidden_layers, val_history_ten_hidden_layers = trainer_ten_hidden_layers.train(num_epochs)


    # Plot loss for first model (task 4d)
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    #Add text with loss and accuracy information
    plt.annotate("\n Final Train Cross Entropy Loss: {} \n Final Validation Cross Entropy Loss: {} \n Train accuracy: {} \n Validation accuracy: {}".format(
          cross_entropy_loss(Y_train, model.forward(X_train)),
          cross_entropy_loss(Y_val, model.forward(X_val)), calculate_accuracy(X_train, Y_train, model), calculate_accuracy(X_val, Y_val, model)),  # Your string

            # The point that we'll place the text in relation to 
            xy=(0, -0.01), 
            # Interpret the x as axes coords, and the y as figure coords
            xycoords=('axes fraction', 'figure fraction'),

            # The distance from the point that the text will be at
            xytext=(0, 10),  
            # Interpret `xytext` as an offset in points...
            textcoords='offset points',

            # Any other text parameters we'd like
            size=14, ha='left', va='bottom')
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.91, 1.02])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    #Change name of png from according to task
    plt.savefig("task4e_train_loss.png")