import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [32, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # 32 Neurons in Hidden Layer
    model_32 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_32 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_32, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_32, val_history_32 = trainer_32.train(num_epochs)

    # 64 Neurons in Hidden Layer
    neurons_per_layer = [64,10]
    model_64 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_64 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_64, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_64, val_history_64 = trainer_64.train(num_epochs)

    # 128 Neurons in Hidden Layer
    neurons_per_layer = [128,10]
    model_128 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_128 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_128, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_128, val_history_128 = trainer_128.train(num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(
        train_history_32["loss"], "Task 4 Model - 32 Neurons", npoints_to_average=10)
    utils.plot_loss(
        train_history_64["loss"], "Task 4 Model - 64 Neurons", npoints_to_average=10)
    utils.plot_loss(
        train_history_128["loss"], "Task 4 Model - 128 Neurons", npoints_to_average=10)

    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.9, 1])
    utils.plot_loss(
        val_history_32["accuracy"], "Task 4 Model - 32 Neurons")
    utils.plot_loss(
        val_history_64["accuracy"], "Task 4 Model - 64 Neurons")
    utils.plot_loss(
        val_history_128["accuracy"], "Task 4 Model - 128 Neurons")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4_network_topology.png")
    plt.show()
