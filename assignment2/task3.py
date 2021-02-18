import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

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

    # Improved sigmoid
    use_improved_sigmoid = True
    model_improved_s = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_s = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_s, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_s, val_history_improved_s = trainer_improved_s.train(
        num_epochs)

    # Improved sigmoid & weights
    use_improved_weight_init = True

    model_improved_s_w = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_s_w = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_s_w, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_s_w, val_history_improved_s_w = trainer_improved_s_w.train(
        num_epochs)

    # Momentum, sigmoid and weights
    use_momentum = True
    learning_rate = .02
    model_improved_s_w_m = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_s_w_m = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_s_w_m, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_s_w_m, val_history_improved_s_w_m = trainer_improved_s_w_m.train(
        num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_improved_s["loss"], "Task 3 Model - Improved S", npoints_to_average=10)
    utils.plot_loss(
        train_history_improved_s_w["loss"], "Task 3 Model - Improved W", npoints_to_average=10)
    utils.plot_loss(
        train_history_improved_s_w_m["loss"], "Task 3 Model - Improved M", npoints_to_average=10)

    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.9, 1])
    utils.plot_loss(val_history["accuracy"], "Task 3 Model")
    utils.plot_loss(
        val_history_improved_s["accuracy"], "Task 3 Model - Improved S")
    utils.plot_loss(
        val_history_improved_s_w["accuracy"], "Task 3 Model - Improved S&W")
    utils.plot_loss(
        val_history_improved_s_w_m["accuracy"], "Task 3 Model - Improved S&W&M")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task3_tricks_of_the_trade.png")
    plt.show()