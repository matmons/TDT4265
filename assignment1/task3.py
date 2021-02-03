import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    predictions = model.forward(X)
    accuracy = 0
    for n in range(X.shape[0]):
        prediction = np.argmax(predictions[n, :])
        target = np.argmax(targets[n, :])
        if prediction == target:
            accuracy += 1
    
    return accuracy /X.shape[0]


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        logits = self.model.forward(X_batch)
        self.model.backward(X_batch, logits, Y_batch)
        self.model.w = self.model.w - self.learning_rate * self.model.grad
        
        return cross_entropy_loss(Y_batch, logits)

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val

def visualize_weights(weights: np.ndarray):
    picture = np.zeros((28, 280))
    for n in range(weights.shape[1]):
        for i in range(picture.shape[0]):
            picture[i, (n*28):(28*(n+1))] = weights[(i*28):(28*(i+1)), n]
    return picture

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    weight = visualize_weights(model.w)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
   

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    l2_norm1 = (np.sum(model1.w*model1.w))
    weight1 = visualize_weights(model1.w)

    
    model2 = SoftmaxModel(l2_reg_lambda=.1)
    trainer = SoftmaxTrainer(
        model2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg02, val_history_reg02 = trainer.train(num_epochs)
    l2_norm2 = (np.sum(model2.w*model2.w))
    
    model3 = SoftmaxModel(l2_reg_lambda=.01)
    trainer = SoftmaxTrainer(
        model3, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg03, val_history_reg03 = trainer.train(num_epochs)
    l2_norm3 = (np.sum(model3.w*model3.w))
    
    model4 = SoftmaxModel(l2_reg_lambda=.001)
    trainer = SoftmaxTrainer(
        model4, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg04, val_history_reg04 = trainer.train(num_epochs)
    l2_norm4 = (np.sum(model4.w*model4.w))
    
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    weights = np.concatenate((weight, weight1))
    print(weights.shape)
    #plt.plot(model1.w)
    plt.imsave("task4b_softmax_weight.png", weights, cmap="gray")
    plt.show()

    # Plotting of accuracy for difference values of lambdas (task 4c)
    
    plt.ylim([0.7, 1.1])
    utils.plot_loss(val_history_reg01["accuracy"], "Validation Accuracy lambda = 1")
    utils.plot_loss(val_history_reg02["accuracy"], "Validation Accuracy lambda = 0.1")
    utils.plot_loss(val_history_reg03["accuracy"], "Validation Accuracy lambda = 0.01")
    utils.plot_loss(val_history_reg04["accuracy"], "Validation Accuracy lambda = 0.001")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    
    lambdas = [1, .1, .01, .001]
    norms = [l2_norm1, l2_norm2, l2_norm3, l2_norm4]
    print(norms)
    #plt.plot((np.dot(model1.w.T,model1.w)),label="lambda = 1")
    #plt.plot((np.dot(model2.w.T,model2.w)),label="lambda = 0.1")
    #plt.plot((np.dot(model3.w.T,model3.w)),label="lambda = 0.01")
    #plt.plot((np.dot(model4.w.T,model4.w)),label="lambda = 0.001")

    plt.xlim([-0.1, 1.1])
    plt.plot(lambdas, norms, marker='o', markerfacecolor='blue')
    plt.xlabel("Lambdas")
    plt.ylabel("L2 norm of weight")
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()