from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    loglike = 0
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c = data["is_correct"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        p_ia = sigmoid(1 - x)
        loglike += c*np.log(p_a) + (1 - c)*np.log(p_ia)
    return -loglike


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """

    num_users = np.max(data['user_id']) + 1
    num_questions = np.max(data['question_id']) + 1

    # Initialize gradients
    grad_theta = np.zeros(num_users)
    grad_beta = np.zeros(num_questions)

    # Compute gradients
    for i in range(len(data['user_id'])):
        u = data['user_id'][i]
        q = data['question_id'][i]
        c = data['is_correct'][i]
        x = theta[u] - beta[q]
        p_a = sigmoid(x)

        grad_theta[u] += (c - p_a) * (1 - p_a)

        grad_beta[q] -= (c - p_a) * (1 - p_a)

    new_theta = theta + lr * grad_theta
    new_beta = beta - lr * grad_beta

    return new_theta, new_beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    num_users = len(np.unique(data['user_id']))
    num_questions = len(np.unique(data['question_id']))
    theta = np.random.randn(num_users)
    beta = np.random.randn(num_questions)

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    #####################################################################
    learning_rates = [0.005]
    iterations = [24]

    best_lr = None
    best_iterations = None
    best_val_acc = 0
    probs = []
    thetas = []
    for lr in learning_rates:
        for it in iterations:
            print("Tuning hyperparameters: lr={}, iterations={}".format(lr, it))
            theta, beta, val_acc_lst = irt(train_data, val_data, lr, it)
            val_acc = val_acc_lst[-1]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_iterations = it

    # Plot probability against theta
    plt.plot(probs, marker='o', linestyle='-')
    plt.xlabel('Theta')
    plt.ylabel('Probability')
    plt.title('Probability vs. Theta')
    plt.grid(True)
    plt.show()
    print("Best hyperparameters: lr={}, iterations={}".format(best_lr, best_iterations))

    theta, beta, _ = irt(train_data, val_data, best_lr, best_iterations)

    test_acc = evaluate(test_data, theta, beta)
    print("Test Accuracy:", test_acc)
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
