from utils import *
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def initialize_parameters(data):
    """Compute the rate of correctness for each question and user.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :return: Tuple (theta, beta)
    """
    # Initialize dictionaries to store counts of correct answers and total attempts for each question and user
    question_correct_counts = {}
    question_total_counts = {}
    user_correct_counts = {}
    user_total_counts = {}

    # Iterate over the data to count correct answers and total attempts
    for i in range(len(data['user_id'])):
        user_id = data['user_id'][i]
        question_id = data['question_id'][i]
        is_correct = data['is_correct'][i]

        # Update question counts
        question_correct_counts[question_id] = question_correct_counts.get(question_id, 0) + is_correct
        question_total_counts[question_id] = question_total_counts.get(question_id, 0) + 1

        # Update user counts
        user_correct_counts[user_id] = user_correct_counts.get(user_id, 0) + is_correct
        user_total_counts[user_id] = user_total_counts.get(user_id, 0) + 1

    # Compute correctness rates for questions and users
    question_rates = {q_id: question_correct_counts[q_id] / question_total_counts[q_id] for q_id in
                      question_correct_counts}
    user_rates = {u_id: user_correct_counts[u_id] / user_total_counts[u_id] for u_id in user_correct_counts}

    theta = np.array([user_rates[u_id] for u_id in user_rates])
    beta = np.array([question_rates[q_id] for q_id in question_rates])

    return theta, beta


def neg_log_likelihood(data, theta, beta, topic, regularization_param):
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
        x = (theta[u] - beta[q])
        p_a = sigmoid(x)
        p_ia = 1 - sigmoid(x)
        loglike += c * np.log(p_a) + (1 - c) * np.log(p_ia) + regularization_param * (theta[u] ** 2) + \
                   regularization_param * (beta[q] ** 2)
    return -loglike


def update_theta_beta(data, lr, theta, beta, topic, regularization_param):
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
    grad_topic = np.zeros(num_questions)

    # Compute gradients
    for i in range(len(data['user_id'])):
        u = data['user_id'][i]
        q = data['question_id'][i]
        c = data['is_correct'][i]
        x = (theta[u] - beta[q])
        p_a = sigmoid(x)

        grad_theta[u] += c - p_a - 2 * regularization_param * theta[u]
        grad_beta[q] -= c - p_a + 2 * regularization_param * beta[q]

    new_theta = theta + lr * grad_theta
    new_beta = beta + lr * grad_beta

    return new_theta, new_beta, topic


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
    df = pd.read_csv('../data/subject_meta.csv')
    word2vec_model = Word2Vec(sentences=[name.split() for name in df["name"]],
                              vector_size=2, window=10, min_count=1, workers=4)
    topic_values = [word2vec_model.wv[name.split()].mean(axis=0) for name in df["name"]]
    topic_values = np.concatenate(topic_values)
    theta, beta = initialize_parameters(data)
    question_meta_df = pd.read_csv('../data/question_meta.csv')
    question_meta_df = question_meta_df.sort_values(by="question_id")
    # Initialize dictionary to store the sum of subject topic values for each question
    question_meta = []

    # Iterate over rows of the DataFrame
    for index, row in question_meta_df.iterrows():
        subject_topics = eval(row['subject_id'])  # Convert string representation of list to actual list
        sum_values = sum(topic_values[i] for i in subject_topics)
        question_meta.append(sum_values)

    val_acc_lst = []
    neg_lld_lst = []
    neg_vlld_lst = []

    for i in range(iterations):
        neg_vlld = neg_log_likelihood(val_data, theta, beta, topic_values, 0)
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, topic=topic_values, regularization_param=0)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        neg_lld_lst.append(neg_lld)
        neg_vlld_lst.append(neg_vlld)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, topic_values = update_theta_beta(data, lr, theta, beta, topic_values, 0)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_lld_lst, neg_vlld_lst


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
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    #####################################################################
    learning_rates = [0.01, 0.001, 0.1]
    iterations = [50]

    best_lr = None
    best_iterations = None
    best_val_acc = 0
    probs = []
    thetas = []
    for lr in learning_rates:
        for it in iterations:
            print("Tuning hyperparameters: lr={}, iterations={}".format(lr, it))
            theta, beta, val_acc_lst, neg_ll_lst, neg_vll_lst = irt(train_data, val_data, lr, it)
            val_acc = val_acc_lst[-1]
            plt.plot(list(range(len(neg_ll_lst))), neg_ll_lst, marker='o')
            plt.title('train log likely hood vs. iterations lr={}'.format(lr))
            plt.xlabel('iterations')
            plt.ylabel('train likely hood')
            plt.grid(True)
            plt.show()
            plt.plot(list(range(len(neg_vll_lst))), neg_vll_lst, marker='o')
            plt.title('valid log likely hood vs. iterations lr={}'.format(lr))
            plt.xlabel('iterations')
            plt.ylabel('valid likely hood')
            plt.grid(True)
            plt.show()
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

    theta, beta, _, _, _ = irt(train_data, val_data, best_lr, best_iterations)

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
