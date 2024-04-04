# 改进ALS-SGD算法的措施可以从以下几个方面入手：
#
# 初始化策略：合适的初始化策略可以对模型的收敛速度和性能产生重要影响。可以尝试使用更智能的初始化方法，如随机初始化、均匀分布初始化或基于先验知识的初始化。
#
# 正则化：引入正则化项可以帮助控制模型的复杂度，防止过拟合，并提高泛化能力。通过在损失函数中加入正则化项，如L1正则化或L2正则化，可以有效地调节模型的复杂度。
#
# 学习率调度：动态调整学习率可以帮助模型在训练过程中更好地收敛到全局最优解。可以尝试使用学习率衰减策略，如指数衰减、余弦退火等，以及基于验证集性能的自适应学习率调整方法。
#
# 批量更新：与每个样本单独更新参数相比，批量更新参数可以提高计算效率并降低训练时间。可以考虑引入批量更新策略，将多个样本组成一个小批量进行参数更新。
#
# 并行化：通过并行化计算，可以加速模型的训练过程，特别是当处理大规模数据集时。可以利用并行计算框架或硬件加速器来实现算法的并行化。
#
# 模型评估：及时而准确地评估模型的性能是优化算法的关键。可以使用更全面的评估指标，如AUC、RMSE等，以更好地衡量模型的质量。
#
# 超参数调优：通过系统地调整超参数，如隐藏因子数量、学习率和迭代次数等，可以进一步提高模型的性能。可以尝试使用自动化的超参数优化技术，如网格搜索、随机搜索或贝叶斯优化。
#
# 通过采取这些改进措施，可以有效地提高ALS-SGD算法的性能和效率，从而更好地应用于实际问题中。
import time

from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import numpy as np
def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z, Lambda, batch_size=1):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    The loss function is given by:
    L = 0.5 * (c - u_n^T * z_q)^2
    where c is the correctness of the answer, u_n is the user vector, and z_q is the question vector.
    Now with L2 regularization, the loss function becomes:
    L = 0.5 * (c - u_n^T * z_q)^2 + 0.5 * lambda * (||u_n||^2 + ||z_q||^2)
    and the gradient of the loss function with respect to u_n and z_q is given by:
    grad_u = (c - u_n^T * z_q) * z_q - lambda * u_n
    grad_z = (c - u_n^T * z_q) * u_n - lambda * z_q
    where lambda is the regularization parameter.
    """
    #####################################################################                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    # SGD
    i = np.random.choice(len(train_data["question_id"]), 1)[0]
    #
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    #
    # Compute the gradient of the loss function with respect to u_n and z_q.
    predicted_rating = np.dot(u[n], z[q])
    error = c - predicted_rating
    #
    # Update u_n and z_q.
    u[n] += lr * error * z[q] - lr * Lambda * u[n]
    z[q] += lr * error * u[n] - lr * Lambda * z[q]

    # Mini-batch SGD
#     num_samples = len(train_data["question_id"])
    #     indices = np.random.choice(num_samples, batch_size, replace=False)  # Randomly sample indices for batch
    #
    #     for i in indices:
    #         c = train_data["is_correct"][i]
    #         n = train_data["user_id"][i]
    #         q = train_data["question_id"][i]
    #
    #         predicted_rating = np.dot(u[n], z[q])
    #         error = c - predicted_rating
    #
    #         # Accumulate gradients for batch
    #         u[n] += lr * error * z[q] - lr * Lambda * u[n]
    #         z[q] += lr * error * u[n] - lr * Lambda * z[q]



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, Lambda, power=1, batch_size=1):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    learning_rate = lr
    #####################################################################                                                            #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(num_iteration):
        u, z = update_u_z(train_data, learning_rate, u, z, Lambda, batch_size)

        # Decrease the learning rate exponentially.
        # learning_rate *= 0.9

        # Cosine annealing learning rate decay
        # lr *= (np.cos(np.pi * i / num_iteration) + 1) / 2

        # Polynomial decay learning rate
        lr *= (1 - i / num_iteration) ** power



    mat = np.dot(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat

def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    initial_k = 6
    initial_lr = 0.06
    initial_num_iteration = 80000
    best_Lambda = 0

    best_acc_als = 0
    Lambda_lst = []
    val_acc_lst = []
    for Lambda in np.arange(0.0, 0.1,0.01):
        reconst_matrix = als(train_data, initial_k, initial_lr, initial_num_iteration, Lambda)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if acc > best_acc_als:
            best_acc_als = acc
            best_Lambda = Lambda
        Lambda_lst.append(Lambda)
        val_acc_lst.append(acc)
        print("Lambda: {}".format(Lambda))

    print("Best Lambda: {} with accuracy: {}".format(best_Lambda, best_acc_als))
    #
    plt.title("Validation Accuracy vs Lambda")
    plt.plot(Lambda_lst, val_acc_lst, label="Validation Accuracy")
    plt.xlabel("Lambda")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("Validation Accuracy vs Lambda.png")
    plt.show()

    #

    best_power = 0
    power_lst = []
    val_acc_lst = []
    for Power in np.arange(1,10,1):
        reconst_matrix = als(train_data, initial_k, initial_lr, initial_num_iteration, best_Lambda, Power)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if acc > best_acc_als:
            best_acc_als = acc
            best_power = Power
        power_lst.append(Power)
        val_acc_lst.append(acc)
        print("Power: {}".format(Power))

    print("Best Power: {} with accuracy: {}".format(best_power, best_acc_als))
    #
    plt.title("Validation Accuracy vs Power of Polynomial Decay")
    plt.plot(power_lst, val_acc_lst, label="Validation Accuracy")
    plt.xlabel("Power")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("Validation Accuracy vs Power of Polynomial Decay.png")
    plt.show()
    #

#     for batch_size in np.arange(1, 100, 10):
#         reconst_matrix = als(train_data, initial_k, initial_lr, initial_num_iteration, best_Lambda, best_power, batch_size)
#         acc = sparse_matrix_evaluate(val_data, reconst_matrix)
#         if acc > best_acc_als:
#             best_acc_als = acc
#             best_batch_size = batch_size
#         print("Batch Size: {}".format(batch_size))
#
#     print("Best Batch Size: {} with accuracy: {}".format(best_batch_size, best_acc_als))


if __name__ == "__main__":
    main()

# log:
# Try using L2 Regularization to improve the ALS-SGD algorithm.Howver the accuracy is not improved.]
# I perceive the reason might be as follows:
# 模型复杂度低：ALS-SGD通常用于矩阵分解等简单模型，这些模型本身就具有较低的复杂度。在模型复杂度低的情况下，过拟合的风险较小，因此正则化对模型的影响也相对较小。
#
# 低维特征空间：ALS-SGD通常使用较低维度的特征空间进行建模，这意味着模型的参数数量相对较少。在低维特征空间中，模型的参数更新相对稳定，不太容易出现过拟合的情况，因此正则化的作用相对较小。
#
# 训练数据量大：在大规模数据集上训练模型时，由于训练样本数量庞大，模型更容易从数据中学习到一般规律，而不容易受到个别样本的影响。因此，正则化在这种情况下的作用相对较小。
#
# 合适的超参数选择：在ALS-SGD中，通常还需要对学习率等超参数进行调优。合适的超参数选择可以帮助模型更好地学习数据的特征，减少过拟合的风险，从而减弱了正则化的作用。]
# Then I tried to decrease the learning rate dynamically to better converge the model. However, the accuracy is not improved even decreased.
# I've tried exponentially decreasing: accuracy is decreased.
# Cosine annealing: accuracy does not change.
# Polynomial decay increase a little bit.
# I perceive the reason might be as follows:
# 学习率衰减方式不合适： 使用的学习率衰减方式可能不适合当前的问题和数据集。不同的问题和数据集可能需要不同的学习率调整策略。因此，可能需要尝试其他的学习率衰减方式，或者调整衰减方式的超参数。
#
# 初始学习率设置不合理： 初始学习率可能设置得太大或太小，导致学习率衰减后的学习率变化不明显，或者导致模型难以收敛到最优解。可以尝试调整初始学习率的大小，以获得更好的效果。
#
# 模型已经收敛到局部最优解： 在训练过程中，模型可能已经收敛到了一个局部最优解附近，此时即使使用学习率衰减也难以进一步提高模型性能。可以尝试增加训练迭代次数，或者尝试其他的优化算法，如随机梯度下降的变种算法，以进一步提高模型性能。
#
# 数据集特性不适合学习率衰减： 数据集的特性可能不适合使用学习率衰减。例如，如果数据集比较简单或者噪声较少，模型可能不需要过多的学习率调整，直接使用固定的学习率即可获得较好的效果。
# Thirdly, I supposed to use mini-batch SGD to alternate the SGD. However, even with batch size 1, The running time is much longer than before.
