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


def update_u_z(train_data, lr, u, z, Lambda):
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
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # Compute the gradient of the loss function with respect to u_n and z_q.
    predicted_rating = np.dot(u[n], z[q])
    error = c - predicted_rating

    # Update u_n and z_q.
    u[n] += lr * error * z[q]
    z[q] += lr * error * u[n]

    # Update u_n and z_q.
    u[n] = u[n] + lr * error
    z[q] = z[q] + lr * error


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, Lambda):
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

    #####################################################################                                                            #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z, Lambda)

    mat = np.dot(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, u, z

def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    initial_k = 6
    initial_lr = 0.06
    initial_num_iteration = 80000



if __name__ == "__main__":
    main()

