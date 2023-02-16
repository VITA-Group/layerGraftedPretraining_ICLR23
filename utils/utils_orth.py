import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace
import logging
logger = logging.getLogger(__name__)


def dominant_eigenvalue(A):

    N, _ = A.size()
    A = A.double()
    x = torch.rand(N, 1, dtype=A.dtype, device=A.device)

    Ax = (A @ x)
    AAx = (A @ Ax)

    term1 = AAx.permute(1, 0) @ Ax
    term2 = Ax.permute(1, 0) @ Ax


    return term1 / term2

def get_singular_values(ATA):
    N, _ = ATA.size()
    largest = dominant_eigenvalue(ATA)
    if torch.isnan(torch.abs(largest)):
        largest = 0
        print("non sense largest")

    I = torch.eye(N, device=ATA.device)  # noqa
    I = I * largest  # noqa
    tmp = dominant_eigenvalue(ATA - I)
    if torch.isnan(torch.abs(tmp)):
        smallest = 0
        print("non sense smallest")
    else:
        smallest = tmp + largest

    # print("smallest is {}, largest is {}".format(smallest, largest))
    return smallest, largest


def rank_loss(features1, features2):
    '''
    :param features1: n * C
    :param features2: n * C
    :return:
    '''
    features1 = (features1 - features1.mean(0)) / features1.std(0)
    features2 = (features2 - features2.mean(0)) / features2.std(0)

    feature_matrix = torch.mm(features1.permute(1,0), features2) / features1.shape[0]
    smallest, largest = get_singular_values(feature_matrix)
    feature_matrix_loss = (largest - smallest) ** 2

    return feature_matrix_loss


def rank_loss_self(features1, features2):
    '''
    :param features1: n * C
    :param features2: n * C
    :return:
    '''
    features = torch.cat([features1, features2], dim=0)

    features = (features - features.mean(0)) / features.std(0)

    feature_matrix = torch.mm(features.permute(1,0), features) / features.shape[0]
    smallest, largest = get_singular_values(feature_matrix)
    feature_matrix_loss = (largest - smallest) ** 2

    return feature_matrix_loss


def similarity(features1, features2):
    '''
    :param features1: n * C
    :param features2: n * C
    :return:
    '''

    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)

    similarity_mean = (features1 * features2).sum(-1).mean(-1)
    return similarity_mean


def test_single_value_cal():
    A = torch.ones(2,2)
    print("A is {}".format(A))
    smallest, largest = get_singular_values(A)
    print("smallest is {}, largest is {}".format(smallest, largest))
    print(torch.linalg.svd(A))



    A = torch.rand(5,5)
    print("A is {}".format(A))
    smallest, largest = get_singular_values(A)
    print("smallest is {}, largest is {}".format(smallest, largest))
    print(torch.linalg.svd(A))

if __name__ == "__main__":
    test_single_value_cal()