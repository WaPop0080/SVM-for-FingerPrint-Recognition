import random

import numpy as np


#
# with open('./DB3_feature/1.txt', 'w', newline='') as f:
#     for i in range(len(M)):
#         if i == len(M):
#             f.write((str(M[i])))
#         else:
#             f.write(str(M[i]) + ' ')
#     f.write('\n')
#
# with open('./DB3_feature/1.txt', 'a', newline='') as f:
#     for i in range(len(M)):
#         if i == len(M):
#             f.write((str(M[i])))
#         else:
#             f.write(str(M[i]) + ' ')
#
# As = []
# Ss = []
# with open('./DB3_feature/1.txt', 'r') as file:
#     att = file.readlines()
#     for i in range(len(att)):
#         A = att[i].split(' ')
#         As.append(A)
#     print(As)
#
#     for i in range(len(As)):
#         for j in range(len(As[i]) - 1):
#             S.append(float(As[i][j]))
#         Ss.append(S)
#     print(Ss)


def writeData(filename, mylist):
    with open(filename, 'a', newline='') as f:
        for j in range(len(mylist)):
            if j == len(mylist):
                f.write((str(mylist[j])))
            else:
                f.write(str(mylist[j]) + ' ')
        f.write('\n')


def readData(filename):
    As = []
    Ss = []
    with open(filename, 'r') as file:
        att = file.readlines()
        A = []
        for i in range(len(att)):
            A = att[i].split(' ')
            As.append(A)
        # print(As)

        for i in range(len(As)):
            S = []
            for j in range(len(As[i]) - 1):
                S.append(float(As[i][j]))
            Ss.append(S)
        # print(Ss)
    return Ss


if __name__ == '__main__':
    # M = []
    # for i in range(10):
    #     M.append(random.uniform(0, 10))
    #
    # print(M)
    #
    # writeData('./DB3_feature/1.txt', M)

    W = readData('./DB3_feature/features.txt')
    print(W)
    print(W[0][0])
    print(len(W))
    print(len(W[0]))
