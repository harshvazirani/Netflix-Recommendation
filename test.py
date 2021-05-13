import numpy as np
import em
import common

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

K = [12]
seed = [1]
n, d = X.shape
for k in K:
    for s in seed:
        mixture, post = common.init(X, k, s)
        mixture, post, cost = em.run(X, mixture, post)
        print("K = " + str(k) + " " + "Seed = " + str(s) + " " + str(cost))
        X_pred = em.fill_matrix(X, mixture)
        print(common.rmse(X_gold, X_pred))

# TODO: Your code here