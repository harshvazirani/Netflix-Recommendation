import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
for K in [1,2,3,4]:
    gm, p = common.init(X, K, 0)
    mixture_em, post_em, cost_em = naive_em.run(X, gm, p)
    print(K)
    print(common.bic(X, mixture_em, cost_em))
