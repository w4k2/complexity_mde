import numpy as np
import problexity as px
from sklearn.datasets import make_classification



clusters = np.arange(1, 20, 1)
class_seps = np.arange(0.01, 1, 0.05)
hypercube = [True]

best_n1 = 100
for n_clusters in clusters:
    for sep in class_seps:
        for hyper in hypercube:
            # X, y = make_classification(n_samples=300, n_features=100, n_informative=10, n_redundant=0, n_repeated=0, n_clusters_per_class=n_clusters, class_sep=sep, hypercube=hyper, random_state=1410)
            X, y = make_classification(n_samples=116, n_features=9, n_informative=9, n_redundant=0, n_repeated=0, n_clusters_per_class=n_clusters, class_sep=sep, hypercube=hyper, random_state=1410)

            # cc = px.ComplexityCalculator()
            cc = px.n1(X, y)
            # cc.fit()
            # complexity = cc.complexity
            if cc < best_n1:
                best_n1 = cc
                print("Clusters: %i | Class_sep: %.3f | Hypercube: %s" % (n_clusters, sep, str(hyper)))
                print(cc)