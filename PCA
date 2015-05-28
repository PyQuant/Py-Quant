# Principal component analysis (PCA) - Reducing dimensionality

from sklearn import datasets
iris=datasets.load_iris()
iris_X=iris.data

# how to do it
from sklearn import decomposition

pca=decomposition.PCA()
iris_pca=pca.fit_transform(iris_X)

# print eigenvalue
pca.explanined_variance_ratio_

pca=decomposition.PCA(n_components=2)
iris_X_prime=pca.fit_transfor(iris_X)
iris_X_prime.shape    # Reducing dimensionality  150 by 4 matrix -> 150 by 2 matrix

pca.explained_variance_ratio_.sum()
# explained PCA
