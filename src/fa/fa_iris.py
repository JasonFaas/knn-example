import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import FactorAnalysis

iris = load_iris()
X, y = iris.data, iris.target


from sklearn.decomposition import PCA
import pandas as pd
pca = PCA().fit(X)
print(pca.explained_variance_ratio_)
print(pd.DataFrame(pca.components_,columns=iris.feature_names))
