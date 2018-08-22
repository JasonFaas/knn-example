import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../countries/4-countries of the world-whitespace_cleanup.csv')
features = ['Population', 'Area (sq. mi.)','Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)','Net migration','Infant mortality (per 1000 births)', 'GDP ($ per capita)']
df = df[['Population', 'Area (sq. mi.)','Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)','Net migration','Infant mortality (per 1000 births)', 'GDP ($ per capita)', 'Literacy (%)']]
df = df.dropna()


target_parameter = 'Literacy (%)'

print("head" + str(df))

from sklearn.preprocessing import StandardScaler
#
print(len(features))
# standards = ['Population', 'Area (sq. mi.)','Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)','Net migration','Infant mortality (per 1000 births)', 'GDP ($ per capita)', target_parameter]
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,[target_parameter]].values
y = StandardScaler().fit_transform(y
                                   )
# Standardizing the features
x = StandardScaler().fit_transform(x)

# print("x" + str(x))
# print("y" + str(y))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print("pca_n_comp:" + str(pca.n_components_))
print("pca_ev:" + str(pca.explained_variance_))
print("pca_evr:" + str(pca.explained_variance_ratio_))
print("pca_sv:" + str(pca.singular_values_))
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

# print("head" + str(principalDf.head()))

finalDf = pd.concat([principalDf, pd.DataFrame(data=y, columns=[target_parameter])], axis = 1)

print("head" + str(finalDf))

finalDf['target'] = finalDf[target_parameter] / 100.0

def test(row):
    return [row['target'], 0.0, 1.0]
finalDf['target_rgb'] = finalDf.apply(test, axis=1)

print("head" + str(finalDf))

# finalDf['target_rgb'] = ['r', 'g', 'b']
# print("head" + str(finalDf))
# print("head" + str(finalDf.loc[:,'target']))




fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
# targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf[target_parameter] == target
#     print(indicesToKeep)
ax.scatter(x=finalDf.loc[:, 'principal component 1']
           , y=finalDf.loc[:, 'principal component 2']
           , c=finalDf.loc[:,'target_rgb']
           , s=50)
# ax.legend(targets)
ax.grid()

plt.show()
