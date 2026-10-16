import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"datasets/03_iris_pca.csv")
X=df.drop(columns="species")
X_scaled=StandardScaler().fit_transform(X)
pca=PCA(n_components=2)
components=pca.fit_transform(X_scaled)
print("Explained variance ratio:",pca.explained_variance_ratio_.round(3))
print("Total variance retained:",round(pca.explained_variance_ratio_.sum(),3))
plt.scatter(components[:,0],components[:,1],c=pd.Categorical(df.species).codes)
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2"); plt.title("Iris Dataset - PCA")
plt.tight_layout(); plt.savefig(ROOT/"outputs/practical_03_pca.png"); plt.show()
