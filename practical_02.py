import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"datasets/02_clustering.csv")
X=StandardScaler().fit_transform(df)
model=KMeans(n_clusters=3,random_state=42,n_init=10)
labels=model.fit_predict(X)
df["cluster"]=labels
print(df.groupby("cluster").mean().round(2))
plt.scatter(df.feature_x,df.feature_y,c=labels)
plt.xlabel("Feature X"); plt.ylabel("Feature Y"); plt.title("K-Means Clustering")
plt.tight_layout(); plt.savefig(ROOT/"outputs/practical_02_clusters.png"); plt.show()
df.to_csv(ROOT/"outputs/practical_02_clustered.csv",index=False)
