import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"datasets/10_multimodal_fake_news.csv")
# Extract simple numeric image features locally: average RGB and image brightness.
features=[]
for rel in df["image_path"]:
    img=Image.open(ROOT/"datasets"/rel).convert("RGB").resize((32,32))
    arr=np.asarray(img,dtype=float)
    features.append([arr[:,:,0].mean(),arr[:,:,1].mean(),arr[:,:,2].mean(),arr.std()])
img_features=pd.DataFrame(features,columns=["red_mean","green_mean","blue_mean","image_std"])
# Combine title and text with TF-IDF, plus simple image statistics.
text=(df["title"].fillna("")+" "+df["text"].fillna(""))
vectorizer=TfidfVectorizer(max_features=150,stop_words="english")
Xt=vectorizer.fit_transform(text).toarray()
X=np.hstack([Xt,img_features.to_numpy()])
y=df["label"]
# Small demo dataset: train/test split is illustrative; results are not statistically reliable.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
model=Pipeline([("scale",StandardScaler()),("classifier",LogisticRegression(max_iter=1000))])
model.fit(X_train,y_train); pred=model.predict(X_test)
print("Demo accuracy:",round(accuracy_score(y_test,pred)*100,2),"%")
print(classification_report(y_test,pred,zero_division=0))
print("Multimodal features = TF-IDF text features + local image colour/statistics.")
print("Educational demonstration only: tiny synthetic dataset; do not use for real news verification.")
