import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"datasets/04_play_tennis.csv")
X=df.drop(columns="play"); y=df["play"]
prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),X.columns)])
# Dense output is needed by GaussianNB
from sklearn.preprocessing import FunctionTransformer
model=make_pipeline(prep,FunctionTransformer(lambda x:x.toarray() if hasattr(x,"toarray") else x),GaussianNB())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
model.fit(X_train,y_train); pred=model.predict(X_test)
print("Accuracy:",round(accuracy_score(y_test,pred)*100,2),"%")
print(classification_report(y_test,pred,zero_division=0))
sample=pd.DataFrame([{"outlook":"Sunny","temperature":"Cool","humidity":"Normal","wind":"Weak"}])
print("Sample prediction:",model.predict(sample)[0])
