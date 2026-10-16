import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,classification_report
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"datasets/05_knn_students.csv")
X=df[["hours_studied","attendance_pct"]]; y=df["passed"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
model=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=3))
model.fit(X_train,y_train); pred=model.predict(X_test)
print("Accuracy:",round(accuracy_score(y_test,pred)*100,2),"%")
print(classification_report(y_test,pred,zero_division=0))
print("Prediction for 6 study hours, 80% attendance:",model.predict(pd.DataFrame([[6,80]],columns=X.columns))[0])
