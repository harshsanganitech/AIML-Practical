import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,r2_score
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"datasets/07_salary_svr.csv")
X=df[["experience_years"]]; y=df["salary_lakh"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=make_pipeline(StandardScaler(),SVR(kernel="rbf",C=100,epsilon=0.2))
model.fit(X_train,y_train); pred=model.predict(X_test)
print("MAE:",round(mean_absolute_error(y_test,pred),2))
print("R2:",round(r2_score(y_test,pred),3))
print("Predicted salary for 6 years:",round(model.predict(pd.DataFrame({"experience_years":[6]}))[0],2),"lakh")
