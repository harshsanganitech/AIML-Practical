import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"datasets/08_regularization.csv")
X=df.drop(columns="target"); y=df["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
models={"Linear Regression":LinearRegression(),"Ridge":Ridge(alpha=1.0),"Lasso":Lasso(alpha=0.1,max_iter=10000)}
for name,est in models.items():
    model=make_pipeline(StandardScaler(),est); model.fit(X_train,y_train); pred=model.predict(X_test)
    print(f"{name}: MSE={mean_squared_error(y_test,pred):.3f}, R2={r2_score(y_test,pred):.3f}")
print("Regularization adds a penalty to model coefficients and can reduce overfitting.")
