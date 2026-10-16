import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"datasets/06_house_prices.csv")
X=df[["area_sqft"]]; y=df["price_lakh"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=LinearRegression().fit(X_train,y_train); pred=model.predict(X_test)
print("Slope:",round(model.coef_[0],3),"Intercept:",round(model.intercept_,3))
print("MAE:",round(mean_absolute_error(y_test,pred),2))
print("R2:",round(r2_score(y_test,pred),3))
print("Predicted price for 1250 sq ft:",round(model.predict(pd.DataFrame({"area_sqft":[1250]}))[0],2),"lakh")
plt.scatter(X,y,label="Actual"); plt.plot(X,model.predict(X),label="Regression line")
plt.xlabel("Area (sq ft)"); plt.ylabel("Price (lakh)"); plt.legend(); plt.title("House Price Linear Regression")
plt.tight_layout(); plt.savefig(ROOT/"outputs/practical_06_linear_regression.png"); plt.show()
