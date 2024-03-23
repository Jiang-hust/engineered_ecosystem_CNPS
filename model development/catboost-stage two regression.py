import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

# read datasets
train_df = pd.read_csv('P_train_regression.csv')
X_train = train_df[['infP','infN','infC','infS','MLSS','MLVSS'
    ,'VSS/TSS','ana-time','ano-time','pH','SR']].values
y_train = train_df['P removal'].values

test_df = pd.read_csv('P_test_regression.csv')
X_test = test_df[['infP','infN','infC','infS','MLSS','MLVSS'
    ,'VSS/TSS','ana-time','ano-time','pH','SR']].values
y_test = test_df['P removal'].values

model = CatBoostRegressor(iterations=275, depth=8, subsample=0.830,
                          learning_rate=0.085,rsm=0.609,random_state=42)
cv=KFold(n_splits=9,shuffle=True,random_state=42)
score_val=cross_val_score(model,X_train, y_train,cv=cv)
model.fit(X_train,y_train)
y_test_pred=model.predict(X_test)
y_train_pred=model.predict(X_train)

RMSE_test=np.round(mean_squared_error(y_test,y_test_pred)**0.5, 2)
RMSE_train=np.round(mean_squared_error(y_train,y_train_pred)**0.5, 2)

print("R2",r2_score(y_test,y_test_pred))
print("R2-train",model.score(X_train,y_train))
print("R2-test",model.score(X_test,y_test))
