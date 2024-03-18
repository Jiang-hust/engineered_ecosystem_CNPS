#set stage one regression as the sample

import pandas as pd
from xgboost import XGBRegressor
import shap
import numpy as np

# read datasets
train_df = pd.read_csv('SR_train.csv')
X_train = train_df[['infP','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time','pH']]
y_train = train_df['SR']

test_df = pd.read_csv('SR_test.csv')
X_test = test_df[['infP','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time','pH']]
y_test = test_df['SR']

# model development
model = XGBRegressor (n_estimators=196, max_depth=10,subsample= 0.656, learning_rate= 0.124,gamma=0.159,random_state=42)
model.fit(X_train,y_train)

# shap values
explainer = shap.TreeExplainer(model)
x = np.concatenate((X_train, X_test), axis=0)
shap_values = explainer.shap_values(x)

# shap plot
shap.summary_plot(shap_values, x, plot_type="dot",show=False)
