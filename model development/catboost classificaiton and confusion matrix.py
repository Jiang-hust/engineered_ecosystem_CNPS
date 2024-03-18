import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import shap
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,\
    recall_score,f1_score,roc_auc_score, roc_curve,confusion_matrix,classification_report

# read datasets
train_df = pd.read_csv('P_train_class.csv')
X_train = train_df[['infP','infN','infC','infS','MLSS','MLVSS'
    ,'VSS/TSS','ana-time','ano-time','pH','SR']].values
y_train = train_df['P class'].values

test_df = pd.read_csv('P_test_class.csv')
X_test = test_df[['infP', 'infN', 'infC', 'infS', 'MLSS', 'MLVSS'
    , 'VSS/TSS', 'ana-time', 'ano-time', 'pH', 'SR']].values
y_test = test_df['P class'].values

# model development
#model = XGBClassifier(n_estimators=342, max_depth=9, subsample=0.523, learning_rate=0.344, gamma=0.277, random_state=42)
model=CatBoostClassifier( iterations=300, depth=13, learning_rate=0.183, rsm=0.690,random_state=42)
#model=GradientBoostingClassifier(n_estimators=50, max_depth=5, max_features=2, random_state=42)
#model=RandomForestClassifier(n_estimators=167, max_depth=10, max_features=3, min_samples_leaf=6,random_state=42)
reg.fit(X_train, y_train)

y_pred=model.predict(X_test)
y_train_pred=model.predict(X_train)

print("accuracy-train",accuracy_score(y_train,y_train_pred))
print("accuracy",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred, average='macro'))
print("recall",recall_score(y_test,y_pred, average='macro'))
print("F1",f1_score(y_test,y_pred, average='macro'))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)   # x-true，y-pred
confusion_matrix = np.array(cm)

# define labels
labels = ['Low', 'Medium', 'High']

# confusion matrix drawing
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(confusion_matrix, cmap='Purples',alpha=0.5)

# set labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

ax.set_xlabel('Predicted Label',fontsize=25)
ax.set_ylabel('True Label',fontsize=25)

# add numbers labelling
thresh = confusion_matrix.max() / 2.
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, confusion_matrix[i, j], ha="center", va="center",fontsize=25,
                color="white" if confusion_matrix[i, j] > thresh else "black")

# add titles
ax.set_title("Confusion Matrix of XGB", fontsize=18)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=14)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_color('k')  

# show
plt.tight_layout()
plt.show()
