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

# 从train_data.csv中读取训练数据
train_df = pd.read_csv('5P_train_class.csv')
X_train = train_df[['infP', 'infN', 'infC', 'infS', 'MLSS', 'MLVSS'
    , 'VSS/TSS', 'ana-time', 'ano-time', 'pH', 'SR']].values
y_train = train_df['P class'].values

# 从test_data.csv中读取测试数据
test_df = pd.read_csv('5P_test_class.csv')
X_test = test_df[['infP', 'infN', 'infC', 'infS', 'MLSS', 'MLVSS'
    , 'VSS/TSS', 'ana-time', 'ano-time', 'pH', 'SR']].values
y_test = test_df['P class'].values

# 定义评估器,需要优化的参数由上述参数空间决定，不需要优化的则填写具体值
reg = XGBClassifier(n_estimators=342, max_depth=9, subsample=0.523, learning_rate=0.344, gamma=0.277, random_state=42)
#reg=CatBoostClassifier( iterations=300, depth=13, learning_rate=0.183, rsm=0.690,random_state=42)
#reg=GradientBoostingClassifier(n_estimators=50, max_depth=5, max_features=2, random_state=42)
#reg=RandomForestClassifier(n_estimators=167, max_depth=10, max_features=3, min_samples_leaf=6,random_state=42)
reg.fit(X_train, y_train)

y_pred=reg.predict(X_test)
y_train_pred=reg.predict(X_train)

print("准确率-train",accuracy_score(y_train,y_train_pred))
print("准确率",accuracy_score(y_test,y_pred))
print("精确率",precision_score(y_test,y_pred, average='macro'))
print("召回率",recall_score(y_test,y_pred, average='macro'))
print("F1值",f1_score(y_test,y_pred, average='macro'))

# 定义混淆矩阵
cm = confusion_matrix(y_test, y_pred)   # 纵坐标为true，横坐标为pred
confusion_matrix = np.array(cm)

# 定义标签
labels = ['Low', 'Medium', 'High']

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(confusion_matrix, cmap='Purples',alpha=0.5)

# 设置坐标轴标签
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

ax.set_xlabel('Predicted Label',fontsize=25)
ax.set_ylabel('True Label',fontsize=25)

# 添加数值标注
thresh = confusion_matrix.max() / 2.
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, confusion_matrix[i, j], ha="center", va="center",fontsize=25,
                color="white" if confusion_matrix[i, j] > thresh else "black")

# 添加标题和颜色条
ax.set_title("Confusion Matrix of XGB", fontsize=18)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=14)

# 在保存之前，将绘图元素渲染为'CMYK'颜色格式
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_color('k')  # 设置为黑色

# 显示图像
plt.tight_layout()
plt.show()

# matplot 输出的图片为illustrator可编辑的字体
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

# 保存图像
fig.savefig('/home/user/modeldevelopment-ALL/DSEBPR/返修版model development/图/stage2class/混淆矩阵4.pdf')