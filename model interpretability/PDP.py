#set stage one regression as the sample

import pandas as pd
import matplotlib.pyplot as plt
from pdpbox import pdp
import numpy as np
from xgboost import XGBRegressor

# read datasets
train_df = pd.read_csv('SR_train.csv')
X_train = train_df[['infP','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time','pH']]
y_train = train_df['SR']

test_df = pd.read_csv('SR_test.csv')
X_test = test_df[['infP','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time','pH']]
y_test = test_df['SR']

#two csvs merged
df = pd.read_csv('SR_all.csv')
X = df[['infP','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time','pH']]

#model development
model =  XGBRegressor (n_estimators=196, max_depth=10,
                    subsample= 0.656, learning_rate= 0.124,gamma=0.159,
                    random_state=42)
model.fit(X_train,y_train)

import seaborn as sns
from sklearn.inspection import partial_dependence
from scipy.interpolate import splev, splrep

features = ['infP','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time','pH']
for i in features:
    pdp = partial_dependence(model, X, [i], kind="both", grid_resolution=50)
    sns.set_theme(style="ticks", palette="deep", font_scale = 1.1)
    fig = plt.figure(figsize=(6, 5), dpi=100)
    ax  = plt.subplot(111)

    plot_x = pd.Series(pdp['values'][0]).rename('x')
    plot_i = pdp['individual'] 
    plot_y = pdp['average'][0]
    tck = splrep(plot_x, plot_y, s=30)
    xnew = np.linspace(plot_x.min(),plot_x.max(),300)
    ynew = splev(xnew, tck, der=0)

    plot_df = pd.DataFrame(columns=['x','y'])
    for a in plot_i[0]:
        a2 = pd.Series(a)
        df_i = pd.concat([plot_x, a2.rename('y')], axis=1)
        plot_df = plot_df.append(df_i)

    sns.lineplot(data=plot_df, x="x", y="y", color='k', linewidth = 1.5, linestyle='--', alpha=0.6)
    plt.plot(xnew, ynew, linewidth=2)
    sns.rugplot(data = df.sample(100), x = i, height=.05, color='k', alpha = 0.3)

    x_min = plot_x.min()-(plot_x.max()-plot_x.min())*0.1
    x_max = plot_x.max()+(plot_x.max()-plot_x.min())*0.1
    plt.title('Partial Dependence Plot of '+i , fontsize=18)
    plt.ylabel('Partial Dependence')
    plt.xlabel(i)
    plt.xlim(x_min,x_max)
    ax.patch.set_facecolor("lightblue")
    ax.patch.set_alpha(0.2)
    plt.show()
