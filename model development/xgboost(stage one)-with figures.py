import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# read datasets
train_df = pd.read_csv('SR_train.csv')
X_train = train_df[['infP','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time']].values
y_train = train_df['SR'].values

test_df = pd.read_csv('SR_test.csv')
X_test = test_df[['infP','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time']].values
y_test = test_df['SR'].values

model =XGBRegressor (n_estimators=196, max_depth=10,
                    subsample= 0.656, learning_rate= 0.124,gamma=0.159,
                    random_state=42)

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

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    #set colors
    train_color = (207/255, 185/255, 128/255)
    test_color=(25/255, 79/255, 103/255)

    # scatter plot
    ax.scatter(y_train, y_train_pred, alpha=1, marker='o', s=100, edgecolor=train_color, color='white', label='Train')
    ax.scatter(y_test, y_test_pred, alpha=1, marker='v', s=100, edgecolor=test_color, color='white', label='Test')

    # linear regression
    for x, y, color in [(y_train, y_train_pred, train_color), (y_test, y_test_pred, test_color)]:
        # LR model
        model = LinearRegression()
        X = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        # model fitting
        model.fit(X, y)
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        # model predict
        y_pred_sorted = model.predict(x_sorted.reshape(-1, 1))
        ax.plot(x_sorted, y_pred_sorted, color=color)

        confidence_interval = 1.96 * np.sqrt(np.sum((y_sorted - y_pred_sorted) ** 2) / (len(y) - 2))
        y1_sorted = y_pred_sorted - confidence_interval
        y2_sorted = y_pred_sorted + confidence_interval

        ax.fill_between(x_sorted, y1_sorted.ravel(), y2_sorted.ravel(), alpha=0.2, color=color)

    # line : y=x
    xline = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    ax.plot(xline, xline, color='blue',linestyle='--',alpha=0.5)

    # x-y-labels
    ax.set_xlabel('Actual Sulfate reduction(mg S/L)',fontsize=18)
    ax.set_ylabel('Predicted Sulfate reduction(mg S/L)',fontsize=18)

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    ax.tick_params(axis='both', labelsize=18)
    ax.legend(loc='upper left',fontsize=15)

    # now determine nice limits by hand:
    binwidth = 5
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    ax_histy.set_ylim(0, 70)
    ax_histx.set_ylim(0, 70)

    # kde
    kde_train_pred = gaussian_kde(y_train_pred)
    kde_test_pred = gaussian_kde(y_test_pred)
    kde_train = gaussian_kde(y_train)
    kde_test = gaussian_kde(y_test)

    train_pred_grid = np.linspace(-5, 115, 100)
    test_pred_grid = np.linspace(-5, 115, 100)
    train_grid = np.linspace(-5, 115, 100)
    test_grid = np.linspace(-5, 115, 100)

    ax_histx.plot(train_grid, kde_train(train_grid) , alpha=0.3, color=train_color)
    ax_histx.plot(test_grid, kde_test(test_grid), alpha=0.3, color=test_color)
    ax_histy.plot(kde_train_pred(train_pred_grid) , train_pred_grid, alpha=0.3, color=train_color)
    ax_histy.plot(kde_test_pred(test_pred_grid) , test_pred_grid, alpha=0.3, color=test_color)

    # fill colors
    ax_histx.fill_between(train_grid, 0, kde_train(train_grid), alpha=0.3, color=train_color)
    ax_histx.fill_between(test_grid, 0, kde_test(test_grid), alpha=0.3, color=test_color)
    ax_histy.fill_betweenx(train_pred_grid, 0, kde_train_pred(train_pred_grid), alpha=0.3, color=train_color)
    ax_histy.fill_betweenx(test_pred_grid, 0, kde_test_pred(test_pred_grid), alpha=0.3, color=test_color)

    # set range
    ax.set_xlim(-5, 115)
    ax.set_ylim(-5, 115)

# define position
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.1, height]

fig = plt.figure(figsize=(9, 8))

#bulid three figures
ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

ax_histx.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax_histx.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax_histy.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax_histy.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

ax_histx.spines['top'].set_visible(False)
ax_histx.spines['right'].set_visible(False)
ax_histx.spines['left'].set_visible(False)

ax_histy.spines['top'].set_visible(False)
ax_histy.spines['bottom'].set_visible(False)
ax_histy.spines['right'].set_visible(False)

ax_histy.set_ylim(-5, 115)
ax_histx.set_ylim(-5, 115)

r2_train = np.round(r2_score(y_train, y_train_pred), 3)
r2_test = np.round(r2_score(y_test, y_test_pred), 3)

# adding text
ax.text(60, 15, f'Train R\u00b2: {r2_train}\nTest R\u00b2: {r2_test}\nTrain RMSE: {RMSE_train}\nTest RMSE: {RMSE_test}',fontsize=15)
ax.text(80, 95, f'y=x',fontsize=20, color='blue',alpha=0.8)
ax.text(80, 40, f'XGB',fontsize=18)

# use the previously defined function
scatter_hist(y_train, y_train_pred, ax, ax_histx, ax_histy)

plt.show()
