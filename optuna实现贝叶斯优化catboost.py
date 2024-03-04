import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold,cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#读取数据
df = pd.read_csv('SR_all.csv')
x=df[['infP','infC','infAc','infpro','infS',
      'MLSS','MLVSS','VSS/TSS','volumn','ana-time',
      'pH','T','salinity']].values
y=df['SR'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=4789)

#定义目标函数与参数空间
def optuna_objective(trial):
    # 定义参数空间
    iterations=trial.suggest_int('iterations',10,300,1)
    depth=trial.suggest_int('depth',3,15,1)
    subsample = trial.suggest_int('subsample', 0.5*1000,1*1000,1)/1000   #控制小数点三位,所以把float转换成int
    learning_rate=trial.suggest_int('learning_rate', 0.03*1000,1*1000,1)/1000
    rsm=trial.suggest_int('rsm', 0.6*1000,0.8*1000,1)/1000

    #定义评估器,需要优化的参数由上述参数空间决定，不需要优化的则填写具体值
    reg=CatBoostRegressor(iterations=iterations,
                          depth=depth,
                          subsample=subsample,
                          learning_rate=learning_rate,
                          rsm=rsm,
                          random_state=42,
                          verbose=False)

    #交叉验证,optuna可以同时支持最大值，也可以支持最小值
    cv=KFold(n_splits=9,shuffle=True,random_state=42)#cv有两种方法 注意这里的shuffle很重要
    validation_loss=cross_validate(reg,X_train,y_train,
                                   scoring='neg_root_mean_squared_error',#这个库只能求最大值，所以寻找RMSE的最小值，就是寻找-RMSE的最大值
                                   cv=cv,
                                   verbose=False,#可以自行决定是否开启森林建树的verbose
                                   error_score='raise')#如果交叉验证的算法执行报错（NAN），则告诉我们错误的理由
    reg.fit(X_train,y_train)
    rf_pred = reg.predict(X_test)
    rf_pred_train = reg.predict(X_train)
    print(np.mean(validation_loss['test_score']))
    print("R2-训练集", r2_score(y_train, rf_pred_train))
    print("R2-测试集", r2_score(y_test, rf_pred))

    #最终输出RMSE
    return np.mean(validation_loss['test_score'])#验证集上的RMSE

#定义优化目标函数的具体流程
def optimizer_optuna(n_trials,algo):

    #定义使用TPE或者GP
    if algo=='TPE':
        algo=optuna.samplers.TPESampler(n_startup_trials=20, n_ei_candidates=24)#TPE
    elif algo=='GP':
        from optuna.integration import SkoptSampler
        import skopt
        algo=SkoptSampler(skopt_kwargs={'base_estimator':'GP',#选择高斯过程
                                        'n_initial_points':30,#初始观测点10个
                                        'acq_func':'EI'})#选择的采集函数为EI，期望增量

    #实际优化过程，首先实例化优化器
    study=optuna.create_study(sampler=algo#要使用的具体算法
                              ,direction='maximize')#优化的方向，可以选择minimize或者maximize

    #开始优化，n_trials为允许的最大迭代次数
    #由于参数空间已经在目标函数中定义好，不再需要输入参数空间
    study.optimize(optuna_objective,
                   n_trials=n_trials,
                   show_progress_bar=False)#是否显示进度条

    #可直接从优化好的对象study中调用优化的结果
    #打印最佳参数与最佳损失值


    print('\n', '\n', 'best params:', study.best_trial.params,
          '\n', '\n', 'best score:', study.best_trial.values,
          '\n')

    return study.best_trial.params,study.best_trial.values

#执行实际化流程

best_params,best_score=optimizer_optuna(50 ,'TPE')

