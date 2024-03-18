import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import KFold,cross_validate

# read train_data
train_df = pd.read_csv('train_data.csv')
X_train = train_df[['infP','infN','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time','ano-time','pH','SR']]
y_train = train_df['P removal']

# read test_data
test_df = pd.read_csv('test_data.csv')
X_test = test_df[['infP','infN','infC','infS','MLSS','MLVSS','VSS/TSS','ana-time','ano-time','pH','SR']]
y_test = test_df['P removal']

def optuna_objective(trial):
    # hyperparameters space
    n_estimators=trial.suggest_int('n_estimaors',10,300,1)
    max_depth=trial.suggest_int('max_depth',3,15,1)
    max_features = trial.suggest_int('max_features', 1,6,1)
    learning_rate= trial.suggest_int('learning_rate', 0.1*1000,1*1000)/1000  #float
    subsample=trial.suggest_int('subsample', 0.5*1000,1*1000)/1000

    #regressor or classifier
    model=GBR(n_estimators=n_estimators,
            subsample=subsample,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42)

    #cross validation
    cv=KFold(n_splits=9,shuffle=True,random_state=42)
    validation_loss=cross_validate(model,X_train,y_train,
                                   scoring='neg_root_mean_squared_error',
                                   cv=cv,
                                   verbose=False,
                                   error_score='raise')
    model.fit(X_train,y_train)

    #RMSE on validation sets
    return np.mean(validation_loss['test_score'])

#optimization pipeline
def optimizer_optuna(n_trials,algo):

    if algo=='TPE':
        algo=optuna.samplers.TPESampler(n_startup_trials=20, n_ei_candidates=24)
    elif algo=='GP':
        from optuna.integration import SkoptSampler
        import skopt
        algo=SkoptSampler(skopt_kwargs={'base_estimator':'GP',
                                        'n_initial_points':20,
                                        'acq_func':'EI'})

    #bayesian optimization
    study=optuna.create_study(sampler=algo
                              ,direction='maximize')

    study.optimize(optuna_objective,
                   n_trials=n_trials,
                   show_progress_bar=False)


    print('\n', '\n', 'best params:', study.best_trial.params,
          '\n', '\n', 'best score:', study.best_trial.values,
          '\n')

    return study.best_trial.params,study.best_trial.values

best_params,best_score=optimizer_optuna(100 ,'TPE')
