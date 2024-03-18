import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold,cross_validate
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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
    iterations=trial.suggest_int('iterations',10,300,1)
    depth=trial.suggest_int('depth',3,15,1)
    subsample = trial.suggest_int('subsample', 0.5*1000,1*1000,1)/1000
    learning_rate=trial.suggest_int('learning_rate', 0.03*1000,1*1000,1)/1000
    rsm=trial.suggest_int('rsm', 0.6*1000,0.8*1000,1)/1000

    #regressor or classifier
    model=CatBoostRegressor(iterations=iterations,
                          depth=depth,
                          subsample=subsample,
                          learning_rate=learning_rate,
                          rsm=rsm,
                          random_state=42,
                          verbose=False)

    #cross validation
    cv=KFold(n_splits=9,shuffle=True,random_state=42)
    validation_loss=cross_validate(model,X_train,y_train,
                                   scoring='neg_root_mean_squared_error',
                                   cv=cv,
                                   verbose=False,
                                   error_score='raise')
    model.fit(X_train,y_train)

    return np.mean(validation_loss['test_score']) #RMSE on validation sets

#optimization pipeline
def optimizer_optuna(n_trials,algo):


    if algo=='TPE':
        algo=optuna.samplers.TPESampler(n_startup_trials=20, n_ei_candidates=24)#TPE
    elif algo=='GP':
        from optuna.integration import SkoptSampler
        algo=SkoptSampler(skopt_kwargs={'base_estimator':'GP',
                                        'n_initial_points':30,
                                        'acq_func':'EI'})

    study=optuna.create_study(sampler=algo
                              ,direction='maximize')

    study.optimize(optuna_objective,
                   n_trials=n_trials,
                   show_progress_bar=False)

    print('\n', '\n', 'best params:', study.best_trial.params,
          '\n', '\n', 'best score:', study.best_trial.values,
          '\n')

    return study.best_trial.params,study.best_trial.values

#optimization approach
best_params,best_score=optimizer_optuna( 100 ,'TPE')

