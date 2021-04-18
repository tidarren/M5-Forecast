"""
Optuna example that demonstrates a pruner for LightGBM.
In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.
You can run this example as follows:
    $ python lightgbm_integration.py
"""
import lightgbm as lgb
import numpy as np
import optuna
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt
from utils import load_data


cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + \
            ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]


def objective(trial):
    train_x, train_y, valid_x, valid_y = load_data()
    dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cat_feats, free_raw_data=False)
    dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cat_feats, free_raw_data=False)

    param = {
        "objective": "poisson",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        'num_iterations' : 500,
    }

    gbm = lgb.train(
        param, dtrain, valid_sets=[dvalid], verbose_eval=100
    )

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    rmse = sqrt(mean_squared_error(valid_y, pred_labels))

    return rmse


if __name__ == "__main__":
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=2)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {},".format(key, value))
    
    with open("Params.txt", "a") as f:
        print("\n===\n", file=f)
        for key, value in trial.params.items():
            print("    {}: {},".format(key, value), file=f)