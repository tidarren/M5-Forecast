import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from utils import load_data, load_param
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, help='define model trial', default="trial_42")
opt = parser.parse_args()
print(opt)

cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + \
            ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

param = load_param(trial=opt.trial)

train_x, train_y, valid_x, valid_y = load_data()
dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cat_feats, free_raw_data=False)
dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cat_feats, free_raw_data=False)

gbm = lgb.train(
    param, dtrain, valid_sets=[dvalid], verbose_eval=100
)

preds = gbm.predict(valid_x)
pred_labels = np.rint(preds)
rmse = sqrt(mean_squared_error(valid_y, pred_labels))
print('rmse: {}'.format(rmse))

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
MODEL_FILE = "model_{}.lgb".format(opt.trial)

gbm.save_model(os.path.join(MODEL_DIR,MODEL_FILE))