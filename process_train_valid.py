import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from  datetime import datetime, timedelta


FIRST_DAY = 500 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !
max_lags = 70
tr_last = 1941
train_valid_day = datetime(2016, 4, 25) #d_1914
saved_file_dir = "../saved_file/"


CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }


def create_dt(is_train = True, nrows = None, first_day = FIRST_DAY):
    prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    
    # dt = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", 
    dt = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    dt["id"] = dt["id"].str.replace("evaluation$", "validation", regex=True)

    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    # Convert from wide to long format
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    # Combine price data from prices dataframe and days data from calendar dataset.
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return dt


def create_fea(dt):
    
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
        
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            if date_feat_func=="weekofyear":
                dt[date_feat_name] = getattr(dt["date"].dt.isocalendar(), "week").astype("int16")
            else:
                dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")


def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i,t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == np.object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df  


if __name__ == "__main__":

    df = create_dt(is_train=True, first_day= FIRST_DAY)
    create_fea(df)
    df = downcast(df)
    df.dropna(inplace = True)

    useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
    train_cols = df.columns[~df.columns.isin(useless_cols)]

    X_train = df[df["date"]<train_valid_day][train_cols]
    y_train = df[df["date"]<train_valid_day]["sales"]
    X_valid = df[df["date"]>=train_valid_day][train_cols]
    y_valid = df[df["date"]>=train_valid_day]["sales"]

    if not os.path.exists('{}'.format(saved_file_dir)):
        os.mkdir('{}'.format(saved_file_dir))

    with open('{}/X_train.pkl'.format(saved_file_dir), 'wb') as f:
        pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
    with open('{}/y_train.pkl'.format(saved_file_dir), 'wb') as f:
        pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
    with open('{}/X_valid.pkl'.format(saved_file_dir), 'wb') as f:
        pickle.dump(X_valid, f, pickle.HIGHEST_PROTOCOL)
    with open('{}/y_valid.pkl'.format(saved_file_dir), 'wb') as f:
        pickle.dump(y_valid, f, pickle.HIGHEST_PROTOCOL)