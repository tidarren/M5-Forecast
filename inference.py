from process_train_valid import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, help='define model trial', default="trial_8")
opt = parser.parse_args()
print(opt)

OUTPUT_NAME = "submission_{}.csv".format(opt.trial)
MODEL_FILE = "model_{}.lgb".format(opt.trial)
m_lgb = lgb.Booster(model_file=MODEL_FILE) 

fday = datetime(2016,5, 23) #d_1942
max_lags = 70
alphas = [1.035, 1.03, 1.025, 1.02]
weights = [1/len(alphas)]*len(alphas)
sub = 0.
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]


for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

    te = create_dt(is_train=False)
    cols = [f"F{i}" for i in range(1,29)]
    
    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(icount, day)
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
        create_fea(tst)
        tst = downcast(tst)
        train_cols = tst.columns[~tst.columns.isin(useless_cols)]
        tst = tst.loc[tst.date == day , train_cols]
        te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev

    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
    te_sub.fillna(0., inplace = True)
    te_sub.sort_values("id", inplace = True)
    te_sub.reset_index(drop=True, inplace = True)
    if icount == 0 :
        sub = te_sub
        sub[cols] *= weight
    else:
        sub[cols] += te_sub[cols]*weight
    print(icount, alpha, weight)


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation", regex=True)
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv(OUTPUT_NAME,index=False)