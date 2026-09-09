import pandas as pd
imp = pd.read_csv("results/xgboost_feature_importance.csv", index_col=0)
print(imp.loc[["Lead_Time","Unit_Cost","Holding_Cost_Rate","Ordering_Cost","Stockout_Penalty"]])