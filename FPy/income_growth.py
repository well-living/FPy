import pandas as pd

#from estat import get_metainfo, get_statsdata, statsdata_to_dataframe, missing_to_nan, cleansing_statsdata, colname_to_japanese

def value_to_age(value):
    cond = value["世帯主の年齢階級"] != "平均"
    cond &= value["世帯主の年齢階級"] != "65歳以上"
    return value.loc[cond,['時間軸（年次）',"世帯主の年齢階級","値"]].rename({"値":"世帯主の年齢"},axis=1)

def value_to_income(value):
    cond = value["世帯主の年齢階級"] != "平均"
    cond &= value["世帯主の年齢階級"] != "65歳以上"
    return value.loc[cond,['時間軸（年次）',"世帯主の年齢階級","値"]].rename({"値":"可処分所得"},axis=1)


