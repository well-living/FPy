import pandas as pd

def calc_line_segment(df):
    # columns=["年齢", "金額"]
    line_segment_lst = []
    for i in range(len(df)-1):
        slope = (df.iloc[i+1, 1] - df.iloc[i, 1]) / (df.iloc[i+1, 0] - df.iloc[i, 0])
        intercept = (df.iloc[i+1, 0] * df.iloc[i, 1] - df.iloc[i, 0] * df.iloc[i+1, 1]) / (df.iloc[i+1, 0] - df.iloc[i, 0])
        line_segment_lst.append([df.iloc[i, 0], df.iloc[i+1, 0], intercept, slope])
    line_segment_table = pd.DataFrame(line_segment_lst, columns=["年齢(以上)", "年齢(未満)", "切片", "傾き"])
    line_segment_table.iloc[0, 0] = 0
    line_segment_table.iloc[-1, 1] = 101
    return line_segment_table

def calc_amount_by_age(line_segment_table, or_more=15, less_than=81, colname="金額"):
    amount_by_age_list = []
    for a in range(or_more, less_than):
        intercept_and_slope_by_age = line_segment_table.loc[(a >= line_segment_table["年齢(以上)"]) & (a < line_segment_table["年齢(未満)"])]
        amount_by_age = intercept_and_slope_by_age.iloc[0, 2] + intercept_and_slope_by_age.iloc[0, 3] * a
        amount_by_age_list.append([a, amount_by_age])
    amount_by_age_table = pd.DataFrame(amount_by_age_list, columns=["年齢", colname])
    return amount_by_age_table

def calc_growth_rate(table, amount_colname="金額", rate_colname="上昇率"):
    table[rate_colname] = table[amount_colname].pct_change()
    growth_rate_table = table.iloc[1:, [0, 2]]
    growth_rate_table = growth_rate_table.set_index("年齢")
    return growth_rate_table

def calc_future_amount(age, retirement_age, amount, growth_rate_table, rate_colname="上昇率", amount_colname="将来金額"):
    amount_t = amount * 12
    future_amount_list = [[age, amount_t]]
    for a in range(age, retirement_age-1):
        amount_t *= 1 + growth_rate_table.loc[a, rate_colname]
        future_amount_list.append([a+1, amount_t])
    future_amount_table = pd.DataFrame(future_amount_list, columns=["年齢", amount_colname])
    return future_amount_table

def calc_expenditure_allocated_by_age(rate_of_expenditure_allocated_by_age_talbe, age, lifespan, retirement_age=None):
    filter_lifetime = (rate_of_expenditure_allocated_by_age_talbe.index >= age) & (rate_of_expenditure_allocated_by_age_talbe.index <= lifespan)
    if retirement_age:
        filter_numerator = (rate_of_expenditure_allocated_by_age_talbe.index >= retirement_age) & (rate_of_expenditure_allocated_by_age_talbe.index <= lifespan)
        rate_of_expenditure_allocated_by_age = sum(rate_of_expenditure_allocated_by_age_talbe.loc[filter_numerator, "消費支出(万円)"]) / sum(rate_of_expenditure_allocated_by_age_talbe.loc[filter_lifetime, "消費支出(万円)"])
    else:
        rate_of_expenditure_allocated_by_age = rate_of_expenditure_allocated_by_age_talbe.loc[age, "消費支出(万円)"] / sum(rate_of_expenditure_allocated_by_age_talbe.loc[filter_lifetime, "消費支出(万円)"])
    return rate_of_expenditure_allocated_by_age

def calc_future_expenditure(age, start_age, end_age, consumption_expenditure, rate_of_expenditure_allocated_by_age_talbe):
    filter_age = (rate_of_expenditure_allocated_by_age_talbe.index >= start_age) & (rate_of_expenditure_allocated_by_age_talbe.index <= end_age)
    future_expenditure_table = rate_of_expenditure_allocated_by_age_talbe.loc[filter_age, "消費支出(万円)"] / rate_of_expenditure_allocated_by_age_talbe.loc[age, "消費支出(万円)"] * consumption_expenditure
    return future_expenditure_table

disposable_income_line_segment_table = calc_line_segment(disposable_income_df)
disposable_income_table = calc_amount_by_age(disposable_income_line_segment_table, colname="可処分所得")
income_growth_rate_table = calc_growth_rate(disposable_income_table, "可処分所得", "所得上昇率")
future_income = calc_future_amount(30, 70, 400000, income_growth_rate_table, "所得上昇率", "将来可処分所得")

consumption_expenditure_line_segment_table = calc_line_segment(consumption_expenditure_df)
consumption_expenditure_table = calc_amount_by_age(consumption_expenditure_line_segment_table, or_more=15, less_than=101, colname="消費支出")
consumption_expenditure_table["消費支出(万円)"] = consumption_expenditure_table["消費支出"] / 10000
rate_of_expenditure_allocated_by_age_talbe = consumption_expenditure_table[["年齢", "消費支出(万円)"]].set_index("年齢")
expenditure_allocated_by_age = calc_expenditure_allocated_by_age(rate_of_expenditure_allocated_by_age_talbe, 40, 96)
future_expenditure = calc_future_expenditure(40, 65, 96, 300000, rate_of_expenditure_allocated_by_age_talbe)