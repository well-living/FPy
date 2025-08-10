

def emergency_living_funds(monthly_deposable_income=0, monthly_living_expenses=0, months=3, income_or_expense=None):
    if income_or_expense == "income":
        emergency_living_money = monthly_deposable_income * months
    elif income_or_expense == "expense":
        emergency_living_money = monthly_living_expenses * months
    else:
        emergency_living_money = max(monthly_deposable_income, monthly_living_expenses) * months
    return emergency_living_money


def needs50wants30savings20(income, nws_ratio=None):
    if income > 0:
        if nws_ratio:
            denominator = sum(nws_ratio)
            needs_ratio = nws_ratio[0] / denominator
            savings_ratio = nws_ratio[-1] / denominator
        else:
            needs_ratio = 0.5
            savings_ratio = 0.2

        autonomous_consumption = income * needs_ratio
        savings = income * savings_ratio
        marginal_consumption = income - autonomous_consumption - savings
        return autonomous_consumption, marginal_consumption, savings
    else:
        print("可処分所得が0円より大きくありません。")

