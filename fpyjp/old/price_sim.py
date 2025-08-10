

def bond(per_value=100, coupon=1, redemption_poriod=10, redemption_amount=None):
    coupon_rate = coupon / per_value
    redemption_amount = redemption_amount if redemption_amount else per_value

    return coupon_rate

def stock(amount, stock_price):
    """
    stock_price 株価(円/株) stock_price_time_series_data
    share (株)
    dividend 配当(円/100株)
    """
    share = amount / stock_price  
    return share

def investment_trust(amount, price):
    """
    net_asset_value(NAV) 基準価額(円/10000口)
    unit (口)
    dividend 分配金(円/10000口)
    """
    unit = amount / price  
    return unit

def securities(t=10, r=0.01, present_price=10000, dividend=10, num_of_devidend=4, risk_free_rate=0.01, drift=0.03, market_volatility=0.01, beta_market_risk_premium=1.5, alpha=0.0, volatility=0.01):
    """
    Price:
        株式の場合、株価 Stock Price (円/株)
        投資信託の場合、基準価額 Net Asset Value NAV (円/10000口)
    share (株)
    unit (口)
    dividend 配当(円/100株)
    dividend 分配金(円/10000口)
    """
    time_series_data = pd.DataFrame(0.0, index=range(0, t+1), columns=["rate of return", "price", "dividend"])
    time_series_data.loc[0, "price"] = present_price
    for i in range(1, t+1):
        if isinstance(r, float):
            time_series_data.loc[i, "rate of return"] = r
        elif isinstance(r, str):
            if r == "capm":
                time_series_data.loc[i, "rate of return"] = capm(risk_free_rate, drift, market_volatility, beta_market_risk_premium, alpha, volatility)  # model
            else:
                time_series_data.loc[i, "rate of return"] = 0.01
        else:
            time_series_data.loc[i, "rate of return"] = 0.01
        time_series_data.loc[i, "price"] = int(time_series_data.loc[i-1, "price"] * (1 + time_series_data.loc[i, "rate of return"]))
        if i % num_of_devidend == 1:
            time_series_data.loc[i, "dividend"] = dividend
         
    return time_series_data

def capm(risk_free_rate=0.01, drift=0.03, market_volatility=0.01, beta_market_risk_premium=1.5, alpha=0.0, volatility=0.01):
    market_portfolio_rate_of_return = drift + market_volatility * np.random.normal(0, 1)
    capm_rate_of_return = beta_market_risk_premium * market_portfolio_rate_of_return + (1 - beta_market_risk_premium) * risk_free_rate 
    rate_of_return = capm_rate_of_return + np.random.normal(alpha, volatility)
    return rate_of_return