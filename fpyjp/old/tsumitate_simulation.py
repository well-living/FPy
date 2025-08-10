import collections
import pandas as pd
# 投資信託(口数)または株式(株数)の積み立て
def tsumitate_simulation(time_period: int, return_rate: float=0.005, price=None
         , amount=100.0, end_accumulation=-1, devidend=5.0, num_of_devidend=12, pension=20.0, start_pension=0
         , tax_rate=0.20315, dividend_tax_rate=0.20315, capitalgain_tax_rate=None, pension_tax_rate=None
         , discount_rate=0.001, to_csv_filename=False) -> pd.DataFrame:
    """
    Args:
        time_period (int): 期末時点
        return_rate (float): デフォルトは月次リターンを想定(各期を年間として扱っても問題ない)
        price (float): 投資信託の基準価額または株価
        amount (float): 初期時点の期末積立額
        devidend (float): 投資信託の分配金または株式の配当
        num_of_devidend (int): 分配金または配当の頻度
        pension (float): 年金額
        start_pension (int): 年金受給開始期
        tax_rate (float): 税率. Should be in the closed range [0.0, 1.0]. Defaults to 0.20315.
        dividend_tax_rate (float): 配当所得税率. Should be in the closed range [0.0, 1.0]. Defaults to 0.20315.
        capitalgain_tax_rate (float): 株式等譲渡所得税率. Should be in the closed range [0.0, 1.0]. Defaults to None.
        pension_tax_rate (float): 年金等雑所得税率. Should be in the closed range [0.0, 1.0]. Defaults to None.
        discount_rate (float): 
        to_csv_filename (str): CSV出力
        
    Returns:
        df (pd.DataFrame): シミュレーションテーブル
    
    """
    # シミュレーションテーブル作成
    df = pd.DataFrame(0.0, 
        index=range(0, time_period+1), 
        columns=["Price", "ReturnRate", "計算前期末資産時価"
                 , "DevidendBeforeTax", "利子配当所得税", "Devidend"
                 , "税引前期末年金額", "期末売却口数", "年金利益", "売却損失", "所得税", "税引後期末年金額"
                 , "期末積立額", "期末購入口数", "期末保有口数"
                 , "計算後期末資産時価", "期末資産簿価", "期末NISA簿価", "期末NISAつみたて投資枠簿価", "期末NISA成長投資枠簿価"]
        )

    # 投資信託の基準価額または株価と期待リターンのインプットを確認
    # 基準価額または株価がシーケンスの場合
    is_price = isinstance(price, collections.abc.Sequence) | isinstance(price, pd.core.series.Series)
    # 基準価額または株価のシーケンスの長さが1以上の場合
    if is_price:
        is_price = len(price) > 0
    # 期待リターンがシーケンスの場合
    is_return_rate = isinstance(return_rate, collections.abc.Sequence) | isinstance(return_rate, pd.core.series.Series)
    # 期待リターンのシーケンスの長さが1以上の場合
    if is_return_rate:
        is_return_rate = len(return_rate) > 0
    # 期待リターンがfloatの場合
    isfloat_return_rate = isinstance(return_rate, float)
    if isfloat_return_rate:
        isfloat_return_rate = return_rate > 0

    # 基準価額または株価の設定を優先
    if is_price:
        df.loc[:, "Price"] = price
        df.loc[:, "ReturnRate"] = df.loc[:, "Price"].pct_change().shift(-1).fillna(0.0)
    elif is_return_rate | isfloat_return_rate:
        df.loc[:, "ReturnRate"] = return_rate
        # 初期時点の基準価額または株価
        if isinstance(price, int) | isinstance(price, float):
            df.loc[0, "Price"] = price  # 基準価額または株価
        elif isinstance(amount, int) | isinstance(amount, float):
            df.loc[0, "Price"] = amount  # 残高
        else:
            df.loc[0, "Price"] = 1.0  # 基準価額または株価と残高いずれもなければ初期時点「1」とする
        for i in range(1, time_period+1):
            df.loc[i, "Price"] = df.loc[i-1, "Price"] * (1 + df.loc[i-1, "ReturnRate"])
    else:
        print("return_rate または prices を設定してください")

    # 積立終了期の処理
    if not isinstance(end_accumulation, int):
        end_accumulation = time_period
    elif end_accumulation > time_period:
        end_accumulation = time_period
    elif end_accumulation < 0:
        if - end_accumulation < time_period:
            end_accumulation = time_period + end_accumulation + 1
        else:
            end_accumulation = 0
            amount = 0.0

    # 年金受給開始期の処理
    if not isinstance(start_pension, int):
        start_pension = time_period
        pension = 0.0
    elif start_pension > time_period:
        start_pension = time_period
        pension = 0.0     
    elif start_pension < 0:
        if - start_pension < time_period:
            start_pension = time_period + start_pension + 1
        else:
            start_pension = 0


    # 積立金額
    is_sequence_amount = isinstance(amount, collections.abc.Sequence) | isinstance(amount, pd.core.series.Series)
    if is_sequence_amount:
        if len(amount) == 0:
            is_sequence_amount = False
            amount = 0.0

    if is_sequence_amount:
        df.loc[:, "期末積立額"] = amount
    else:
        if isinstance(amount, float) | isinstance(amount, int):
            df.loc[:, "期末積立額"] = df.loc[:, "期末積立額"].mask(df.index < end_accumulation, amount)
        else:
            amount = 0.0
            df.loc[:, "期末積立額"] = 0.0
    df.loc[time_period, "期末積立額"] = 0.0

    # 購入口数
    df.loc[:, "期末購入口数"] = df.loc[:, "期末積立額"] / df.loc[:, "Price"]

    # 初期時点の保有口数、資産時価、資産簿価
    df.loc[0, "期末保有口数"] = df.loc[0, "期末購入口数"]
    df.loc[0, "計算後期末資産時価"] = df.loc[0, "期末積立額"]
    df.loc[0, "期末資産簿価"] = df.loc[0, "期末積立額"]

    # NISA
    df.loc[0, "期末NISAつみたて投資枠簿価"] = min(df.loc[0, "期末積立額"], 100000.0)
    df.loc[0, "期末NISA成長投資枠簿価"] = min(df.loc[0, "期末積立額"] - df.loc[0, "期末NISAつみたて投資枠簿価"], 2400000.0 / 12)
    df.loc[0, "期末NISA簿価"] = df.loc[0, "期末NISAつみたて投資枠簿価"] + df.loc[0, "期末NISA成長投資枠簿価"]
        
    # 年金額
    is_sequence_pension = isinstance(pension, collections.abc.Sequence) | isinstance(pension, pd.core.series.Series)
    if is_sequence_pension:
        if len(pension) == 0:
            is_sequence_pension = False
            pension = 0.0
        
    if is_sequence_pension:
        df.loc[:, "税引前期末年金額"] = pension
    else:
        if isinstance(pension, float) | isinstance(pension, int):
            df.loc[:, "税引前期末年金額"] = df.loc[:, "税引前期末年金額"].mask(df.index >= start_pension, pension)
        else:
            pension = 0.0
            df.loc[:, "税引前期末年金額"] = 0.0
    df.loc[0, "税引前期末年金額"] = 0.0

    # 税率
    if not isinstance(tax_rate, float):
        tax_rate = 0.0
    else:
        if tax_rate < 0.0:
            tax_rate = 0.0
        elif tax_rate > 1.0:
            tax_rate = 1.0

    # 配当所得税
    if not isinstance(dividend_tax_rate, float):
        dividend_tax_rate = tax_rate

    # 株式売却の譲渡所得税または年金の雑所得税
    if isinstance(capitalgain_tax_rate, float):
        tax_rate = capitalgain_tax_rate
    elif isinstance(pension_tax_rate, float):
        tax_rate = pension_tax_rate
        
    for i in range(1, time_period+1):
        if is_price:
            df.loc[i, "計算前期末資産時価"] = df.loc[i, "Price"] * df.loc[i-1, "期末保有口数"]
        else:
            df.loc[i, "計算前期末資産時価"] = df.loc[i-1, "計算後期末資産時価"] * (1.0 + return_rate)
        # 配当・分配金と利子・配当所得税
        if num_of_devidend == 0:
            #df.loc[i, "DevidendBeforeTax"] = 0.0
            #df.loc[i, "利子配当所得税"] = 0.0
            df.loc[i, "Devidend"] = 0.0
        elif i % num_of_devidend == 0:
            df.loc[i, "DevidendBeforeTax"] = devidend
            df.loc[i, "利子配当所得税"] = df.loc[i, "DevidendBeforeTax"] * dividend_tax_rate
            df.loc[i, "Devidend"] = df.loc[i, "DevidendBeforeTax"] - df.loc[i, "利子配当所得税"]
        # 年金
        if i >= start_pension:
            df.loc[i, "期末売却口数"] = df.loc[i, "税引前期末年金額"] / df.loc[i, "Price"]
            df.loc[i, "年金利益"] = max(min([df.loc[i, "税引前期末年金額"], df.loc[i, "計算前期末資産時価"] - df.loc[i-1, "期末資産簿価"]]), 0)
            df.loc[i, "売却損失"] = min(min([df.loc[i, "税引前期末年金額"], df.loc[i, "計算前期末資産時価"] - df.loc[i-1, "期末資産簿価"]]), 0)
            df.loc[i, "所得税"] = df.loc[i, "年金利益"] * tax_rate
            df.loc[i, "税引後期末年金額"] = df.loc[i, "税引前期末年金額"] - df.loc[i, "所得税"]
            
        df.loc[i, "期末保有口数"] = df.loc[i-1, "期末保有口数"] - df.loc[i, "期末売却口数"] + df.loc[i, "期末購入口数"]

        if is_price:
            df.loc[i, "計算後期末資産時価"] = df.loc[i, "期末保有口数"] * df.loc[i, "Price"]
        else:
            df.loc[i, "計算後期末資産時価"] = df.loc[i, "計算前期末資産時価"] - df.loc[i, "DevidendBeforeTax"] - df.loc[i, "税引前期末年金額"] + df.loc[i, "期末積立額"]
        df.loc[i, "期末資産簿価"] = df.loc[i-1, "期末資産簿価"] + df.loc[i, "期末積立額"] - df.loc[i, "税引前期末年金額"] + df.loc[i, "年金利益"]
        # NISA
        tsumitate_nisa = min(df.loc[i, "期末積立額"], 100000.0)
        if df.loc[i, "期末NISA成長投資枠簿価"] - df.loc[i, "税引前期末年金額"] > 0:
            seicho_nenkin = df.loc[i, "税引前期末年金額"]
            tsumitate_nenkin = 0.0
        else:
            if df.loc[i, "期末NISA成長投資枠簿価"] <= 0:
                if df.loc[i, "期末NISAつみたて投資枠簿価"] - df.loc[i, "税引前期末年金額"] > 0:
                    seicho_nenkin = 0.0
                    tsumitate_nenkin = df.loc[i, "税引前期末年金額"]
                else:
                    seicho_nenkin = 0.0
                    tsumitate_nenkin = df.loc[i, "税引前期末年金額"] - df.loc[i, "期末NISAつみたて投資枠簿価"]
            else:
                seicho_nenkin = df.loc[i, "税引前期末年金額"] - df.loc[i, "期末NISA成長投資枠簿価"]
                tsumitate_nenkin = df.loc[i, "税引前期末年金額"] - seicho_nenkin
        
        df.loc[i, "期末NISAつみたて投資枠簿価"] = min(df.loc[i-1, "期末NISAつみたて投資枠簿価"] + tsumitate_nisa - tsumitate_nenkin + df.loc[i, "売却損失"], 18000000 - df.loc[i-1, "期末NISA成長投資枠簿価"])
        df.loc[i, "期末NISA成長投資枠簿価"] = min(df.loc[i-1, "期末NISA成長投資枠簿価"] + min(df.loc[i, "期末積立額"] - seicho_nenkin - tsumitate_nisa, 2400000.0 / 12), max(18000000 - df.loc[i, "期末NISAつみたて投資枠簿価"], 12000000))        
        df.loc[i, "期末NISA簿価"] = min(df.loc[i, "期末NISAつみたて投資枠簿価"] + df.loc[i, "期末NISA成長投資枠簿価"], 18000000)

    df.loc[:, "累積受給額"] = df.loc[:, "Devidend"].cumsum()
    df.loc[:, "累積年金額"] = df.loc[:, "税引後期末年金額"].cumsum()
    df.loc[:, "累積所得税額"] = df.loc[:, "所得税"].cumsum()
    df.loc[:, "売却した場合の税引前収益"] = df.loc[:, "計算後期末資産時価"] - df.loc[:, "期末資産簿価"]
    df.loc[:, "売却した場合の所得税"] = (df.loc[:, "売却した場合の税引前収益"] * tax_rate).apply(lambda x: max(x, 0))
    df.loc[:, "売却した場合の利益"] = df.loc[:, "売却した場合の税引前収益"] - df.loc[:, "売却した場合の所得税"]
    df.loc[:, "売却した際の受取額"] = df.loc[:, "計算後期末資産時価"] - df.loc[:, "売却した場合の所得税"]
    df.loc[:, "売却した際の累積受取額"] = df.loc[:, "売却した際の受取額"].cumsum()
    
    df.loc[:, "一時売却のキャッシュインフロー"] = df.loc[:, "Devidend"].copy()
    df.loc[start_pension, "一時売却のキャッシュインフロー"] = df.loc[start_pension, "Devidend"] + df.loc[start_pension, "売却した際の受取額"]
    df.loc[:, "一時売却のキャッシュインフロー"] = df.loc[:, "一時売却のキャッシュインフロー"].mask(df.index > start_pension, 0.0)
    df.loc[:, "一時売却のキャッシュフロー"] = df.loc[:, "一時売却のキャッシュインフロー"] - df.loc[:, "期末積立額"]
    df.loc[:, "期末積立額将来価値"] = 0.0
    df.loc[:, "期末積立額将来価値"] = df.loc[:, "期末積立額将来価値"].mask(df.index <= start_pension, df.loc[:, "期末積立額"] * (1 + discount_rate) ** (start_pension - df.index))
    df.loc[:, "一時売却のキャッシュインフローの将来価値"] = 0.0
    df.loc[:, "一時売却のキャッシュインフローの将来価値"] = df.loc[:, "一時売却のキャッシュインフローの将来価値"].mask(df.index <= start_pension, df.loc[:, "一時売却のキャッシュインフロー"] * (1 + discount_rate) ** (start_pension - df.index))
    df.loc[:, "一時売却のキャッシュフローの将来価値"] = df.loc[:, "一時売却のキャッシュインフローの将来価値"] - df.loc[:, "期末積立額将来価値"]
    df.loc[:, "累積将来価値"] = df.loc[:, "一時売却のキャッシュフローの将来価値"].cumsum()

    if isinstance(to_csv_filename, str):
        df.to_csv(to_csv_filename+".csv")
    return df