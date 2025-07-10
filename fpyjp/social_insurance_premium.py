import pandas as pd
# pip install openpyxl
from decimal import Decimal, ROUND_HALF_UP

def insurance_standard_monthly_remuneration(remuneration_april, remuneration_may, remuneration_june, insurance_premium_table=None):
    """
    標準報酬月額
    """
    if insurance_premium_table is None:
        insurance_premium_table = pd.read_excel("https://www.kyoukaikenpo.or.jp/~/media/Files/shared/hokenryouritu/r5/ippan/r5ippan3.xlsx", 
            sheet_name="東京", skiprows=10, nrows=50).drop("Unnamed: 3", axis=1).iloc[:, :4]
        insurance_premium_table.columns = ["等級", "標準報酬月額", "報酬月額(円以上)", "報酬月額(円未満)"]
    monthly_remuneration = (remuneration_april + remuneration_may + remuneration_june) / 3

    for i in range(len(insurance_premium_table)-1):
        if monthly_remuneration < insurance_premium_table.iloc[i, 3]:
            standard_monthly_remuneration = insurance_premium_table.iloc[i, 1]
            break
        else:
            standard_monthly_remuneration = insurance_premium_table.iloc[-1, 1]
    return standard_monthly_remuneration

def pension_standard_monthly_remuneration(remuneration_april, remuneration_may, remuneration_june, insurance_premium_table=None):
    """
    標準報酬月額
    """
    if insurance_premium_table is None:
        insurance_premium_table = pd.read_excel("https://www.kyoukaikenpo.or.jp/~/media/Files/shared/hokenryouritu/r5/ippan/r5ippan3.xlsx", 
            sheet_name="東京", skiprows=10, nrows=50).drop("Unnamed: 3", axis=1).iloc[:, :4]
        insurance_premium_table.columns = ["等級", "標準報酬月額", "報酬月額(円以上)", "報酬月額(円未満)"]
    monthly_remuneration = (remuneration_april + remuneration_may + remuneration_june) / 3

    for i in range(len(insurance_premium_table)-4):
        if monthly_remuneration < insurance_premium_table.iloc[i+3, 3]:
            standard_monthly_remuneration = insurance_premium_table.iloc[i+3, 1]
            break
        else:
            standard_monthly_remuneration = insurance_premium_table.iloc[-1, 1]
    return standard_monthly_remuneration

def association_health_insurance(standard_monthly_remuneration=0, bonus, prefecture="東京"):
    """
    全国健康保険協会管掌健康保険
    https://www.kyoukaikenpo.or.jp/g7/cat330/sb3150/r04/r4ryougakuhyou3gatukara/
    """
    standard_bonus = min(bonus, 5730000)
    health_insurance_premium = (standard_monthly_remuneration + standard_bonus) * 0.1 / 2
    return health_insurance_premium

def union_health_insurance(standard_monthly_remuneration=0):
    """
    健康保険組合管掌健康保険
    https://www.its-kenpo.or.jp/hoken/jimu/hokenryou/index.html
    """
    standard_bonus = min(bonus, 5730000)
    health_insurance_premium = (standard_monthly_remuneration + standard_bonus) * 0.1 / 2
    return health_insurance_premium


def national_pension_premium(
        premium_revision_rate=None, 
        previous_premium_revision_rate=None, 
        price_change_rate=None,
        real_wage_change_rate=None
    ):
    """
    国民年金保険料
    https://www.nenkin.go.jp/service/kokunen/hokenryo/hokenryo.html
    国民年金保険料の金額は、1カ月あたり16,520円です（令和5年度）。
    """
    base_premium = 17000
    if premium_revision_rate:
        pension_premium = base_premium * premium_revision_rate
        pension_premium = int(Decimal(str(pension_premium)).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP))
    elif isinstance(previous_premium_revision_rate, float) & isinstance(price_change_rate, float) & isinstance(real_wage_change_rate, float):
        premium_revision_rate = previous_premium_revision_rate * (1 + price_change_rate) * (1 + real_wage_change_rate)
        pension_premium = base_premium * premium_revision_rate
        pension_premium = int(Decimal(str(pension_premium)).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP))
    else:
        pension_premium = base_premium
    return pension_premium

def employees_pension_premium(standard_monthly_remuneration, bonus=0):
    """
    厚生年金保険料
    """
    standard_bonus = min(bonus, 1500000)
    premium_rate = 183 / 1000
    pension_premium = (standard_monthly_remuneration + standard_bonus) * premium_rate / 2
    pension_premium = int(Decimal(str(pension_premium)).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP))
    return pension_premium

def employment_insurance_premium(wages, job=None):
    """
    雇用保険 employment_insurance_premium
    https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000108634.html
    労働者災害補償保険 IndustrialAccidentCompensationInsurance
    https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/koyou_roudou/roudoukijun/rousai/rousaihoken06/rousai_hokenritsu_kaitei.html
    労働保険料 
    https://jsite.mhlw.go.jp/osaka-roudoukyoku/hourei_seido_tetsuzuki/roudou_hoken/hourei_seido/gaku.html
    （労働保険料）＝（賃金総額）×労働保険料率（労災保険率＋雇用保険率
    """
    return wages * (2.5 / 1000 + 6 / 1000)
    