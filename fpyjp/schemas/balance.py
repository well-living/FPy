"""
資産・負債管理システムのスキーマ定義

Pydantic V2を使用した資産・負債のデータモデルを定義します。
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field


from typing import Optional, Union, List
from pydantic import BaseModel, field_validator, model_validator
from pydantic_core import ValidationError


class AssetLiability(BaseModel):
    """
    Asset and Liability model for financial calculations.
    
    This class represents an asset or liability with price, unit, and balance
    information, along with optional book balance, cash inflow per unit, and rate.
    
    Attributes
    ----------
    price : Optional[float]
        Price per unit of the asset/liability
    unit : Optional[float]
        Number of units
    balance : Optional[float]
        Total balance (Asset Liability Balance)
    book_balance : Optional[float]
        Book balance, defaults to balance if not provided
    cashinflow_per_unit : Optional[Union[float, List[float]]]
        Cash inflow per unit, defaults to 0.0 if not provided
    rate : Optional[float]
        Rate value, must be greater than -1, defaults to 0.0 if not provided
    """
    
    price: Optional[float] = None
    unit: Optional[float] = None
    balance: Optional[float] = None
    book_balance: Optional[float] = None
    cashinflow_per_unit: Optional[Union[float, List[float]]] = 0.0
    rate: Optional[float] = 0.0
    
    @field_validator('rate')
    @classmethod
    def validate_rate(cls, v: Optional[float]) -> float:
        """
        Validate that rate is greater than -1.
        
        Parameters
        ----------
        v : Optional[float]
            Rate value to validate
            
        Returns
        -------
        float
            Validated rate value
            
        Raises
        ------
        ValueError
            If rate is not greater than -1
        """
        if v is None:
            return 0.0
        if v <= -1:
            raise ValueError('rate must be greater than -1')
        return v
    
    @model_validator(mode='after')
    def validate_price_unit_balance(self):
        """
        Validate that exactly two of price, unit, and balance are provided,
        and calculate the third value. Also set book_balance if not provided.
        
        Returns
        -------
        AssetLiability
            Self with calculated values
            
        Raises
        ------
        ValueError
            If not exactly two of price, unit, balance are provided,
            or if the calculation results in invalid values
        """
        provided_values = sum([
            self.price is not None,
            self.unit is not None,
            self.balance is not None
        ])
        
        if provided_values != 2:
            raise ValueError(
                'Exactly two of price, unit, and balance must be provided'
            )
        
        # Calculate the missing value
        if self.price is None:
            if self.unit == 0:
                raise ValueError('Cannot calculate price when unit is 0')
            self.price = self.balance / self.unit
        elif self.unit is None:
            if self.price == 0:
                raise ValueError('Cannot calculate unit when price is 0')
            self.unit = self.balance / self.price
        elif self.balance is None:
            self.balance = self.price * self.unit
        
        # Verify the relationship holds
        if abs(self.price * self.unit - self.balance) > 1e-10:
            raise ValueError(
                'price * unit must equal balance'
            )
        
        # Set book_balance to balance if not provided
        if self.book_balance is None:
            self.book_balance = self.balance
            
        return self

class AssetType(str, Enum):
    """資産種別"""
    STOCK = "STOCK"              # 株式
    BOND = "BOND"                # 債券
    MUTUAL_FUND = "MUTUAL_FUND"  # 投資信託
    CASH = "CASH"                # 現金
    REAL_ESTATE = "REAL_ESTATE"  # 不動産


class LiabilityType(str, Enum):
    """負債種別"""
    LOAN = "LOAN"                # ローン
    MORTGAGE = "MORTGAGE"        # 住宅ローン
    CREDIT_CARD = "CREDIT_CARD"  # クレジットカード債務


class InterestRateType(str, Enum):
    """金利種別"""
    FIXED = "FIXED"      # 固定金利
    VARIABLE = "VARIABLE"  # 変動金利


class Balance(BaseModel):
    """
    貸借対照表項目基底クラス
    
    資産・負債の共通属性を定義します。
    """
    
    item_id: str = Field(
        ...,
        description="項目ID",
        min_length=1,
        max_length=50
    )
    
    name: str = Field(
        ...,
        description="項目名",
        min_length=1,
        max_length=200
    )
    
    book_value: Decimal = Field(
        ...,
        description="簿価・取得原価",
        ge=0,
        decimal_places=2
    )
    
    market_value: Decimal = Field(
        ...,
        description="時価・現在価値",
        ge=0,
        decimal_places=2
    )
    
    currency: CurrencyType = Field(
        default=CurrencyType.JPY,
        description="通貨"
    )
    
    acquisition_date: date = Field(
        ...,
        description="取得日・開始日"
    )
    
    notes: Optional[str] = Field(
        default=None,
        description="備考",
        max_length=500
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="最終更新日時"
    )
    
    @field_validator('acquisition_date')
    @classmethod
    def validate_acquisition_date(cls, v: date) -> date:
        """取得日・開始日のバリデーション"""
        if v > date.today():
            raise ValueError('取得日・開始日は今日以前の日付である必要があります')
        return v
    
    # エイリアス用プロパティ
    @property
    def current_value(self) -> Decimal:
        """時価のエイリアス"""
        return self.market_value
    
    @property
    def acquisition_cost(self) -> Decimal:
        """取得原価のエイリアス"""
        return self.book_value


class Asset(Balance):
    """
    資産基底クラス
    
    すべての資産の共通属性を定義します。
    """
    
    asset_type: AssetType = Field(
        ...,
        description="資産種別"
    )
    
    @computed_field
    @property
    def unrealized_gain_loss(self) -> Decimal:
        """未実現損益を計算"""
        return self.market_value - self.book_value
    
    @computed_field
    @property
    def return_rate(self) -> Decimal:
        """収益率を計算（小数）"""
        if self.book_value == 0:
            return Decimal('0')
        return self.unrealized_gain_loss / self.book_value


class Stock(Asset):
    """
    株式クラス
    
    株式特有の属性を定義します。
    """
    
    asset_type: AssetType = Field(
        default=AssetType.STOCK,
        description="資産種別（株式固定）"
    )
    
    ticker_symbol: str = Field(
        ...,
        description="銘柄コード",
        min_length=1,
        max_length=20
    )
    
    company_name: str = Field(
        ...,
        description="会社名",
        min_length=1,
        max_length=200
    )
    
    shares: Decimal = Field(
        ...,
        description="株式数",
        gt=0,
        decimal_places=0
    )
    
    current_price: Decimal = Field(
        ...,
        description="現在株価",
        gt=0,
        decimal_places=2
    )
    
    dividend_yield: Optional[Decimal] = Field(
        default=None,
        description="配当利回り（小数、例: 0.023 = 2.3%）",
        ge=0,
        le=0.3,
        decimal_places=6
    )
    
    market: str = Field(
        ...,
        description="市場（例: 東証プライム、NASDAQ）",
        min_length=1,
        max_length=50
    )
    
    @model_validator(mode='after')
    def validate_market_value(self) -> 'Stock':
        """時価の整合性チェック"""
        expected_value = self.shares * self.current_price
        if abs(self.market_value - expected_value) > Decimal('0.01'):
            raise ValueError('時価は株式数×現在株価と一致する必要があります')
        return self


class Bond(Asset):
    """
    債券クラス
    
    債券特有の属性を定義します。
    """
    
    asset_type: AssetType = Field(
        default=AssetType.BOND,
        description="資産種別（債券固定）"
    )
    
    face_value: Decimal = Field(
        ...,
        description="額面価格",
        gt=0,
        decimal_places=2
    )
    
    coupon_rate: Decimal = Field(
        ...,
        description="クーポン金利（小数、例: 0.005 = 0.5%）",
        ge=0,
        le=0.3,
        decimal_places=6
    )
    
    maturity_date: date = Field(
        ...,
        description="満期日"
    )
    
    credit_rating: Optional[str] = Field(
        default=None,
        description="信用格付け",
        max_length=10
    )
    
    issuer: str = Field(
        ...,
        description="発行体",
        min_length=1,
        max_length=200
    )
    
    @field_validator('maturity_date')
    @classmethod
    def validate_maturity_date(cls, v: date) -> date:
        """満期日のバリデーション"""
        if v <= date.today():
            raise ValueError('満期日は今日より後の日付である必要があります')
        return v
    
    @computed_field
    @property
    def years_to_maturity(self) -> Decimal:
        """満期までの年数を計算"""
        days_to_maturity = (self.maturity_date - date.today()).days
        return Decimal(str(days_to_maturity / 365.25))
    
    @computed_field
    @property
    def annual_coupon(self) -> Decimal:
        """年間クーポン支払額を計算"""
        return self.face_value * self.coupon_rate


class MutualFund(Asset):
    """
    投資信託クラス
    
    投資信託特有の属性を定義します。
    """
    
    asset_type: AssetType = Field(
        default=AssetType.MUTUAL_FUND,
        description="資産種別（投資信託固定）"
    )
    
    fund_code: str = Field(
        ...,
        description="ファンドコード",
        min_length=1,
        max_length=20
    )
    
    units: Decimal = Field(
        ...,
        description="口数",
        gt=0,
        decimal_places=0
    )
    
    net_asset_value: Decimal = Field(
        ...,
        description="基準価額",
        gt=0,
        decimal_places=2
    )
    
    expense_ratio: Decimal = Field(
        ...,
        description="信託報酬（小数、例: 0.002 = 0.2%）",
        ge=0,
        le=0.05,
        decimal_places=6
    )
    
    fund_category: str = Field(
        ...,
        description="ファンド分類",
        min_length=1,
        max_length=100
    )
    
    management_company: str = Field(
        ...,
        description="運用会社",
        min_length=1,
        max_length=200
    )
    
    @model_validator(mode='after')
    def validate_market_value(self) -> 'MutualFund':
        """時価の整合性チェック"""
        expected_value = self.units * self.net_asset_value
        if abs(self.market_value - expected_value) > Decimal('0.01'):
            raise ValueError('時価は口数×基準価額と一致する必要があります')
        return self


class Liability(Balance):
    """
    負債基底クラス
    
    すべての負債の共通属性を定義します。
    """
    
    liability_type: LiabilityType = Field(
        ...,
        description="負債種別"
    )
    
    interest_rate: Decimal = Field(
        ...,
        description="金利（小数、例: 0.025 = 2.5%）",
        ge=0,
        le=0.3,
        decimal_places=6
    )
    
    interest_rate_type: InterestRateType = Field(
        ...,
        description="金利種別"
    )
    
    lender: str = Field(
        ...,
        description="貸付先",
        min_length=1,
        max_length=200
    )
    
    @computed_field
    @property
    def annual_interest(self) -> Decimal:
        """年間利息を計算"""
        return self.market_value * self.interest_rate
    
    @model_validator(mode='after')
    def sync_market_value_with_book_value(self) -> 'Liability':
        """負債では時価を簿価と同期（負債は通常時価評価しない）"""
        self.market_value = self.book_value
        return self
    
    # エイリアス用プロパティ
    @property
    def principal(self) -> Decimal:
        """元本残高のエイリアス"""
        return self.book_value


class Loan(Liability):
    """
    ローンクラス
    
    一般的なローンの属性を定義します。
    """
    
    liability_type: LiabilityType = Field(
        default=LiabilityType.LOAN,
        description="負債種別（ローン固定）"
    )
    
    original_amount: Decimal = Field(
        ...,
        description="当初借入額",
        gt=0,
        decimal_places=2
    )
    
    monthly_payment: Decimal = Field(
        ...,
        description="月次返済額",
        gt=0,
        decimal_places=2
    )
    
    term_months: int = Field(
        ...,
        description="返済期間（月）",
        gt=0
    )
    
    purpose: str = Field(
        ...,
        description="借入目的",
        min_length=1,
        max_length=200
    )
    
    @field_validator('book_value')
    @classmethod
    def validate_principal(cls, v: Decimal, info) -> Decimal:
        """元本残高のバリデーション"""
        if 'original_amount' in info.data and v > info.data['original_amount']:
            raise ValueError('元本残高は当初借入額を超えることはできません')
        return v
    
    @computed_field
    @property
    def remaining_months(self) -> int:
        """残存期間（月）を計算"""
        if self.monthly_payment == 0:
            return 0
        return max(0, int(self.book_value / self.monthly_payment))


class Mortgage(Loan):
    """
    住宅ローンクラス
    
    住宅ローン特有の属性を定義します。
    """
    
    liability_type: LiabilityType = Field(
        default=LiabilityType.MORTGAGE,
        description="負債種別（住宅ローン固定）"
    )
    
    property_address: str = Field(
        ...,
        description="担保物件住所",
        min_length=1,
        max_length=300
    )
    
    property_value: Decimal = Field(
        ...,
        description="担保物件評価額",
        gt=0,
        decimal_places=2
    )
    
    loan_to_value_ratio: Optional[Decimal] = Field(
        default=None,
        description="LTV比率（小数、例: 0.8 = 80%）",
        ge=0,
        le=1,
        decimal_places=6
    )
    
    @model_validator(mode='after')
    def calculate_ltv(self) -> 'Mortgage':
        """LTV比率の自動計算"""
        if self.loan_to_value_ratio is None and self.property_value > 0:
            self.loan_to_value_ratio = self.book_value / self.property_value
        return self
    
    @computed_field
    @property
    def equity(self) -> Decimal:
        """担保物件の純資産価値を計算"""
        return self.property_value - self.book_value


# 使用例
if __name__ == "__main__":
    # 株式の例
    stock = Stock(
        item_id="STK001",
        name="トヨタ自動車",
        ticker_symbol="7203",
        company_name="トヨタ自動車株式会社",
        shares=Decimal("100"),
        current_price=Decimal("2500.00"),
        market_value=Decimal("250000.00"),
        book_value=Decimal("240000.00"),
        acquisition_date=date(2023, 1, 15),
        dividend_yield=Decimal("0.023"),  # 2.3%
        market="東証プライム"
    )
    
    # 債券の例
    bond = Bond(
        item_id="BND001",
        name="日本国債10年",
        face_value=Decimal("1000000.00"),
        coupon_rate=Decimal("0.005"),  # 0.5%
        maturity_date=date(2030, 3, 20),
        issuer="日本国政府",
        market_value=Decimal("1002000.00"),
        book_value=Decimal("1000000.00"),
        acquisition_date=date(2023, 3, 20),
        credit_rating="AAA"
    )
    
    # 投資信託の例
    mutual_fund = MutualFund(
        item_id="MF001",
        name="日経225インデックスファンド",
        fund_code="12345",
        units=Decimal("5000"),
        net_asset_value=Decimal("15000.00"),
        expense_ratio=Decimal("0.002"),  # 0.2%
        fund_category="国内株式インデックス",
        management_company="○○投信株式会社",
        market_value=Decimal("75000000.00"),
        book_value=Decimal("70000000.00"),
        acquisition_date=date(2022, 6, 1)
    )
    
    # 一般ローンの例
    loan = Loan(
        item_id="LN001",
        name="マイカーローン",
        book_value=Decimal("2000000.00"),
        market_value=Decimal("2000000.00"),
        interest_rate=Decimal("0.025"),  # 2.5%
        interest_rate_type=InterestRateType.FIXED,
        lender="○○銀行",
        acquisition_date=date(2023, 4, 1),
        original_amount=Decimal("2500000.00"),
        monthly_payment=Decimal("45000.00"),
        term_months=60,
        purpose="自動車購入"
    )
    
    # 住宅ローンの例
    mortgage = Mortgage(
        item_id="MTG001",
        name="住宅ローン",
        book_value=Decimal("25000000.00"),
        market_value=Decimal("25000000.00"),
        interest_rate=Decimal("0.015"),  # 1.5%
        interest_rate_type=InterestRateType.FIXED,
        lender="○○銀行",
        acquisition_date=date(2022, 4, 1),
        original_amount=Decimal("30000000.00"),
        monthly_payment=Decimal("95000.00"),
        term_months=360,
        purpose="住宅購入",
        property_address="東京都渋谷区○○1-2-3",
        property_value=Decimal("35000000.00")
    )
    
    print("=== 資産情報 ===")
    print(f"株式の未実現損益: {stock.unrealized_gain_loss:,}円")
    print(f"株式の収益率: {stock.return_rate:.4f} ({stock.return_rate * 100:.2f}%)")
    print(f"債券の年間クーポン: {bond.annual_coupon:,}円")
    print(f"債券の満期までの年数: {bond.years_to_maturity:.2f}年")
    print(f"投資信託の未実現損益: {mutual_fund.unrealized_gain_loss:,}円")
    
    print("\n=== 負債情報 ===")
    print(f"ローンの年間利息: {loan.annual_interest:,}円")
    print(f"ローンの残存期間: {loan.remaining_months}ヶ月")
    print(f"住宅ローンの年間利息: {mortgage.annual_interest:,}円")
    print(f"住宅ローンのLTV比率: {mortgage.loan_to_value_ratio:.4f} ({mortgage.loan_to_value_ratio * 100:.2f}%)")
    print(f"担保物件の純資産価値: {mortgage.equity:,}円")