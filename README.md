# FPy - Financial Planning in Python (Based on the Japanese System)

A comprehensive Python package for financial planning and analysis, providing tools for cash flow modeling, asset/liability management and financial simulations, based on the Japanese system.

## Features

- **Cash Flow Analysis**: Define and analyze complex cash flow patterns with flexible timing
- **Interest Factor Calculations**: Comprehensive time value of money calculations
- **Asset and Liability Management**: Model and simulate financial assets and liabilities with price dynamics
- **Financial Simulations**: Run detailed balance sheet simulations with tax considerations
- **Pydantic Integration**: Type-safe data models with automatic validation

## Installation

```bash
pip install fpyjp
```

## Requirements

- Python >= 3.12
- numpy
- pandas
- pydantic

## Quick Start

### Cash Flow Modeling

```python
from fpyjp.schemas.cashflow import CashflowSchema

# Monthly salary for 5 years
salary = CashflowSchema(
    name="Salary",
    amount=5000.0,
    sign=1,  # Inflow
    n_periods=60,  # 5 years * 12 months
    start=0,
    end=59
)

# Variable bonus payments
bonus = CashflowSchema(
    name="Annual Bonus",
    amount=[10000.0, 12000.0, 15000.0],  # Increasing over time
    sign=1,
    start=12  # Start after first year
)
```

### Asset and Liability Modeling

```python
from fpyjp.schemas.balance import AssetLiabilitySchema

# Create an asset with price growth
asset = AssetLiabilitySchema(
    name="Stock Portfolio",
    price=100.0,
    unit=10.0,
    balance=1000.0,
    cashinflow_per_unit=2.0,  # Dividend per share
    rate=0.05  # 5% annual growth
)

print(f"Initial balance: {asset.balance}")
print(f"Book balance: {asset.book_balance}")
```

### Interest Factor Calculations

```python
from fpyjp.schemas.interest_factor import InterestFactor

# Calculate loan payments
loan = InterestFactor(rate=0.05, time_period=30, amount=300000)

monthly_payment = loan.calculate_capital_recovery()
print(f"Monthly payment: ${monthly_payment:.2f}")

# Calculate investment growth
investment = InterestFactor(rate=0.07, time_period=20, amount=10000)

future_value = investment.calculate_future_value()
print(f"Future value: ${future_value:.2f}")
```

### Financial Simulation

```python
from fpyjp.core.balance_simulator import AssetLiabilitySimulator

# Create simulator with initial conditions
simulator = AssetLiabilitySimulator(
    al_schema=asset,
    initial_cash_balance=5000.0,
    capital_cash_inflow_before_tax=0.0,
    cash_outflow=500.0,  # Monthly investment
    income_gain_tax_rate=0.20315,
    capital_gain_tax_rate=0.20315
)

# Run 60-month simulation
results = simulator.simulate(n_periods=60)

# Analyze results
print(f"Final cash balance: {results['cash_balance'].iloc[-1]:.2f}")
print(f"Final asset value: {results['al_balance'].iloc[-1]:.2f}")
print(f"Total unrealized gains: {results['unrealized_gl'].iloc[-1]:.2f}")
```

## Core Components

### CashflowSchema

Flexible cash flow modeling with:

- **Scalar or List Amounts**: Single values or time-varying amounts
- **Timing Control**: Start, end, and step parameters
- **Direction**: Inflow (+1) or outflow (-1) specification
- **Automatic Validation**: Period consistency and amount validation

### AssetLiabilitySchema

Models financial assets and liabilities with the following features:

- **Price and Unit Tracking**: Automatic balance calculation and validation
- **Price Dynamics**: Support for growth rates and price evolution over time
- **Cash Flows**: Income generation (dividends, interest, rent)
- **Book Value Management**: Separate tracking of book vs. market values

Key validation rules:
- `price * unit = balance` relationship is maintained
- `price[i] * (1 + rate[i]) = price[i+1]` for price evolution
- Growth rates must be greater than -1

### InterestFactor

Time value of money calculations:

- **Future Value**: Compound interest calculations
- **Present Value**: Discounting future cash flows
- **Annuities**: Regular payment series analysis
- **Loan Calculations**: Payment schedules and amortization
- **Investment Analysis**: Growth projections and comparisons

### AssetLiabilitySimulator

Comprehensive financial simulation engine:

- **Period-by-Period Analysis**: Detailed tracking of all financial metrics
- **Tax Calculations**: Separate income and capital gains tax handling
- **Cash Flow Integration**: Income, capital, and operational cash flows
- **Balance Sheet Evolution**: Assets, liabilities, and book values over time

Simulation outputs include:
- Price evolution and market values
- Cash balances and flows
- Unit transactions (buy/sell)
- Tax calculations and net cash flows
- Unrealized gains and losses

## Advanced Usage

### Complex Asset Modeling

```python
# Asset with time-varying growth rates
complex_asset = AssetLiabilitySchema(
    name="Real Estate",
    price=[100000, 105000, 112000, 118000],  # Historical prices
    unit=1.0,
    cashinflow_per_unit=[500, 520, 540, 560],  # Increasing rent
    # Rate is automatically calculated from price evolution
)
```

### Comprehensive Financial Planning

```python
# Combine multiple components for complete financial plan
portfolio = AssetLiabilitySchema(
    name="Investment Portfolio",
    balance=100000.0,
    price=50.0,
    rate=0.08,
    cashinflow_per_unit=1.0
)

simulator = AssetLiabilitySimulator(
    al_schema=portfolio,
    initial_cash_balance=10000.0,
    capital_cash_inflow_before_tax=0.0,
    cash_outflow=[1000] * 120 + [0] * 180,  # Invest for 10 years, then hold
    income_gain_tax_rate=0.15,
    capital_gain_tax_rate=0.20
)

# 25-year simulation
long_term_results = simulator.simulate(n_periods=300)

# Analyze key metrics
total_return = (long_term_results['al_balance'].iloc[-1] + 
                long_term_results['cash_balance'].iloc[-1] - 
                100000 - 10000)
print(f"Total portfolio return: ${total_return:.2f}")
```

## Data Validation

FPy uses Pydantic for comprehensive data validation:

- **Type Safety**: Automatic type checking and conversion
- **Range Validation**: Constraints on numerical values
- **Relationship Validation**: Complex inter-field validations
- **Error Messages**: Clear, actionable error descriptions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or feature requests, please open an issue on the GitHub repository.