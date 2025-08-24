# FPy-JP - Financial Planning in Python (Based on the Japanese System)

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


## Quick Start

### Asset and Liability Modeling

```python
from fpyjp.schemas.balance import AssetLiabilitySchema

# Create an asset with price growth - CORRECTED
asset = AssetLiabilitySchema(
    name="Stock Portfolio",
    price=100.0,
    unit=10.0,
    balance=1000.0,
    book_balance=1000.0,  # Required field missing from README
    cashinflow_per_unit=2.0,  # Dividend per share
    rate=0.05  # 5% annual growth
)

print(f"Initial balance: {asset.balance}")
print(f"Book balance: {asset.book_balance}")
```

### Interest Factor Calculations

```python
from fpyjp.core.interest_factor import InterestFactor

# Calculate loan payments - CORRECTED
loan = InterestFactor(rate=0.05, time_period=30, amount=300000)

monthly_payment = loan.calculate_capital_recovery()
print(f"Monthly payment: ${monthly_payment:.2f}")

# Calculate investment growth
investment = InterestFactor(rate=0.07, time_period=20, amount=10000)

future_value = investment.calculate_future_value()
print(f"Future value: ${future_value:.2f}")
```

### Financial Simulation - CORRECTED VERSION

```python
from fpyjp.core.balance_simulator import AssetLiabilitySimulator
from fpyjp.schemas.balance import AssetLiabilitySchema

# Create asset schema properly
asset = AssetLiabilitySchema(
    name="Stock Portfolio",
    price=50.0,
    unit=2000.0,  # 100,000 / 50 = 2000 units
    balance=100000.0,
    book_balance=100000.0,  # Required field
    cashinflow_per_unit=1.0,
    rate=0.08
)

# Create simulator with correct parameter structure
simulator = AssetLiabilitySimulator(
    al_schema=asset,  # Use al_schema parameter
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

## Key Corrections Made

### 1. AssetLiabilitySchema Requirements
- **Added `book_balance`**: This is a required field that was missing from the README
- **Fixed validation**: The `price * unit = balance` relationship must be maintained

### 2. AssetLiabilitySimulator Parameters
- **Use `al_schema`**: The constructor expects `al_schema` parameter, not individual asset parameters
- **Parameter structure**: The README mixed the old-style individual parameters with the new schema-based approach

### 3. Working Bond Example (from notebooks)

```python
from fpyjp.schemas.balance import AssetLiabilitySchema
from fpyjp.core.balance_simulator import AssetLiabilitySimulator

# Bond simulation - 5% coupon, 5-year term
rate = 0.05
time_period = 5
amount = 1

al_schema = AssetLiabilitySchema(
    price=1,
    unit=0,
    balance=0,
    book_balance=0,
    cashinflow_per_unit=rate,  # Coupon payments
    rate=0,  # No price appreciation for bond
)

simulator = AssetLiabilitySimulator(
    al_schema=al_schema,
    initial_cash_balance=0,
    capital_cash_inflow_before_tax=[0] * time_period + [amount],  # Principal repayment
    cash_outflow=amount,  # Initial investment
    income_gain_tax_rate=0,
    capital_gain_tax_rate=0,
)

results = simulator.simulate(n_periods=time_period+1)
```

### 4. Working Loan Example (from notebooks)

```python
from fpyjp.schemas.balance import AssetLiabilitySchema
from fpyjp.core.balance_simulator import AssetLiabilitySimulator
from fpyjp.core.interest_factor import InterestFactor

# Equal payment loan simulation
rate = 0.05
time_period = 5
amount = 1

al_schema = AssetLiabilitySchema(
    price=1,
    unit=-amount,  # Negative for liability
    balance=-amount,
    book_balance=-amount,
    cashinflow_per_unit=0,  # Interest is handled by rate
    rate=rate,  # Interest rate for growing liability
    allow_negative_unit=True  # Required for liabilities
)

monthly_payment = InterestFactor(
    rate=rate, 
    time_period=time_period, 
    amount=amount
).calculate_capital_recovery()

simulator = AssetLiabilitySimulator(
    al_schema=al_schema,
    initial_cash_balance=0,
    capital_cash_inflow_before_tax=0,
    cash_outflow=[0] + [monthly_payment] * time_period,  # Payments start period 1
    income_gain_tax_rate=0,
    capital_gain_tax_rate=0,
)

results = simulator.simulate(n_periods=time_period+1)
```

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