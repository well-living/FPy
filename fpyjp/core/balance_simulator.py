# fpyjp/core/balance_simulator.py
"""
Asset and Liability Balance Simulator

Implements spreadsheet simulation using pandas for asset and liability management.
"""

from typing import List, Union, Optional

import pandas as pd

from fpyjp.schemas.balance import AssetLiabilitySchema
from fpyjp.utils.math_utils import safe_divide
from fpyjp.utils.list_utils import get_padded_value_at_period, ensure_list


TAX_RATE = 0.20315  # Default tax rate for income and capital gains

class AssetLiabilitySimulator:
    """
    Asset and Liability Balance Simulator
    
    Simulates asset and liability balances over time periods based on
    cash flows, tax rates, and market conditions. Implements the same
    calculation logic as the provided Excel spreadsheet.
    
    The simulator tracks the evolution of:
    - Cash balances
    - Asset/liability units and balances
    - Book values and unrealized gains/losses
    - Tax calculations on income and capital gains
    - Cash flows from operations and investments
    
    Attributes
    ----------
    al_schema : AssetLiabilitySchema
        AssetLiabilitySchema object containing asset/liability information
    initial_cash_balance : float
        Initial cash balance
    income_gain_tax_rate : Union[float, List[float]]
        Tax rate applied to income gains
    capital_gain_tax_rate : Union[float, List[float]]
        Tax rate applied to capital gains
    capital_cash_inflow_before_tax : List[float]
        Capital cash inflow before tax for each period
    cash_outflow : List[float]
        Cash outflow for each period
    initial_price : float
        Initial price per unit (extracted from al_schema)
    initial_al_balance : float
        Initial asset/liability balance (extracted from al_schema)
    initial_al_book_balance : float
        Initial asset/liability book balance (extracted from al_schema)
    initial_al_unit : float
        Initial number of units (extracted from al_schema)
    cash_inflow_per_unit : List[float]
        Cash inflow per unit for each period (extracted from al_schema)
    rate : List[float]
        Price growth rate for each period (extracted from al_schema)
    """
    
    def __init__(
            self, 
            al_schema: Optional[AssetLiabilitySchema] = None,
            initial_cash_balance: Optional[float] = 0.0,
            initial_al_balance: Optional[float] = None,
            initial_al_book_balance: Optional[float] = None,
            initial_price: Optional[float] = 1.0,
            rate: Optional[Union[float, List[float]]] = None,
            cash_inflow_per_unit: Optional[Union[float, List[float]]] = None,
            capital_cash_inflow_before_tax: Union[float, List[float]] = 0,
            cash_outflow: Union[float, List[float]] = 0,
            income_gain_tax_rate: Union[float, List[float]] = TAX_RATE,
            capital_gain_tax_rate: Union[float, List[float]] = TAX_RATE,
        ):
        """
        Initialize the simulator with initial values and parameters.
        
        Parameters
        ----------
        al_schema : Optional[AssetLiabilitySchema], default None
            AssetLiabilitySchema object containing asset/liability information.
            If provided, values from al_schema take precedence over individual parameters.
        initial_cash_balance : Optional[float], default 0.0
            Initial cash balance
        initial_al_balance : Optional[float], default None
            Initial asset/liability balance (used if al_schema is not provided)
        initial_al_book_balance : Optional[float], default None
            Initial asset/liability book balance (used if al_schema is not provided)
        initial_price : Optional[float], default 1.0
            Initial price per unit (used if al_schema is not provided)
        rate : Optional[Union[float, List[float]]], default None
            Price growth rate for each period (used if al_schema is not provided)
        cash_inflow_per_unit : Optional[Union[float, List[float]]], default None
            Cash inflow per unit for each period (used if al_schema is not provided)
        capital_cash_inflow_before_tax : Union[float, List[float]], default 0
            Capital cash inflow before tax for each period
        cash_outflow : Union[float, List[float]], default 0
            Cash outflow for each period
        income_gain_tax_rate : Union[float, List[float]], default TAX_RATE(0.20315)
            Tax rate for income gains
        capital_gain_tax_rate : Union[float, List[float]], default TAX_RATE(0.20315)
            Tax rate for capital gains
            
        Raises
        ------
        ValueError
            If neither al_schema nor required individual parameters are provided
        """
        self.initial_cash_balance = initial_cash_balance or 0.0
        # Convert scalar inputs to lists if needed using utility function
        self.capital_cash_inflow_before_tax = ensure_list(capital_cash_inflow_before_tax)
        self.cash_outflow = ensure_list(cash_outflow)
        self.income_gain_tax_rate = ensure_list(income_gain_tax_rate)
        self.capital_gain_tax_rate = ensure_list(capital_gain_tax_rate)

        if al_schema is not None:
            # Use provided al_schema
            self.al_schema = al_schema
        else:
            # Create al_schema from individual parameters
            if initial_price is None:
                raise ValueError("initial_price is required when al_schema is not provided")
            if initial_al_balance is None:
                raise ValueError("initial_al_balance is required when al_schema is not provided")
                
            # Calculate unit from price and balance
            unit = safe_divide(initial_al_balance, initial_price, 0.0)
            
            self.al_schema = AssetLiabilitySchema(
                price=initial_price,
                unit=unit,
                balance=initial_al_balance,
                book_balance=initial_al_book_balance or initial_al_balance,
                cashinflow_per_unit=cash_inflow_per_unit,
                rate=rate or 0.0
            )


    
    def _extract_schema_values(self):
        """
        Extract and process values from the AssetLiabilitySchema.
        
        This method extracts validated values from the al_schema and converts them 
        to the appropriate format for simulation calculations.
        """
        # Extract validated values directly (no recalculation needed)
        self.initial_al_balance = self.al_schema.balance
        self.initial_al_book_balance = self.al_schema.book_balance or self.al_schema.balance

        # Extract validated unit
        self.initial_al_unit = self.al_schema.unit

        # Extract price (use first value if list)
        if isinstance(self.al_schema.price, list):
            self.initial_price = self.al_schema.price[0]
        else:
            self.initial_price = self.al_schema.price
        
        # Convert al_schema values to lists using utility function
        self.cash_inflow_per_unit = ensure_list(self.al_schema.cashinflow_per_unit or 0.0)
        self.rate = ensure_list(self.al_schema.rate or 0.0)

    def simulate(self, n_periods: int) -> pd.DataFrame:
        """
        Run the simulation for the specified number of periods.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing simulation results with the following columns:
            
            price : float
                Price per unit at the beginning of each period.
                Calculated as: price[t] = price[t-1] * (1 + rate[t-1])
                
            pre_cash_balance : float
                Cash balance at the beginning of the period (before transactions).
                Equals cash_balance from the previous period.
                
            pre_al_unit : float
                Asset/liability units at the beginning of the period.
                For period 0: Calculated as pre_al_balance / price
                For period 1 onwards: Carried forward from previous period's al_unit
                
            pre_al_balance : float
                Asset/liability balance at the beginning of the period.
                Calculated as: price * pre_al_unit
                
            pre_al_book_balance : float
                Book balance of asset/liability at the beginning of the period.
                Carried forward from previous period's al_book_balance.
                
            pre_unrealized_gl : float
                Unrealized gain/loss at the beginning of the period.
                Calculated as: pre_al_balance - pre_al_book_balance
                
            cash_inflow_per_unit : float
                Specified cash inflow per unit for the period (input parameter).
                
            income_cash_inflow_before_tax : float
                Total income cash inflow before tax.
                Calculated as: pre_al_unit * cash_inflow_per_unit
                
            income_gain_tax_rate : float
                Tax rate applied to income gains (input parameter).
                
            income_gain_tax : float
                Tax on income gains.
                Calculated as: max(income_cash_inflow_before_tax * income_gain_tax_rate, 0)
                
            income_cash_inflow : float
                Net income cash inflow after tax.
                Calculated as: income_cash_inflow_before_tax - income_gain_tax
                
            unit_outflow : float
                Units sold/disposed during the period.
                Calculated as: capital_cash_inflow_before_tax / price
                
            capital_cash_inflow_before_tax : float
                Capital cash inflow before tax (input parameter).
                
            capital_gain_tax_rate : float
                Tax rate applied to capital gains (input parameter).
                
            capital_gain_tax : float
                Tax on capital gains on the gain portion only.
                Calculated as: max(0, (price - avg_book_price) * unit_outflow) * capital_gain_tax_rate
                
            capital_cash_inflow : float
                Net capital cash inflow after tax.
                Calculated as: capital_cash_inflow_before_tax - capital_gain_tax
                
            cash_inflow : float
                Total cash inflow for the period.
                Calculated as: income_cash_inflow + capital_cash_inflow
                
            unit_inflow : float
                Units purchased during the period.
                Calculated as: cash_outflow / price
                
            cash_outflow : float
                Specified cash outflow for the period (input parameter).
                
            cash_flow : float
                Net cash flow for the period.
                Calculated as: cash_inflow - cash_outflow
                
            unit_flow : float
                Net unit flow for the period.
                Calculated as: unit_inflow - unit_outflow
                
            cash_balance : float
                Cash balance at the end of the period.
                Calculated as: pre_cash_balance + cash_flow
                
            al_unit : float
                Asset/liability units at the end of the period.
                Calculated as: pre_al_unit + unit_flow
                
            al_balance : float
                Asset/liability balance at the end of the period.
                Calculated as: al_unit * price
                
            al_book_balance : float
                Book balance at the end of the period.
                Calculated as: 
                    pre_al_book_balance * (1 - unit_outflow/pre_al_unit if pre_al_unit > 0 else 1) + cash_outflow
                
            unrealized_gl : float
                Unrealized gain/loss at the end of the period.
                Calculated as: al_balance - al_book_balance
                
            rate : float
                Price growth rate for the period (input parameter).
        
        Notes
        -----
        The simulation follows these key principles:
        - Units are sold first (unit_outflow), then purchased (unit_inflow)
        - Book balance is adjusted proportionally when units are sold
        - Capital gains tax is only applied to the gain portion of realized gains
        - All cash flows are processed sequentially within each period
        """
        # Extract values from al_schema
        self._extract_schema_values()
        
        # Initialize result DataFrame
        columns = [
            'price', 
            'pre_cash_balance', 'pre_al_unit', 
            'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl', 
            'cash_inflow_per_unit',
            'income_cash_inflow_before_tax', 'income_gain_tax_rate', 'income_gain_tax',
            'income_cash_inflow', 
            'unit_outflow', 
            'capital_cash_inflow_before_tax', 'capital_gain_tax_rate', 'capital_gain_tax', 
            'capital_cash_inflow',
            'cash_inflow', 
            'unit_inflow', 
            'cash_outflow', 
            'cash_flow', 
            'unit_flow',
            'cash_balance', 'al_unit', 
            'al_balance', 'al_book_balance', 'unrealized_gl', 
            'rate'
        ]
        
        df = pd.DataFrame(index=range(n_periods), columns=columns)
        df.index.name = 'time_period'
        
        # Initialize first period values
        current_price = self.initial_price
        current_cash_balance = self.initial_cash_balance
        current_al_unit = self.initial_al_unit
        current_al_balance = self.initial_al_balance
        current_al_book_balance = self.initial_al_book_balance

        print("self.cash_inflow_per_unit",self.cash_inflow_per_unit)
        
        for period in range(n_periods):
            # Get period-specific values using utility function
            cash_inflow_per_unit = get_padded_value_at_period(self.cash_inflow_per_unit, period, pad_mode="last")
            capital_cash_inflow_before_tax = get_padded_value_at_period(self.capital_cash_inflow_before_tax, period)
            cash_outflow = get_padded_value_at_period(self.cash_outflow, period)
            rate = get_padded_value_at_period(self.rate, period, pad_mode="last")
            income_gain_tax_rate = get_padded_value_at_period(self.income_gain_tax_rate, period, pad_mode="last")
            capital_gain_tax_rate = get_padded_value_at_period(self.capital_gain_tax_rate, period, pad_mode="last")
            
            # Set basic values
            df.loc[period, 'price'] = current_price
            df.loc[period, 'pre_cash_balance'] = current_cash_balance
            df.loc[period, 'pre_al_unit'] = current_al_unit
            df.loc[period, 'pre_al_balance'] = current_al_balance
            df.loc[period, 'pre_al_book_balance'] = current_al_book_balance
            df.loc[period, 'pre_unrealized_gl'] = current_al_balance - current_al_book_balance
            df.loc[period, 'cash_inflow_per_unit'] = cash_inflow_per_unit
            df.loc[period, 'rate'] = rate
            
            # Income gain (or loss) calculations
            income_cash_inflow_before_tax = current_al_unit * cash_inflow_per_unit
            df.loc[period, 'income_cash_inflow_before_tax'] = income_cash_inflow_before_tax
            df.loc[period, 'income_gain_tax_rate'] = income_gain_tax_rate
            
            income_gain_tax = max(income_cash_inflow_before_tax * income_gain_tax_rate, 0)
            df.loc[period, 'income_gain_tax'] = income_gain_tax
            
            income_cash_inflow = income_cash_inflow_before_tax - income_gain_tax
            df.loc[period, 'income_cash_inflow'] = income_cash_inflow
            
            # Unit outflow calculation (based on capital cash inflow) - WITH ZERO DIVISION PROTECTION
            unit_outflow = safe_divide(capital_cash_inflow_before_tax, current_price, 0.0)
            df.loc[period, 'unit_outflow'] = unit_outflow
            
            # Capital calculations - FIXED CAPITAL GAINS TAX CALCULATION
            df.loc[period, 'capital_cash_inflow_before_tax'] = capital_cash_inflow_before_tax
            df.loc[period, 'capital_gain_tax_rate'] = capital_gain_tax_rate
            
            # Capital gain tax - only on the gain portion of sold units
            if unit_outflow > 0 and current_al_unit > 0:
                # Calculate average book price per unit using utility function
                avg_book_price = safe_divide(current_al_book_balance, current_al_unit, current_price)
                # Calculate realized gain per unit (only positive gains are taxable)
                gain_per_unit = max(0, current_price - avg_book_price)
                # Calculate total realized gain
                total_realized_gain = gain_per_unit * unit_outflow
                # Apply tax to realized gain only
                capital_gain_tax = total_realized_gain * capital_gain_tax_rate
            else:
                capital_gain_tax = 0.0
                
            df.loc[period, 'capital_gain_tax'] = capital_gain_tax
            
            capital_cash_inflow = capital_cash_inflow_before_tax - capital_gain_tax
            df.loc[period, 'capital_cash_inflow'] = capital_cash_inflow
            
            # Cash flow calculations - WITH ZERO DIVISION PROTECTION
            cash_inflow = income_cash_inflow + capital_cash_inflow
            df.loc[period, 'cash_inflow'] = cash_inflow
            
            unit_inflow = safe_divide(cash_outflow, current_price, 0.0)
            df.loc[period, 'unit_inflow'] = unit_inflow
            df.loc[period, 'cash_outflow'] = cash_outflow
            
            cash_flow = cash_inflow - cash_outflow
            df.loc[period, 'cash_flow'] = cash_flow
            
            unit_flow = unit_inflow - unit_outflow
            df.loc[period, 'unit_flow'] = unit_flow
            
            # Update balances
            new_cash_balance = current_cash_balance + cash_flow
            df.loc[period, 'cash_balance'] = new_cash_balance
            
            new_al_unit = current_al_unit + unit_flow
            df.loc[period, 'al_unit'] = new_al_unit
            
            new_al_balance = new_al_unit * current_price
            df.loc[period, 'al_balance'] = new_al_balance
            
            # Book balance adjustment - WITH ZERO DIVISION PROTECTION
            if current_al_unit == 0:
                book_balance_adjustment_rate = 0  # No units to adjust
                new_al_book_balance = cash_outflow  # All outflow becomes new book balance
            else:
                book_balance_adjustment_rate = safe_divide(unit_outflow, current_al_unit, 0.0)
                new_al_book_balance = (current_al_book_balance - 
                                       current_al_book_balance * book_balance_adjustment_rate + 
                                       cash_outflow)
            
            df.loc[period, 'al_book_balance'] = new_al_book_balance
            
            new_unrealized_gl = new_al_balance - new_al_book_balance
            df.loc[period, 'unrealized_gl'] = new_unrealized_gl
            
            # Update for next period
            current_price = current_price * (1 + rate)
            current_cash_balance = new_cash_balance
            current_al_unit = new_al_unit
            current_al_balance = new_al_balance * (1 + rate)
            current_al_book_balance = new_al_book_balance
            
        return df