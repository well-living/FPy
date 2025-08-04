"""
Asset and Liability Balance Simulator

Implements spreadsheet simulation using pandas for asset and liability management.
"""

from typing import List, Union, Optional

import numpy as np
import pandas as pd

from fpyjp.schemas.balance import AssetLiabilitySchema

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
        # Convert scalar inputs to lists if needed
        self.capital_cash_inflow_before_tax = self._ensure_list(capital_cash_inflow_before_tax)
        self.cash_outflow = self._ensure_list(cash_outflow)
        self.income_gain_tax_rate = self._ensure_list(income_gain_tax_rate)
        self.capital_gain_tax_rate = self._ensure_list(capital_gain_tax_rate)

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
            unit = initial_al_balance / initial_price if initial_price != 0 else 0
            
            self.al_schema = AssetLiabilitySchema(
                price=initial_price,
                unit=unit,
                balance=initial_al_balance,
                book_balance=initial_al_book_balance or initial_al_balance,
                cashinflow_per_unit=cash_inflow_per_unit or 0.0,
                rate=rate or 0.0
            )
        
    def _ensure_list(self, value: Union[float, List[float]]) -> List[float]:
        """
        Ensure value is a list.
        
        Parameters
        ----------
        value : Union[float, List[float]]
            Input value that can be scalar or list
            
        Returns
        -------
        List[float]
            List representation of the input value
        """
        if isinstance(value, (int, float)):
            return [float(value)]
        return list(value)
    
    def _extract_schema_values(self):
        """
        Extract and process values from the AssetLiabilitySchema.
        
        This method extracts values from the al_schema and converts them to the
        appropriate format for simulation calculations.
        """
        # Extract price (use first value if list)
        if isinstance(self.al_schema.price, list):
            self.initial_price = self.al_schema.price[0]
        else:
            self.initial_price = self.al_schema.price or 0
            
        # Extract or calculate balance
        if self.al_schema.balance is not None:
            self.initial_al_balance = self.al_schema.balance
        else:
            self.initial_al_balance = self.initial_price * (self.al_schema.unit or 0)
            
        # Extract book balance
        self.initial_al_book_balance = self.al_schema.book_balance or self.initial_al_balance
        
        # Calculate initial units
        self.initial_al_unit = self.al_schema.unit or (
            self.initial_al_balance / self.initial_price if self.initial_price != 0 else 0
        )
        
        # Convert al_schema values to lists
        self.cash_inflow_per_unit = self._ensure_list(self.al_schema.cashinflow_per_unit or 0.0)
        self.rate = self._ensure_list(self.al_schema.rate or 0.0)

    def _get_value_for_period(self, array: List[float], period: int) -> float:
        """
        Get value for a specific period, using the last value if period exceeds array length.
        
        Parameters
        ----------
        array : List[float]
            Array of values indexed by period
        period : int
            Target period index
            
        Returns
        -------
        float
            Value for the specified period, or the last value if period is beyond array length
        """
        if not array:
            return 0.0
        return array[min(period, len(array) - 1)]
    
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
                Tax on capital gains, limited by unrealized gains.
                Calculated as: min(pre_unrealized_gl, capital_cash_inflow_before_tax) * capital_gain_tax_rate
                
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
        - Capital gains tax is only applied to realized gains
        - All cash flows are processed sequentially within each period
        """
        # Extract values from al_schema
        self._extract_schema_values()
        
        # Initialize result DataFrame
        columns = [
            'price', 'pre_cash_balance', 'pre_al_unit', 'pre_al_balance', 
            'pre_al_book_balance', 'pre_unrealized_gl', 'cash_inflow_per_unit',
            'income_cash_inflow_before_tax', 'income_gain_tax_rate', 'income_gain_tax',
            'income_cash_inflow', 'unit_outflow', 'capital_cash_inflow_before_tax',
            'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
            'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow', 'unit_flow',
            'cash_balance', 'al_unit', 'al_balance', 'al_book_balance', 
            'unrealized_gl', 'rate'
        ]
        
        df = pd.DataFrame(index=range(n_periods), columns=columns)
        df.index.name = 'time_period'
        
        # Initialize first period values
        current_price = self.initial_price
        current_cash_balance = self.initial_cash_balance
        current_al_unit = self.initial_al_unit
        current_al_balance = self.initial_al_balance
        current_al_book_balance = self.initial_al_book_balance
        
        for period in range(n_periods):
            # Get period-specific values
            cash_inflow_per_unit = self._get_value_for_period(self.cash_inflow_per_unit, period)
            capital_cash_inflow_before_tax = self._get_value_for_period(self.capital_cash_inflow_before_tax, period)
            cash_outflow = self._get_value_for_period(self.cash_outflow, period)
            rate = self._get_value_for_period(self.rate, period)
            income_gain_tax_rate = self._get_value_for_period(self.income_gain_tax_rate, period)
            capital_gain_tax_rate = self._get_value_for_period(self.capital_gain_tax_rate, period)
            
            # Set basic values
            df.loc[period, 'price'] = current_price
            df.loc[period, 'pre_cash_balance'] = current_cash_balance
            df.loc[period, 'pre_al_unit'] = current_al_unit
            df.loc[period, 'pre_al_balance'] = current_al_balance
            df.loc[period, 'pre_al_book_balance'] = current_al_book_balance
            df.loc[period, 'pre_unrealized_gl'] = current_al_balance - current_al_book_balance
            df.loc[period, 'cash_inflow_per_unit'] = cash_inflow_per_unit
            df.loc[period, 'rate'] = rate
            
            # Income calculations
            income_cash_inflow_before_tax = current_al_unit * cash_inflow_per_unit
            df.loc[period, 'income_cash_inflow_before_tax'] = income_cash_inflow_before_tax
            df.loc[period, 'income_gain_tax_rate'] = income_gain_tax_rate
            
            income_gain_tax = max(income_cash_inflow_before_tax * income_gain_tax_rate, 0)
            df.loc[period, 'income_gain_tax'] = income_gain_tax
            
            income_cash_inflow = income_cash_inflow_before_tax - income_gain_tax
            df.loc[period, 'income_cash_inflow'] = income_cash_inflow
            
            # Unit outflow calculation (based on capital cash inflow)
            unit_outflow = capital_cash_inflow_before_tax / current_price if current_price != 0 else 0
            df.loc[period, 'unit_outflow'] = unit_outflow
            
            # Capital calculations
            df.loc[period, 'capital_cash_inflow_before_tax'] = capital_cash_inflow_before_tax
            df.loc[period, 'capital_gain_tax_rate'] = capital_gain_tax_rate
            
            # Capital gain tax (only on realized gains, limited by unrealized gains)
            unrealized_gl = current_al_balance - current_al_book_balance
            capital_gain_tax = min(unrealized_gl, capital_cash_inflow_before_tax) * capital_gain_tax_rate
            df.loc[period, 'capital_gain_tax'] = capital_gain_tax
            
            capital_cash_inflow = capital_cash_inflow_before_tax - capital_gain_tax
            df.loc[period, 'capital_cash_inflow'] = capital_cash_inflow
            
            # Cash flow calculations
            cash_inflow = income_cash_inflow + capital_cash_inflow
            df.loc[period, 'cash_inflow'] = cash_inflow
            
            unit_inflow = cash_outflow / current_price if current_price != 0 else 0
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
            
            # Book balance adjustment
            if current_al_unit == 0:
                book_balance_adjustment_rate = 1
            else:
                book_balance_adjustment_rate = unit_outflow / current_al_unit
            
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
            current_al_balance = new_al_balance
            current_al_book_balance = new_al_book_balance
            
        return df
    
    @classmethod
    def from_schema(cls, 
                    al_schema: AssetLiabilitySchema,
                    initial_cash_balance: float = 0.0,
                    capital_cash_inflow_before_tax: Union[float, List[float]] = 0,
                    cash_outflow: Union[float, List[float]] = 0,
                    income_gain_tax_rate: Union[float, List[float]] = 0.20315,
                    capital_gain_tax_rate: Union[float, List[float]] = 0.20315):
        """
        Create simulator from AssetLiabilitySchema.
        
        This is a convenience method that creates a simulator with the al_schema
        as the primary parameter source.
        
        Parameters
        ----------
        al_schema : AssetLiabilitySchema
            Schema containing asset/liability information with the following mapping:
            - al_schema.price -> initial_price
            - al_schema.unit -> calculated initial_al_unit
            - al_schema.balance -> initial_al_balance
            - al_schema.book_balance -> initial_al_book_balance
            - al_schema.cashinflow_per_unit -> cash_inflow_per_unit
            - al_schema.rate -> rate
        initial_cash_balance : float, default 0.0
            Initial cash balance (not included in al_schema)
        capital_cash_inflow_before_tax : Union[float, List[float]], default 0
            Capital cash inflow before tax for each period
        cash_outflow : Union[float, List[float]], default 0
            Cash outflow for each period
        income_gain_tax_rate : Union[float, List[float]], default 0.20315
            Tax rate for income gains
        capital_gain_tax_rate : Union[float, List[float]], default 0.20315
            Tax rate for capital gains
            
        Returns
        -------
        AssetLiabilitySimulator
            Configured simulator instance
            
        Notes
        -----
        This method is equivalent to calling the constructor with al_schema as the first parameter.
        """
        return cls(
            al_schema=al_schema,
            initial_cash_balance=initial_cash_balance,
            capital_cash_inflow_before_tax=capital_cash_inflow_before_tax,
            cash_outflow=cash_outflow,
            income_gain_tax_rate=income_gain_tax_rate,
            capital_gain_tax_rate=capital_gain_tax_rate
        )


# Example usage
if __name__ == "__main__":
    # Example 1: Create using AssetLiabilitySchema
    al_schema = AssetLiabilitySchema(
        price=10,
        unit=5,  # This will result in balance = 50
        book_balance=40,
        cashinflow_per_unit=[0.5, 0.5, 0.5],
        rate=[0.05, 0.05, 0.03]
    )
    
    simulator1 = AssetLiabilitySimulator(
        al_schema=al_schema,
        initial_cash_balance=100,
        cash_outflow=[50, 30, 30]
    )
    
    # Example 2: Create without al_schema (al_schema will be created internally)
    simulator2 = AssetLiabilitySimulator(
        initial_cash_balance=100,
        initial_al_balance=50,
        initial_al_book_balance=40,
        initial_price=10,
        rate=[0.05, 0.05, 0.03],
        cash_inflow_per_unit=[0.5, 0.5, 0.5],
        capital_cash_inflow_before_tax=[0, 0, 0],
        cash_outflow=[50, 30, 30]
    )
    
    # Example 3: Using from_schema class method
    simulator3 = AssetLiabilitySimulator.from_schema(
        al_schema=al_schema,
        initial_cash_balance=100,
        cash_outflow=[50, 30, 30]
    )
    
    result = simulator1.simulate(3)
    print("Schema object used:")
    print(simulator1.al_schema)
    print("\nSimulation result:")
    print(result)