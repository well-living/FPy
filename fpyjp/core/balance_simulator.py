# fpyjp/core/balance_simulator.py
"""
Asset and Liability Balance Simulator

Implements spreadsheet simulation using pandas for asset and liability management.
"""

from typing import List, Union, Optional

import pandas as pd

from fpyjp.utils.math_utils import safe_divide
from fpyjp.utils.list_utils import get_padded_value_at_period, ensure_list
from fpyjp.schemas.balance import AssetLiabilitySchema
from fpyjp.core.interest_factor import InterestFactor


TAX_RATE = 0.0  # Default tax rate for income and capital gains

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
    allow_negative_unit : bool
        Whether to allow negative unit values (for short positions)
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
            allow_negative_unit: bool = False
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
        income_gain_tax_rate : Union[float, List[float]], default TAX_RATE
            Tax rate for income gains
        capital_gain_tax_rate : Union[float, List[float]], default TAX_RATE
            Tax rate for capital gains
        allow_negative_unit : bool, default False
            Whether to allow negative unit values (for short positions).
            Used when al_schema is not provided.
            
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
            # Extract allow_negative_unit from al_schema
            self.allow_negative_unit = al_schema.allow_negative_unit
        else:
            # Create al_schema from individual parameters
            if initial_price is None:
                raise ValueError("initial_price is required when al_schema is not provided")
            if initial_al_balance is None:
                raise ValueError("initial_al_balance is required when al_schema is not provided")
                
            # Calculate unit from price and balance
            unit = safe_divide(initial_al_balance, initial_price, 0.0)
            
            # Store allow_negative_unit flag
            self.allow_negative_unit = allow_negative_unit
            
            self.al_schema = AssetLiabilitySchema(
                price=initial_price,
                unit=unit,
                balance=initial_al_balance,
                book_balance=initial_al_book_balance or initial_al_balance,
                cashinflow_per_unit=cash_inflow_per_unit,
                rate=rate or 0.0,
                allow_negative_unit=allow_negative_unit
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
                Can be negative if short positions are allowed.
                
            pre_al_balance : float
                Asset/liability balance at the beginning of the period.
                Calculated as: price * pre_al_unit
                Can be negative for short positions.
                
            pre_al_book_balance : float
                Book balance of asset/liability at the beginning of the period.
                Carried forward from previous period's al_book_balance.
                Can be negative for short positions.
                
            pre_unrealized_gl : float
                Unrealized gain/loss at the beginning of the period.
                Calculated as: pre_al_balance - pre_al_book_balance
                
            cash_inflow_per_unit : float
                Specified cash inflow per unit for the period (input parameter).
                For short positions, this may represent cash outflow.
                
            income_cash_inflow_before_tax : float
                Total income cash inflow before tax.
                Calculated as: pre_al_unit * cash_inflow_per_unit
                Can be negative for short positions.
                
            income_gain_tax_rate : float
                Tax rate applied to income gains (input parameter).
                
            income_gain_tax : float
                Tax on income gains.
                Calculated as: max(income_cash_inflow_before_tax * income_gain_tax_rate, 0)
                Only positive income is taxed.
                
            income_cash_inflow : float
                Net income cash inflow after tax.
                Calculated as: income_cash_inflow_before_tax - income_gain_tax
                
            unit_outflow : float
                Units sold/disposed during the period.
                Calculated as: capital_cash_inflow_before_tax / price
                For short positions, negative outflow means covering short positions.
                
            capital_cash_inflow_before_tax : float
                Capital cash inflow before tax (input parameter).
                
            capital_gain_tax_rate : float
                Tax rate applied to capital gains (input parameter).
                
            capital_gain_tax : float
                Tax on capital gains on the gain portion only.
                Calculated differently for long and short positions:
                - Long: max(0, (price - avg_book_price) * unit_outflow) * tax_rate
                - Short: max(0, (avg_book_price - price) * abs(unit_outflow)) * tax_rate
                
            capital_cash_inflow : float
                Net capital cash inflow after tax.
                Calculated as: capital_cash_inflow_before_tax - capital_gain_tax
                
            cash_inflow : float
                Total cash inflow for the period.
                Calculated as: income_cash_inflow + capital_cash_inflow
                
            unit_inflow : float
                Units purchased during the period.
                Calculated as: cash_outflow / price
                For short positions, this represents short selling.
                
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
                Can be negative for short positions.
                
            al_balance : float
                Asset/liability balance at the end of the period.
                Calculated as: al_unit * price
                Can be negative for short positions.
                
            al_book_balance : float
                Book balance at the end of the period.
                For long positions: pre_al_book_balance * (1 - unit_outflow/pre_al_unit if pre_al_unit > 0) + cash_outflow
                For short positions: Similar logic but handles negative units appropriately.
                
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
        - For short positions:
          - Negative units represent short positions
          - Income flows may be negative (e.g., dividend payments on short positions)
          - Capital gains are calculated as (avg_book_price - current_price) for shorts
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
            # For short positions, this may be negative (e.g., dividend payments)
            income_cash_inflow_before_tax = current_al_unit * cash_inflow_per_unit
            df.loc[period, 'income_cash_inflow_before_tax'] = income_cash_inflow_before_tax
            df.loc[period, 'income_gain_tax_rate'] = income_gain_tax_rate
            
            # Only tax positive income gains
            income_gain_tax = max(income_cash_inflow_before_tax * income_gain_tax_rate, 0)
            df.loc[period, 'income_gain_tax'] = income_gain_tax
            
            income_cash_inflow = income_cash_inflow_before_tax - income_gain_tax
            df.loc[period, 'income_cash_inflow'] = income_cash_inflow
            
            # Unit outflow calculation (based on capital cash inflow) - WITH ZERO DIVISION PROTECTION
            unit_outflow = safe_divide(capital_cash_inflow_before_tax, current_price, 0.0)
            df.loc[period, 'unit_outflow'] = unit_outflow
            
            # Capital calculations - ENHANCED FOR SHORT POSITIONS
            df.loc[period, 'capital_cash_inflow_before_tax'] = capital_cash_inflow_before_tax
            df.loc[period, 'capital_gain_tax_rate'] = capital_gain_tax_rate
            
            # Capital gain tax calculation - handles both long and short positions
            capital_gain_tax = 0.0
            if unit_outflow != 0 and current_al_unit != 0:
                # Calculate average book price per unit using utility function
                avg_book_price = safe_divide(current_al_book_balance, current_al_unit, current_price)
                
                if current_al_unit > 0:  # Long position
                    # For long positions: gain = (current_price - avg_book_price) * units_sold
                    gain_per_unit = max(0, current_price - avg_book_price)
                    total_realized_gain = gain_per_unit * unit_outflow
                elif current_al_unit < 0:  # Short position
                    # For short positions: gain = (avg_book_price - current_price) * abs(units_covered)
                    # Note: unit_outflow for covering shorts should be negative
                    if unit_outflow < 0:  # Covering short position
                        gain_per_unit = max(0, avg_book_price - current_price)
                        total_realized_gain = gain_per_unit * abs(unit_outflow)
                    else:  # Adding to short position (rare case with positive unit_outflow on short)
                        total_realized_gain = 0.0
                
                # Apply tax to realized gain only
                capital_gain_tax = total_realized_gain * capital_gain_tax_rate
                
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
            
            # Book balance adjustment - ENHANCED FOR SHORT POSITIONS
            if current_al_unit == 0:
                book_balance_adjustment_rate = 0  # No units to adjust
                new_al_book_balance = cash_outflow  # All outflow becomes new book balance
            else:
                # For both long and short positions, the adjustment rate is the same
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
            
        # Store the simulation result
        self.simulation_dataframe = df.copy()
        
        return df
    


    def calculate_grouped(
        self,
        group_periods: int = 12,
        first_group_periods: Optional[int] = None,
        rate_aggregation: str = "last"
    ) -> pd.DataFrame:
        """
        Calculate grouped simulation results by aggregating periods.
        
        Parameters
        ----------
        group_periods : int, default 12
            Number of periods to group together for regular groups
        first_group_periods : Optional[int], default None
            Number of periods for the first group. If None, uses group_periods
        rate_aggregation : Literal["first", "last", "mean"], default "last"
            How to aggregate rate-type columns ('price', 'rate', 'income_gain_tax_rate', 'capital_gain_tax_rate'):
            - "first": Use the first value in each group (period beginning)
            - "last": Use the last value in each group (period end)  
            - "mean": Use the mean value across the group (period average)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with grouped results containing the same columns as simulation_dataframe.
            Flow items are summed, beginning-of-period stock items use first values,
            end-of-period stock items use last values, and rate items are aggregated 
            according to rate_aggregation parameter.
            
        Raises
        ------
        ValueError
            If simulation_dataframe is None (simulate() must be called first)
            If group_periods <= 0
            If first_group_periods is not None and <= 0
            If rate_aggregation is not one of the valid options
        """
        if self.simulation_dataframe is None:
            raise ValueError("simulation_dataframe is None. Call simulate() first.")
        
        if group_periods <= 0:
            raise ValueError("group_periods must be positive")
            
        if first_group_periods is not None and first_group_periods <= 0:
            raise ValueError("first_group_periods must be positive if provided")
            
        if rate_aggregation not in ["first", "last", "mean"]:
            raise ValueError("rate_aggregation must be one of: 'first', 'last', 'mean'")
        
        # Use group_periods for first group if not specified
        if first_group_periods is None:
            first_group_periods = group_periods
            
        df = self.simulation_dataframe.copy()
        
        # Define column categories
        flow_columns = [
            'income_cash_inflow_before_tax', 'income_gain_tax',
            'income_cash_inflow', 'unit_outflow', 'capital_cash_inflow_before_tax',
            'capital_gain_tax', 'capital_cash_inflow', 'cash_inflow', 'unit_inflow',
            'cash_outflow', 'cash_flow', 'unit_flow'
        ]
        
        beginning_stock_columns = [
            'pre_cash_balance', 'pre_al_unit', 'pre_al_balance',
            'pre_al_book_balance', 'pre_unrealized_gl'
        ]
        
        ending_stock_columns = [
            'cash_balance', 'al_unit', 'al_balance', 'al_book_balance', 'unrealized_gl'
        ]
        
        rate_columns = [
            'price', 'rate', 'cash_inflow_per_unit', 'income_gain_tax_rate', 'capital_gain_tax_rate'
        ]
        
        # Create group labels
        groups = []
        current_period = 0
        group_id = 0
        
        while current_period < len(df):
            if group_id == 0:
                # First group
                periods_in_group = min(first_group_periods, len(df) - current_period)
            else:
                # Regular groups
                periods_in_group = min(group_periods, len(df) - current_period)
            
            # Assign group_id to periods in this group
            for i in range(periods_in_group):
                groups.append(group_id)
            
            current_period += periods_in_group
            group_id += 1
        
        # Add group column
        df['group'] = groups
        
        # Prepare aggregation dictionary
        agg_dict = {}
        
        # Flow columns: sum
        for col in flow_columns:
            if col in df.columns:
                agg_dict[col] = 'sum'
        
        # Beginning stock columns: first
        for col in beginning_stock_columns:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        # Ending stock columns: last
        for col in ending_stock_columns:
            if col in df.columns:
                agg_dict[col] = 'last'
        
        # Rate columns: based on rate_aggregation parameter
        for col in rate_columns:
            if col in df.columns:
                agg_dict[col] = rate_aggregation
        
        # Group by and aggregate
        grouped_df = df.groupby('group').agg(agg_dict).reset_index(drop=True)
        
        # Maintain the original column order
        original_columns = [
            'price', 'pre_cash_balance', 'pre_al_unit', 'pre_al_balance',
            'pre_al_book_balance', 'pre_unrealized_gl', 'cash_inflow_per_unit',
            'income_cash_inflow_before_tax', 'income_gain_tax_rate',
            'income_gain_tax', 'income_cash_inflow', 'unit_outflow',
            'capital_cash_inflow_before_tax', 'capital_gain_tax_rate',
            'capital_gain_tax', 'capital_cash_inflow', 'cash_inflow', 'unit_inflow',
            'cash_outflow', 'cash_flow', 'unit_flow', 'cash_balance', 'al_unit',
            'al_balance', 'al_book_balance', 'unrealized_gl', 'rate'
        ]
        
        # Reorder columns to match original order
        available_columns = [col for col in original_columns if col in grouped_df.columns]
        grouped_df = grouped_df[available_columns]
        
        # Set index name
        grouped_df.index.name = 'time_period'
        
        return grouped_df


class LifecycleInvestmentSimulator:
    """
    Investment Lifecycle Simulator
    
    Executes investment simulations across three phases: accumulation, hold, and decumulation.
    Uses AssetLiabilitySimulator three times internally to model each phase.
    
    Supports two specification methods:
    1. Period-based: accumulation_periods, hold_periods, decumulation_periods
    2. Time-point-based: accumulation_end_period, decumulation_start_period, simulation_end_period
    
    Lifecycle phases:
    1. Accumulation: Regular contributions each period
    2. Hold: Investment growth only (no contributions or withdrawals)
    3. Decumulation: Complete asset liquidation over specified periods
    
    Note: hold_periods=0 is supported and will skip the hold phase entirely.
    """
    
    def __init__(
        self,
        contribution_amount: float,
        
        # Return rate specifications
        rate: Optional[Union[float, List[float]]] = None,
        rate_during_accumulation: Optional[Union[float, List[float]]] = None,
        rate_during_hold: Optional[Union[float, List[float]]] = None,
        rate_during_decumulation: Optional[Union[float, List[float]]] = None,
        
        # Method 1: Period-based specification
        accumulation_periods: Optional[int] = None,
        hold_periods: Optional[int] = None,
        decumulation_periods: Optional[int] = None,
        
        # Method 2: Time-point-based specification
        accumulation_end_period: Optional[int] = None,
        decumulation_start_period: Optional[int] = None,
        simulation_end_period: Optional[int] = None,
        
        # Initial settings
        initial_cash_balance: float = 0.0,
        initial_price: float = 1.0,
        cash_inflow_per_unit: Union[float, List[float]] = 0.0,
        
        # Tax rate settings
        income_gain_tax_rate: Union[float, List[float]] = TAX_RATE,
        capital_gain_tax_rate: Union[float, List[float]] = TAX_RATE,
    ):
        """
        Initialize Investment Lifecycle Simulator
        
        Parameters
        ----------
        contribution_amount : float
            Amount contributed each period during accumulation
        rate : Optional[Union[float, List[float]]], default None
            Default investment return rate per period
        rate_during_accumulation : Optional[Union[float, List[float]]], default None
            Return rate during accumulation phase
        rate_during_hold : Optional[Union[float, List[float]]], default None
            Return rate during hold phase
        rate_during_decumulation : Optional[Union[float, List[float]]], default None
            Return rate during decumulation phase
        accumulation_periods : Optional[int], default None
            Number of periods for accumulation phase
        hold_periods : Optional[int], default None
            Number of periods for hold phase (can be 0)
        decumulation_periods : Optional[int], default None
            Number of periods for decumulation phase
        accumulation_end_period : Optional[int], default None
            Period when accumulation phase ends (accumulation occurs in periods 0 to accumulation_end_period-1).
            The accumulation phase includes periods: 0, 1, 2, ..., accumulation_end_period-1
        decumulation_start_period : Optional[int], default None
            Period when decumulation phase starts (decumulation begins in period decumulation_start_period).
            The decumulation phase includes periods: decumulation_start_period, decumulation_start_period+1, ..., simulation_end_period-1
        simulation_end_period : Optional[int], default None
            Final period when simulation completes (simulation runs until period simulation_end_period-1).
            The simulation includes periods: 0, 1, 2, ..., simulation_end_period-1
            
        Note for time-point specification:
        - Hold phase includes periods: accumulation_end_period, accumulation_end_period+1, ..., decumulation_start_period-1
        - If accumulation_end_period = decumulation_start_period, then hold_periods = 0 (no hold phase)
        - If accumulation_end_period < decumulation_start_period, then hold_periods = decumulation_start_period - accumulation_end_period
        initial_cash_balance : float, default 0.0
            Initial cash balance
        initial_price : float, default 1.0
            Initial asset price
        cash_inflow_per_unit : Union[float, List[float]], default 0.0
            Cash inflow per unit (dividends, etc.)
        income_gain_tax_rate : Union[float, List[float]], default 0.0
            Tax rate for income gains
        capital_gain_tax_rate : Union[float, List[float]], default 0.0
            Tax rate for capital gains
        """
        self.contribution_amount = contribution_amount
        self.initial_cash_balance = initial_cash_balance
        self.initial_price = initial_price
        self.cash_inflow_per_unit = cash_inflow_per_unit
        self.income_gain_tax_rate = income_gain_tax_rate
        self.capital_gain_tax_rate = capital_gain_tax_rate
        
        # Validate and set period specifications
        self._validate_and_set_periods(
            accumulation_periods, hold_periods, decumulation_periods,
            accumulation_end_period, decumulation_start_period, simulation_end_period
        )
        
        # Process and set rate specifications
        self._process_rates(
            rate, rate_during_accumulation, rate_during_hold, rate_during_decumulation
        )
        
        # Storage for simulation results
        self.accumulation_df: Optional[pd.DataFrame] = None
        self.hold_df: Optional[pd.DataFrame] = None
        self.decumulation_df: Optional[pd.DataFrame] = None
        self.combined_df: Optional[pd.DataFrame] = None
    
    def _validate_and_set_periods(
        self,
        accumulation_periods: Optional[int],
        hold_periods: Optional[int], 
        decumulation_periods: Optional[int],
        accumulation_end_period: Optional[int],
        decumulation_start_period: Optional[int],
        simulation_end_period: Optional[int]
    ):
        """
        Validate input parameters and calculate period lengths
        
        For time-point-based specification, the periods are interpreted as follows:
        - Accumulation phase: periods 0, 1, ..., accumulation_end_period-1
        - Hold phase: periods accumulation_end_period, ..., decumulation_start_period-1  
        - Decumulation phase: periods decumulation_start_period, ..., simulation_end_period-1
        
        Examples:
        - accumulation_end_period=10, decumulation_start_period=12, simulation_end_period=20
          → accumulation: periods 0-9, hold: periods 10-11, decumulation: periods 12-19
        - accumulation_end_period=10, decumulation_start_period=10, simulation_end_period=15  
          → accumulation: periods 0-9, hold: none (hold_periods=0), decumulation: periods 10-14
        
        Parameters
        ----------
        accumulation_periods : Optional[int]
            Number of accumulation periods (period-based specification)
        hold_periods : Optional[int]
            Number of hold periods (period-based specification, can be 0)
        decumulation_periods : Optional[int]
            Number of decumulation periods (period-based specification)
        accumulation_end_period : Optional[int]
            Period when accumulation ends (time-point-based specification)
        decumulation_start_period : Optional[int]
            Period when decumulation starts (time-point-based specification)
        simulation_end_period : Optional[int]
            Period when simulation ends (time-point-based specification)
        """
        
        # Check if period-based specification is provided
        period_based = all(p is not None for p in [accumulation_periods, hold_periods, decumulation_periods])
        
        # Check if time-point-based specification is provided
        timepoint_based = all(p is not None for p in [accumulation_end_period, decumulation_start_period, simulation_end_period])
        
        if period_based and timepoint_based:
            raise ValueError("Cannot specify both period-based and time-point-based parameters simultaneously")
        
        if not period_based and not timepoint_based:
            raise ValueError("Must specify either all period-based parameters or all time-point-based parameters")
        
        if period_based:
            # Use period-based specification - hold_periods can be 0
            if accumulation_periods <= 0 or decumulation_periods <= 0:
                raise ValueError("accumulation_periods and decumulation_periods must be positive")
            if hold_periods < 0:
                raise ValueError("hold_periods must be non-negative")
            
            self.accumulation_periods = accumulation_periods
            self.hold_periods = hold_periods
            self.decumulation_periods = decumulation_periods
            
        else:
            # Use time-point-based specification and calculate periods
            if not (0 <= accumulation_end_period <= decumulation_start_period < simulation_end_period):
                raise ValueError("Time points must be in order: 0 <= accumulation_end <= decumulation_start < simulation_end")
            
            self.accumulation_periods = accumulation_end_period
            self.hold_periods = decumulation_start_period - accumulation_end_period
            self.decumulation_periods = simulation_end_period - decumulation_start_period
            
            if self.accumulation_periods <= 0 or self.decumulation_periods <= 0:
                raise ValueError("Calculated accumulation_periods and decumulation_periods must be positive")
            if self.hold_periods < 0:
                raise ValueError("Calculated hold_periods must be non-negative")
    
    def _process_rates(
        self,
        rate: Union[float, List[float]],
        rate_during_accumulation: Optional[Union[float, List[float]]],
        rate_during_hold: Optional[Union[float, List[float]]],
        rate_during_decumulation: Optional[Union[float, List[float]]]
    ):
        """
        Process rate specifications and assign rates to each phase
        
        Parameters
        ----------
        rate : Union[float, List[float]]
            Default rate
        rate_during_accumulation : Optional[Union[float, List[float]]]
            Rate during accumulation phase
        rate_during_hold : Optional[Union[float, List[float]]]
            Rate during hold phase
        rate_during_decumulation : Optional[Union[float, List[float]]]
            Rate during decumulation phase
        """
        phase_rates = [rate_during_accumulation, rate_during_hold, rate_during_decumulation]
        
        # Case 1: All three phase rates are specified
        if all(r is not None for r in phase_rates):
            self.accumulation_rate = rate_during_accumulation
            self.hold_rate = rate_during_hold
            self.decumulation_rate = rate_during_decumulation
            return
        
        # Case 2: Some phase rates are None and rate is float
        if isinstance(rate, (int, float)) and rate is not None:
            self.accumulation_rate = rate_during_accumulation if rate_during_accumulation is not None else rate
            self.hold_rate = rate_during_hold if rate_during_hold is not None else rate
            self.decumulation_rate = rate_during_decumulation if rate_during_decumulation is not None else rate
            return
        
        # Case 3: All phase rates are None, use rate
        if all(r is None for r in phase_rates):
            if rate is None:
                raise ValueError("At least one rate specification must be provided")
            if isinstance(rate, list):
                # Distribute list rates across phases
                self._distribute_list_rates(rate)
            else:
                self.accumulation_rate = rate
                self.hold_rate = rate
                self.decumulation_rate = rate
            return
        
        # Case 4: Mixed None and non-None with list rate
        if isinstance(rate, list):
            raise ValueError("Cannot mix None phase rates with list rate when some phase rates are specified")
        
        # Default fallback
        self.accumulation_rate = rate_during_accumulation if rate_during_accumulation is not None else rate
        self.hold_rate = rate_during_hold if rate_during_hold is not None else rate
        self.decumulation_rate = rate_during_decumulation if rate_during_decumulation is not None else rate
    
    def _distribute_list_rates(self, rate_list: List[float]):
        """
        Distribute list rates across phases sequentially
        
        Parameters
        ----------
        rate_list : List[float]
            List of rates to distribute
        """
        total_periods = self.accumulation_periods + self.hold_periods + self.decumulation_periods
        if len(rate_list) != total_periods:
            raise ValueError(f"Rate list length ({len(rate_list)}) must equal total periods ({total_periods})")
        
        # Distribute rates to each phase
        acc_end = self.accumulation_periods
        hold_end = acc_end + self.hold_periods
        
        self.accumulation_rate = rate_list[:acc_end]
        
        # Handle hold_periods=0 case
        if self.hold_periods == 0:
            self.hold_rate = []
        else:
            self.hold_rate = rate_list[acc_end:hold_end]
            
        self.decumulation_rate = rate_list[hold_end:]
    
    def simulate(self) -> pd.DataFrame:
        """
        Execute lifecycle simulation across all phases
        
        Returns
        -------
        pd.DataFrame
            Combined simulation results across all phases
        """
        # 1. Accumulation phase simulation
        self.accumulation_df = self._simulate_accumulation()
        
        # 2. Hold phase simulation (may return empty DataFrame if hold_periods=0)
        self.hold_df = self._simulate_hold()
        
        # 3. Decumulation phase simulation
        self.decumulation_df = self._simulate_decumulation()
        
        # 4. Combine results
        self.combined_df = self._combine_results()
        
        return self.combined_df
    
    def _simulate_accumulation(self) -> pd.DataFrame:
        """
        Execute accumulation phase simulation
        
        Returns
        -------
        pd.DataFrame
            Accumulation phase simulation results
        """
        al_schema = AssetLiabilitySchema(
            price=self.initial_price,
            unit=0,
            balance=0,
            book_balance=0,
            cashinflow_per_unit=self.cash_inflow_per_unit,
            rate=self.accumulation_rate,
        )
        
        # Ensure contribution_amount is applied to all accumulation periods
        cash_outflow_schedule = [self.contribution_amount] * self.accumulation_periods
        
        simulator = AssetLiabilitySimulator(
            al_schema=al_schema,
            initial_cash_balance=self.initial_cash_balance,
            capital_cash_inflow_before_tax=0,
            cash_outflow=cash_outflow_schedule,
            income_gain_tax_rate=self.income_gain_tax_rate,
            capital_gain_tax_rate=self.capital_gain_tax_rate,
        )
        
        return simulator.simulate(n_periods=self.accumulation_periods)
    
    def _simulate_hold(self) -> pd.DataFrame:
        """
        Execute hold phase simulation
        
        Returns
        -------
        pd.DataFrame
            Hold phase simulation results (empty DataFrame if hold_periods=0)
        """
        if self.accumulation_df is None:
            raise ValueError("Accumulation phase simulation must be completed first")
        
        # If no hold periods, return empty DataFrame with correct columns
        if self.hold_periods == 0:
            columns = [
                'price', 'pre_cash_balance', 'pre_al_unit', 'pre_al_balance',
                'pre_al_book_balance', 'pre_unrealized_gl', 'cash_inflow_per_unit',
                'income_cash_inflow_before_tax', 'income_gain_tax_rate',
                'income_gain_tax', 'income_cash_inflow', 'unit_outflow',
                'capital_cash_inflow_before_tax', 'capital_gain_tax_rate',
                'capital_gain_tax', 'capital_cash_inflow', 'cash_inflow', 'unit_inflow',
                'cash_outflow', 'cash_flow', 'unit_flow', 'cash_balance', 'al_unit',
                'al_balance', 'al_book_balance', 'unrealized_gl', 'rate'
            ]
            empty_df = pd.DataFrame(columns=columns)
            empty_df.index.name = 'time_period'
            return empty_df
        
        # Get final state from accumulation phase
        final_accumulation = self.accumulation_df.iloc[-1]
        
        # Calculate initial rate for hold phase
        hold_rate = self._get_initial_rate(self.hold_rate)
        
        al_schema = AssetLiabilitySchema(
            price=final_accumulation['price'] * (1 + hold_rate),
            unit=final_accumulation['al_unit'],
            balance=final_accumulation['al_balance'] * (1 + hold_rate),
            book_balance=final_accumulation['al_book_balance'],
            cashinflow_per_unit=self.cash_inflow_per_unit,
            rate=self.hold_rate,
        )
        
        simulator = AssetLiabilitySimulator(
            al_schema=al_schema,
            initial_cash_balance=final_accumulation['cash_balance'],
            capital_cash_inflow_before_tax=0,
            cash_outflow=0,  # No contributions during hold phase
            income_gain_tax_rate=self.income_gain_tax_rate,
            capital_gain_tax_rate=self.capital_gain_tax_rate,
        )
        
        return simulator.simulate(n_periods=self.hold_periods)
    
    def _simulate_decumulation(self) -> pd.DataFrame:
        """
        Execute decumulation phase simulation
        
        Returns
        -------
        pd.DataFrame
            Decumulation phase simulation results
        """
        if self.hold_df is None:
            raise ValueError("Hold phase simulation must be completed first")
        
        # Determine the final state from the previous phase
        if len(self.hold_df) == 0:
            if self.accumulation_df is None or len(self.accumulation_df) == 0:
                raise ValueError("No valid data from previous phases")
            final_state = self.accumulation_df.iloc[-1]
        else:
            final_state = self.hold_df.iloc[-1]
        
        # Calculate decumulation parameters
        decumulation_rate = self._get_initial_rate(self.decumulation_rate)
        
        # Calculate the correct asset value for decumulation start
        # This should be the hold phase end balance grown by one period
        total_asset_value = final_state['al_balance'] * (1 + decumulation_rate)
        
        # Calculate withdrawal amount using InterestFactor
        interest_factor = InterestFactor(
            rate=decumulation_rate,
            time_period=self.decumulation_periods,
            amount=total_asset_value
        )
        withdrawal_amount = interest_factor.calculate_capital_recovery()
        
        # Create the correct capital inflow schedule:
        # [0, withdrawal_amount, withdrawal_amount, ...]
        # The first period (0) is for initial asset investment
        capital_inflow_schedule = [0] + [withdrawal_amount] * self.decumulation_periods
        
        # Create the correct cash outflow schedule:
        # [total_asset_value, 0, 0, ...]
        # The first period invests all assets, subsequent periods have no outflow
        cash_outflow_schedule = [total_asset_value] + [0] * self.decumulation_periods
        
        # Start with empty portfolio
        al_schema = AssetLiabilitySchema(
            price=final_state['price'] * (1 + decumulation_rate),
            unit=0,  # Start with no units
            balance=0,  # Start with no balance
            book_balance=0,  # Start with no book balance
            cashinflow_per_unit=self.cash_inflow_per_unit,
            rate=self.decumulation_rate,
        )
        
        simulator = AssetLiabilitySimulator(
            al_schema=al_schema,
            initial_cash_balance=final_state['cash_balance'],
            capital_cash_inflow_before_tax=capital_inflow_schedule,
            cash_outflow=cash_outflow_schedule,
            income_gain_tax_rate=self.income_gain_tax_rate,
            capital_gain_tax_rate=self.capital_gain_tax_rate,
        )
        
        # Simulate for decumulation_periods + 1 to handle initial investment
        result = simulator.simulate(n_periods=self.decumulation_periods + 1)
        
        # Return only the decumulation periods (skip the initial investment period)
        return result.iloc[1:].reset_index(drop=True)
    
    def _get_initial_rate(self, rate: Union[float, List[float]]) -> float:
        """
        Get initial rate for phase transition calculations
        
        Parameters
        ----------
        rate : Union[float, List[float]]
            Rate specification for the phase
            
        Returns
        -------
        float
            Initial rate value
        """
        if isinstance(rate, list):
            return rate[0] if rate else 0.0
        else:
            return rate
    
    def _combine_results(self) -> pd.DataFrame:
        """
        Combine results from all three phases
        
        Returns
        -------
        pd.DataFrame
            Combined simulation results
        """
        if any(df is None for df in [self.accumulation_df, self.hold_df, self.decumulation_df]):
            raise ValueError("All phase simulations must be completed first")
        
        # Prepare phase DataFrames
        phases_to_combine = []
        current_index_offset = 0
        
        # 1. Accumulation phase
        accumulation_adjusted = self.accumulation_df.copy()
        phases_to_combine.append(accumulation_adjusted)
        current_index_offset += len(accumulation_adjusted)
        
        # 2. Hold phase (only add if not empty)
        if len(self.hold_df) > 0:
            hold_adjusted = self.hold_df.copy()
            hold_adjusted.index = hold_adjusted.index + current_index_offset
            phases_to_combine.append(hold_adjusted)
            current_index_offset += len(hold_adjusted)
        
        # 3. Decumulation phase
        decumulation_adjusted = self.decumulation_df.copy()
        decumulation_adjusted.index = decumulation_adjusted.index + current_index_offset
        phases_to_combine.append(decumulation_adjusted)
        
        # Combine all phases
        combined = pd.concat(phases_to_combine).reset_index(drop=True)
        combined.index.name = 'time_period'
        
        return combined