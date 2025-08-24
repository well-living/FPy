# fpyjp/core/loan_simulator.py
"""
Loan Simulator

Implements loan simulation using the AssetLiabilitySimulator for various loan types
including mortgage loans and education loans with different repayment methods.
"""

from typing import List, Union, Optional
import datetime

import pandas as pd

from fpyjp.schemas.loan import Loan, RepaymentMethod, InterestRateType
from fpyjp.schemas.balance import AssetLiabilitySchema
from fpyjp.core.balance_simulator import AssetLiabilitySimulator
from fpyjp.core.interest_factor import InterestFactor


class LoanSimulator:
    """
    Loan Simulator
    
    Simulates loan repayment schedules and cash flows over time periods based on
    loan parameters using the underlying AssetLiabilitySimulator.
    
    Supports different loan types such as:
    - Mortgage loans (住宅ローン) like Flat35
    - Education loans (教育ローン) from JASSO
    - Other consumer loans
    
    Supports different repayment methods:
    - Equal principal repayment (元金均等返済)
    - Equal payment repayment (元利均等返済)
    
    Attributes
    ----------
    loan : Loan
        Loan object containing loan information
    initial_cash_balance : float
        Initial cash balance
    """
    
    def __init__(
        self,
        loan: Optional[Loan] = None,
        initial_cash_balance: float = 0.0,
        # Loan fields as individual parameters
        name: Optional[str] = None,
        interest_rate: Optional[Union[float, List[float]]] = None,
        contract_date: Optional[datetime.date] = None,
        principal: Optional[float] = None,
        remaining_balance: Optional[float] = None,
        total_term_months: Optional[int] = None,
        remaining_term_months: Optional[int] = None,
        repayment_method: Union[RepaymentMethod, str] = RepaymentMethod.EQUAL_PRINCIPAL,
        interest_rate_type: Union[InterestRateType, str] = InterestRateType.FIXED,
        payment_frequency: int = 1,
        repayment_amount: Optional[float] = None,
    ):
        """
        Initialize the loan simulator.
        
        Parameters
        ----------
        loan : Optional[Loan], default None
            Loan object containing all loan parameters. If provided, individual parameters are ignored.
        initial_cash_balance : float, default 0.0
            Initial cash balance
        name : Optional[str], default None
            Name of the loan product (used if loan is not provided)
        interest_rate : Optional[Union[float, List[float]]], default None
            Interest rate as decimal (used if loan is not provided)
        contract_date : Optional[datetime.date], default None
            Date when the loan contract was signed (used if loan is not provided)
        principal : Optional[float], default None
            Original loan amount (used if loan is not provided)
        remaining_balance : Optional[float], default None
            Current remaining balance of the loan (used if loan is not provided)
        total_term_months : Optional[int], default None
            Total loan term in months (used if loan is not provided)
        remaining_term_months : Optional[int], default None
            Remaining loan term in months (used if loan is not provided)
        repayment_method : Union[RepaymentMethod, str], default RepaymentMethod.EQUAL_PRINCIPAL
            Method of loan repayment (used if loan is not provided)
        interest_rate_type : Union[InterestRateType, str], default InterestRateType.FIXED
            Type of interest rate (used if loan is not provided)
        payment_frequency : int, default 1
            Payment frequency in months (used if loan is not provided)
        repayment_amount : Optional[float], default None
            Regular payment amount per payment period (used if loan is not provided)
        """
        if loan is not None:
            self.loan = loan
        else:
            # Create loan from individual parameters
            if name is None or interest_rate is None or remaining_balance is None or remaining_term_months is None:
                raise ValueError("Either loan object or required individual parameters (name, interest_rate, remaining_balance, remaining_term_months) must be provided")
            
            self.loan = Loan(
                name=name,
                interest_rate=interest_rate,
                contract_date=contract_date,
                principal=principal,
                remaining_balance=remaining_balance,
                total_term_months=total_term_months,
                remaining_term_months=remaining_term_months,
                repayment_method=repayment_method,
                interest_rate_type=interest_rate_type,
                payment_frequency=payment_frequency,
                repayment_amount=repayment_amount,
            )
        
        self.initial_cash_balance = initial_cash_balance
        
        # Validate loan object
        if loan.remaining_balance is None or loan.remaining_balance <= 0:
            raise ValueError("Loan must have a positive remaining balance")
        if loan.remaining_term_months is None or loan.remaining_term_months <= 0:
            raise ValueError("Loan must have a positive remaining term")
    
    def _create_asset_liability_schema(self) -> AssetLiabilitySchema:
        """
        Create AssetLiabilitySchema from loan parameters.
        
        Returns
        -------
        AssetLiabilitySchema
            Schema configured for loan simulation
        """
        # For loan simulation, we treat the loan as a negative asset (liability)
        # Unit represents the loan amount in currency units
        # Price is always 1.0 for loan simulations
        
        if self.loan.repayment_method == RepaymentMethod.EQUAL_PRINCIPAL:
            # For equal principal repayment, interest is paid as cash inflow per unit
            monthly_rate = self.loan.get_current_interest_rate() / 12
            cash_inflow_per_unit = monthly_rate
            rate = 0.0  # No price growth for equal principal
        else:
            # For equal payment repayment, compound interest growth
            monthly_rate = self.loan.get_current_interest_rate() / 12
            cash_inflow_per_unit = 0.0
            rate = monthly_rate
        
        return AssetLiabilitySchema(
            name=self.loan.name,
            price=1.0,
            unit=-self.loan.remaining_balance,  # Negative for liability
            balance=-self.loan.remaining_balance,
            book_balance=-self.loan.remaining_balance,
            cashinflow_per_unit=cash_inflow_per_unit,
            rate=rate,
            allow_negative_unit=True
        )
    
    def _calculate_cash_outflows(self) -> List[float]:
        """
        Calculate cash outflows (loan repayments) based on repayment method.
        
        Returns
        -------
        List[float]
            List of cash outflows for each period
        """
        monthly_rate = self.loan.get_current_interest_rate() / 12
        
        if self.loan.repayment_method == RepaymentMethod.EQUAL_PRINCIPAL:
            # Equal principal repayment
            monthly_principal = self.loan.remaining_balance / self.loan.remaining_term_months
            return [monthly_principal] * self.loan.remaining_term_months
            
        elif self.loan.repayment_method == RepaymentMethod.EQUAL_PAYMENT:
            # Equal payment repayment using capital recovery factor
            if self.loan.repayment_amount is not None:
                # Use provided repayment amount
                monthly_payment = self.loan.repayment_amount
            else:
                # Calculate using InterestFactor
                interest_factor = InterestFactor(
                    rate=monthly_rate,
                    time_period=self.loan.remaining_term_months,
                    amount=self.loan.remaining_balance
                )
                monthly_payment = interest_factor.calculate_capital_recovery()
            
            # For equal payment, first period is 0, then constant payments
            return [0.0] + [monthly_payment] * self.loan.remaining_term_months
        
        else:
            raise ValueError(f"Unsupported repayment method: {self.loan.repayment_method}")
    
    def simulate(self, n_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Run the loan simulation for the specified number of periods.
        
        Parameters
        ----------
        n_periods : Optional[int], default None
            Number of periods to simulate. If None, uses remaining_term_months
            (plus 1 for equal payment method to show initial state)
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing simulation results from AssetLiabilitySimulator.
            
            The loan is represented as a negative asset (liability) in the simulation:
            - al_balance: Negative value representing outstanding loan balance
            - income_cash_inflow_before_tax: Negative value representing interest charges
            - cash_outflow: Positive value representing loan payments
        
        Notes
        -----
        The simulation interprets AssetLiabilitySimulator results in loan context:
        - Negative al_unit represents the outstanding loan amount
        - cash_inflow_per_unit represents interest rate (for equal principal)
        - rate represents compound interest growth (for equal payment)
        - cash_outflow represents loan payments
        - income_cash_inflow_before_tax represents interest charges
        """
        if n_periods is None:
            if self.loan.repayment_method == RepaymentMethod.EQUAL_PAYMENT:
                n_periods = self.loan.remaining_term_months + 1  # Include initial state
            else:
                n_periods = self.loan.remaining_term_months
        
        # Create AssetLiabilitySchema
        al_schema = self._create_asset_liability_schema()
        
        # Calculate cash outflows
        cash_outflows = self._calculate_cash_outflows()
        
        # Ensure cash_outflows has correct length
        if len(cash_outflows) < n_periods:
            # Pad with zeros if needed
            cash_outflows.extend([0.0] * (n_periods - len(cash_outflows)))
        elif len(cash_outflows) > n_periods:
            # Truncate if needed
            cash_outflows = cash_outflows[:n_periods]
        
        # Run simulation using AssetLiabilitySimulator
        simulator = AssetLiabilitySimulator(
            al_schema=al_schema,
            initial_cash_balance=self.initial_cash_balance,
            capital_cash_inflow_before_tax=0,
            cash_outflow=cash_outflows,
            income_gain_tax_rate=0,
            capital_gain_tax_rate=0,
        )
        
        df = simulator.simulate(n_periods=n_periods)
        
        return df