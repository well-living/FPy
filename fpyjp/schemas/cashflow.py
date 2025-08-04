
from typing import List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class CashFlowSchema(BaseModel):

    amount: Union[float, List[float]] = Field(
        default=0.0,
    )
    sign: int = Field(
        default=1,
        description="1 for inflow, -1 for outflow",
    )
    start: Optional[int] = Field(
        default=None,
        description="Start period for cash flow",
    )
    end: Optional[int] = Field(
        default=None,
        description="End period for cash flow",
    )


