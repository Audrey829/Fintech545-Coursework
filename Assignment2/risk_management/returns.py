"""Functions for Functional Tests 6.1 and 6.2.

Test/function map
-----------------
- Test 6.1: ``calculate_returns(..., method="DISCRETE")``
- Test 6.2: ``calculate_returns(..., method="LOG")``
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_returns(
    prices: pd.DataFrame,
    *,
    method: str = "DISCRETE",
    date_column: str = "Date",
    demean: bool = False,
) -> pd.DataFrame:
    """Calculate arithmetic or log returns while retaining dates from row two.

    Set ``demean=True`` to center each return column after calculation.
    """

    if date_column not in prices.columns:
        raise ValueError(f"date_column {date_column!r} is not in the DataFrame")
    asset_columns = [column for column in prices.columns if column != date_column]
    if not asset_columns:
        raise ValueError("prices must contain at least one asset column")

    numeric = prices[asset_columns].astype(float)
    ratios = numeric.iloc[1:].to_numpy() / numeric.iloc[:-1].to_numpy()
    normalized_method = method.upper()
    if normalized_method == "DISCRETE":
        values = ratios - 1.0
    elif normalized_method == "LOG":
        values = np.log(ratios)
    else:
        raise ValueError('method must be either "DISCRETE" or "LOG"')

    if demean:
        values = values - values.mean(axis=0)

    result = pd.DataFrame(values, columns=asset_columns)
    result.insert(0, date_column, prices[date_column].iloc[1:].to_numpy())
    return result
