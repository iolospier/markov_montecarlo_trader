import pandas as pd
import numpy as np
from src.utils.logger import get_logger


def classify_regimes(df, k: float = 1.0, logger=None) -> pd.DataFrame:
    """Classify market regimes based on returns vs volatility threshold."""
    logger = logger or get_logger(".")
    if "Return" not in df.columns or "Volatility" not in df.columns:
        logger.error("Missing 'Return' or 'Volatility' columns in input data.")
        raise KeyError("DataFrame must contain 'Return' and 'Volatility'.")

    df = df.copy()
    conds = [
        df["Return"] > k * df["Volatility"],
        df["Return"] < -k * df["Volatility"],
    ]
    choices = ["Bull", "Bear"]
    df["State"] = np.select(conds, choices, default="Neutral")

    logger.info(f"Classified regimes using k={k}.")
    return df


def estimate_transition_matrix(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """Estimate Markov transition matrix empirically from classified states."""
    logger = logger or get_logger(".")
    states = ["Bear", "Neutral", "Bull"]
    matrix = pd.DataFrame(0, index=states, columns=states, dtype=float)

    if "State" not in df.columns:
        logger.error("Missing 'State' column in DataFrame for transition estimation.")
        raise KeyError("Missing 'State' column in DataFrame.")

    for current, nxt in zip(df["State"], df["State"].shift(-1)):
        if pd.notna(current) and pd.notna(nxt):
            matrix.loc[current, nxt] += 1

    matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    logger.info("Estimated transition matrix between regimes.")
    return matrix


def simulate_states(
    P: pd.DataFrame,
    start_state: str,
    n_steps: int = 252,
    seed: int | None = None,
    logger=None,
) -> list[str]:
    """Generate a synthetic sequence of regimes using the transition matrix P."""
    logger = logger or get_logger(".")
    rng = np.random.default_rng(seed)

    states = list(P.index)
    if start_state not in states:
        logger.error(f"Invalid start state '{start_state}' — must be one of {states}")
        raise ValueError(f"Invalid start state: {start_state}")

    sequence = [start_state]
    current = start_state
    for _ in range(n_steps - 1):
        probs = P.loc[current].values
        current = rng.choice(states, p=probs)
        sequence.append(current)

    logger.debug(f"Simulated {n_steps} regime states from {start_state}.")
    return sequence


def estimate_regime_params(df, state_col="State", ret_col="Return", logger=None):
    """Estimate mean drift (mu) and volatility (sigma) for each regime."""
    logger = logger or get_logger(".")
    if state_col not in df.columns or ret_col not in df.columns:
        logger.error(f"Missing required columns '{state_col}' or '{ret_col}'.")
        raise KeyError("Missing required columns for regime parameter estimation.")

    regime_params = {}
    for state, group in df.groupby(state_col):
        mu = group[ret_col].mean()
        sigma = group[ret_col].std()
        regime_params[state] = {"mu": mu, "sigma": sigma}
        logger.debug(f"{state} regime: μ={mu:.5f}, σ={sigma:.5f}")

    logger.info("Estimated regime parameters for all states.")
    return regime_params
