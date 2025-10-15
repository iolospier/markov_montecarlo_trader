import pandas as pd
import numpy as np


def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each day as Bull, Bear, or Neutral based on return vs volatility.
    """
    if "Return" not in df.columns or "Volatility" not in df.columns:
        raise ValueError("DataFrame must include 'Return' and 'Volatility' columns.")

    df = df.copy()
    conditions = [
        df["Return"] > df["Volatility"],
        df["Return"] < -df["Volatility"],
    ]
    choices = ["Bull", "Bear"]
    df["State"] = np.select(conditions, choices, default="Neutral")
    return df


def estimate_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the Markov transition matrix empirically.
    """
    states = ["Bear", "Neutral", "Bull"]
    matrix = pd.DataFrame(0, index=states, columns=states, dtype=float)

    for current, nxt in zip(df["State"], df["State"].shift(-1)):
        if pd.notna(current) and pd.notna(nxt):
            matrix.loc[current, nxt] += 1

    # Normalise rows to probabilities
    matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    return matrix


def simulate_states(P: pd.DataFrame, start_state: str, n_steps: int = 252) -> list[str]:
    """
    Generate a synthetic sequence of regimes using the transition matrix P.
    """
    states = list(P.index)
    current = start_state
    sequence = [current]
    for _ in range(n_steps - 1):
        probs = P.loc[current].values
        current = np.random.choice(states, p=probs)
        sequence.append(current)
    return sequence


def estimate_regime_params(df, state_col="State", ret_col="Return"):
    """
    Estimate mean drift (mu) and volatility (sigma) for each regime.
    Uses daily returns from the classified dataset.
    """
    regime_params = {}
    for state, group in df.groupby(state_col):
        mu = group[ret_col].mean()
        sigma = group[ret_col].std()
        regime_params[state] = {"mu": mu, "sigma": sigma}
    return regime_params
