import glob
import re

import pandas as pd


def get_data():
    def natural_key(s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    # Load and append all matching csv files
    files = sorted(glob.glob("checkpoints/default.mbpo_sac/*.csv"), key=natural_key)
    print(files)

    step_offset = 0

    # Load CSV
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        if len(df) < 1:
            continue

        # steps are continuous, but zero indexed per file
        df = df.sort_values("step").reset_index(drop=True)
        df["step"] = df["step"] + step_offset
        step_offset = df["step"].iloc[-1] + 1
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df


def smooth(df, window=100):
    # smooth with rolling window
    loss_cols = [
        "reward",
        "world_nrmse",
        "world_sign_accuracy",
        "world_loss",
        "reward_loss",
        "q1_loss",
        "q2_loss",
        "policy_loss",
        "alpha_loss",
    ]
    df[loss_cols] = (
        df[loss_cols].rolling(window=window, min_periods=1, center=False).mean()
    )

    mean_df = df.copy()
    std_df = df.copy()
    mean_df[loss_cols] = df[loss_cols].rolling(window, min_periods=1).mean()
    std_df[loss_cols] = df[loss_cols].rolling(window, min_periods=1).std()

    return df, mean_df, std_df


def unify(df):
    # unify sizes into the range [-1,1]
    dfmax = df.max()
    dfmin = df.min()
    dfrange = dfmax - dfmin

    # save and restore the steps, as they should not be unified
    df_step_backup = df["step"]
    df = (df - dfmin) / dfrange
    df["step"] = df_step_backup
    return df
