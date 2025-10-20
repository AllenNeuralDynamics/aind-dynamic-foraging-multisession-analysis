"""
Tools for aggregation of multiple sessions into a dataframe

import aind_dynamic_foraging_multisession_analysis.load as load
NWB_FILES = glob.glob(DATA_DIR + 'behavior_<mouse_id>_**.nwb')
nwbs, df = load.make_multisession_trials_df(NWB_FILES)

"""

import numpy as np
import pandas as pd
from aind_dynamic_foraging_data_utils import nwb_utils as nu
import aind_dynamic_foraging_basic_analysis.licks.annotation as a
import aind_dynamic_foraging_basic_analysis.metrics.trial_metrics as tm


def make_multisession_trials_df(nwb_list, allow_duplicates=True):
    """
    Builds a dataframe of trials concatenated across multiple sessions
    nwb_list, a list of NWBs to concatenate. Can either be paths to the files
            or NWB files themselves

    The multisession dataframe will contain the union of the columns in the
    individual nwb.df_trials. The rows will be sorted by the session date,
    and then trial number within each session
    """
    unique_sessions = set()
    nwbs = []
    crash_list = []
    for n in nwb_list:
        print(n)
        try:
            nwb = nu.load_nwb_from_filename(n)
            if (not allow_duplicates) and (nwb.session_id in unique_sessions):
                continue
            else:
                unique_sessions.add(nwb.session_id)
            nwb.df_trials = nu.create_df_trials(nwb, verbose=False)
            nwb.df_events = nu.create_df_events(nwb, verbose=False)
            nwb.df_licks = a.annotate_licks(nwb)
            nwb.df_trials = tm.compute_trial_metrics(nwb)
            nwb.df_trials = add_side_bias(nwb)
            nwbs.append(nwb)
        except Exception as e:
            crash_list.append(n)
            print("Bad {}".format(n))
            print("   " + str(e))

    # Log summary of sessions with loading errors
    if len(crash_list) > 0:
        print("\n\nThe following sessions could not be loaded")
        print("\n".join(crash_list))

    # Make a dataframe of trials
    for nwb in nwbs:
        nwb.df_trials["ses_idx"] = [nwb.session_id[9:]] * len(nwb.df_trials)
    df = pd.concat([x.df_trials for x in nwbs])

    return nwbs, df


def add_side_bias(nwb, compute=True):
    """
    Adds a column "side_bias" and "side_bias_confidence_interval"
    to the nwb.df_trials, if it does not already exist

    compute (bool) whether to compute the side bias or add NaNs
    """

    if not hasattr(nwb, "df_trials"):
        print("No df_trials")
        return

    if "side_bias" in nwb.df_trials:
        return nwb.df_trials.copy()

    # Whether to compute the bias or not
    df_trials = nwb.df_trials.copy()
    if compute:
        df_trials = tm.compute_side_bias(nwb)
    else:
        df_trials["side_bias"] = [np.nan] * len(df_trials)
        df_trials["side_bias_confidence_interval"] = [[np.nan, np.nan]] * len(
            df_trials
        )

    return df_trials
