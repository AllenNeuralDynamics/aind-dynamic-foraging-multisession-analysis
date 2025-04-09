import os
from datetime import datetime, timedelta

import aind_dynamic_foraging_basic_analysis.licks.annotation as a
import aind_dynamic_foraging_basic_analysis.metrics.trial_metrics as tm
import numpy as np
import pandas as pd
from aind_dynamic_foraging_data_utils import nwb_utils as nu

"""

import aind_dynamic_foraging_multisession_analysis.load as load
NWB_FILES = glob.glob(DATA_DIR + 'behavior_<mouse_id>_**.nwb')
nwbs, df = load.make_multisession_trials_df(NWB_FILES)

"""


# TODO, move this?
# TODO, compute bias?
def add_side_bias(nwb):
    if "side_bias" in nwb.df_trials:
        return nwb.df_trials
    nwb.df_trials["side_bias"] = [np.nan] * len(nwb.df_trials)
    nwb.df_trials["side_bias_confidence_interval"] = [[np.nan, np.nan]] * len(
        nwb.df_trials
    )
    return nwb.df_trials


def make_multisession_trials_df(nwb_list):
    """
    takes a list of NWBs
    loads each NWB file
    makes trials table
    adds metrics
    makes aggregate trials table
    """
    nwbs = []
    crash_list = []
    for n in nwb_list:
        try:
            nwb = nu.load_nwb_from_filename(n)
            nwb.df_trials = nu.create_df_trials(nwb, verbose=False)
            nwb.df_events = nu.create_events_df(nwb, verbose=False)
            nwb.df_licks = a.annotate_licks(nwb)
            nwb.df_trials = tm.compute_trial_metrics(nwb)
            nwb.df_trials = tm.add_intertrial_licking(nwb)
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
    df = pd.concat([x.df_trials for x in nwbs])

    return nwbs, df
