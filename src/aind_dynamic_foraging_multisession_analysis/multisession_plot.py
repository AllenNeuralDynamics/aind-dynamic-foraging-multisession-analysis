"""
Plotting tools for multisession data

"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np


def plot_foraging_multisession(  # NOQA C901
    multisession_df,
    plot_list=["side_bias", "lickspout_position"],
    interactive=True,
):
    """
    Takes a dataframe of the aggregate for all sessions from this animal

    plot_list is a list of columns in the dataframe to plot
        can additionally pass a list which include a list of columns
        that will all be plotted together
        example: plot_list = ['side_bias',['response_rate','reward_rate']]
    interactive (bool), if true, make the plot interactive
    """

    # Ensure dataframe is sorted by session then trial
    df = multisession_df.copy()
    df = df.sort_values(by=["ses_idx", "trial"])
    df["multisession_trial"] = df.reset_index().index

    # Set up figure
    fig, ax = plt.subplots(
        len(plot_list) + 1,
        1,
        figsize=(14, 2 * (1 + len(plot_list))),
        sharex=True,
    )

    # Plot basic behavior
    plot_foraging_behavior(ax[0], df)

    # Plot each element
    for index, plot in enumerate(plot_list):
        if isinstance(plot, list):
            # We have a list to plot together
            for inner_plot in plot:
                plot_foraging_multisession_inner(ax[index + 1], inner_plot, df)
        else:
            plot_foraging_multisession_inner(ax[index + 1], plot, df)

    # Add session breaks to each axis
    session_breaks = list(
        df.query("trial == 0")["multisession_trial"].values - 0.5
    ) + [df["multisession_trial"].values[-1]]
    multiday_breaks = find_multiday_breaks(df)
    for a in ax:
        for index, x in enumerate(session_breaks):
            if multiday_breaks[index]:
                a.axvline(x, color="gray", alpha=1, linestyle="--")
            else:
                a.axvline(x, color="gray", alpha=0.25, linestyle="--")

    # Determine xtick positions and labels
    ticks = list(df.query("trial == 0")["multisession_trial"].values) + [
        df["multisession_trial"].values[-1]
    ]
    ticks = ticks[:-1] + np.diff(ticks) / 2
    labels = [
        "-".join(x.split("_")[1].split("-")[1:])
        for x in df.query("trial == 0")["ses_idx"].values
    ]

    # Add ticks to the bottom plot
    ax[-1].set_xticks(ticks, labels)
    ax[-1].set_xlabel("Session")
    ax[-1].set_xlim(
        df["multisession_trial"].values[0], df["multisession_trial"].values[-1]
    )
    if interactive:
        plt.suptitle(
            df["ses_idx"].values[0].split("_")[0]
            + " use arrow keys in scroll or zoom in/out, use h to reset"
        )
    else:
        plt.suptitle(df["ses_idx"].values[0].split("_")[0])
    plt.tight_layout()

    # Add interactive scrolling
    xhome = ax[0].get_xlim()

    def on_key_press(event):
        """
        Define interaction resonsivity
        """
        x = ax[0].get_xlim()
        xmin = x[0]
        xmax = x[1]
        xStep = (xmax - xmin) / 4
        if event.key == "<" or event.key == "," or event.key == "left":
            xmin -= xStep
            xmax -= xStep
        elif event.key == ">" or event.key == "." or event.key == "right":
            xmin += xStep
            xmax += xStep
        elif event.key == "up":
            xmin -= xStep
            xmax += xStep
        elif event.key == "down":
            xmin += xStep * (2 / 3)
            xmax -= xStep * (2 / 3)
        elif event.key == "h":
            xmin = xhome[0]
            xmax = xhome[1]
        ax[0].set_xlim(xmin, xmax)
        plt.draw()

    if interactive:
        fig.canvas.mpl_connect("key_press_event", on_key_press)  # noqa: F841


def plot_foraging_behavior(ax, df):
    """
    Plot basic behavior information including licks and rewards
    ax, the axis to plot on
    df, the multisession dataframe
    """

    # Grab data
    choice_history = np.array(
        [np.nan if x == 2 else x for x in df["animal_response"].values]
    )
    reward_history = df["earned_reward"].values
    autowater_offered = df[["auto_waterL", "auto_waterR"]].any(axis=1)
    manual_water = df["extra_reward"].values

    # Compute things
    ignored = np.isnan(choice_history)
    rewarded_excluding_autowater = reward_history & ~autowater_offered
    autowater_collected = autowater_offered & ~ignored
    autowater_ignored = autowater_offered & ignored
    unrewarded_trials = ~reward_history & ~ignored & ~autowater_offered
    manual_water_ignored = manual_water & ignored
    manual_water_collected = manual_water & ~ignored

    # Mark unrewarded trials
    xx = np.nonzero(unrewarded_trials)[0] + 1
    yy_temp = choice_history[unrewarded_trials]
    yy_right = yy_temp[yy_temp > 0.5]
    xx_right = xx[yy_temp > 0.5]
    yy_left = yy_temp[yy_temp < 0.5] + 1
    xx_left = xx[yy_temp < 0.5]
    ax.vlines(
        xx_right,
        yy_right + 0.05,
        yy_right + 0.1,
        alpha=1,
        linewidth=1,
        color="gray",
        label="Unrewarded choices",
    )
    ax.vlines(
        xx_left,
        yy_left - 0.1,
        yy_left - 0.05,
        alpha=1,
        linewidth=1,
        color="gray",
    )

    # Rewarded trials (real foraging, autowater excluded)
    xx = np.nonzero(rewarded_excluding_autowater)[0] + 1
    yy_temp = choice_history[rewarded_excluding_autowater]
    yy_right = yy_temp[yy_temp > 0.5] + 0.05
    xx_right = xx[yy_temp > 0.5]
    yy_left = yy_temp[yy_temp < 0.5] - 0.05 + 1
    xx_left = xx[yy_temp < 0.5]
    ax.vlines(
        xx_right,
        yy_right + 0.05,
        yy_right + 0.1,
        alpha=1,
        linewidth=1,
        color="black",
        label="Rewarded choices",
    )
    ax.vlines(
        xx_right,
        yy_right + 0,
        yy_right + 0.05,
        alpha=1,
        linewidth=1,
        color="gray",
    )
    ax.vlines(
        xx_left,
        yy_left - 0.1,
        yy_left - 0.05,
        alpha=1,
        linewidth=1,
        color="black",
    )
    ax.vlines(
        xx_left,
        yy_left - 0.05,
        yy_left,
        alpha=1,
        linewidth=1,
        color="gray",
    )

    # Ignored trials
    xx = np.nonzero(ignored & ~autowater_ignored)[0] + 1
    yy = [0.99] * sum(ignored & ~autowater_ignored)
    ax.plot(
        *(xx, yy),
        "x",
        color="red",
        markersize=3,
        markeredgewidth=0.5,
        label="Ignored",
    )

    # Manual water offered and collected
    xx = np.nonzero(manual_water_collected)[0] + 1
    yy_temp = choice_history[manual_water_collected]
    yy_right = yy_temp[yy_temp > 0.5] + 0.05
    xx_right = xx[yy_temp > 0.5]
    yy_left = yy_temp[yy_temp < 0.5] - 0.05 + 1
    xx_left = xx[yy_temp < 0.5]
    ax.vlines(
        xx_right,
        yy_right + 0.05,
        yy_right + 0.1,
        alpha=1,
        linewidth=1,
        color="cyan",
        label="Autowater collected",
    )
    ax.vlines(
        xx_left,
        yy_left - 0.1,
        yy_left - 0.05,
        alpha=1,
        linewidth=1,
        color="cyan",
    )
    # Also highlight the autowater offered but still ignored
    xx = np.nonzero(manual_water_ignored)[0] + 1
    yy = [1.01] * sum(manual_water_ignored)
    ax.plot(
        *(xx, yy),
        "x",
        color="cyan",
        markersize=3,
        markeredgewidth=0.5,
        label="Manual water ignored",
    )

    # Autowater offered and collected
    xx = np.nonzero(autowater_collected)[0] + 1
    yy_temp = choice_history[autowater_collected]
    yy_right = yy_temp[yy_temp > 0.5] + 0.05
    xx_right = xx[yy_temp > 0.5]
    yy_left = yy_temp[yy_temp < 0.5] - 0.05 + 1
    xx_left = xx[yy_temp < 0.5]
    ax.vlines(
        xx_right,
        yy_right + 0.05,
        yy_right + 0.1,
        alpha=1,
        linewidth=1,
        color="royalblue",
        label="Autowater collected",
    )
    ax.vlines(
        xx_left,
        yy_left - 0.1,
        yy_left - 0.05,
        alpha=1,
        linewidth=1,
        color="royalblue",
    )

    # Also highlight the autowater offered but still ignored
    xx = np.nonzero(autowater_ignored)[0] + 1
    yy = [1.01] * sum(autowater_ignored)
    ax.plot(
        *(xx, yy),
        "x",
        color="royalblue",
        markersize=3,
        markeredgewidth=0.5,
        label="Autowater ignored",
    )

    ax.set_yticks([0.875, 0.925, 1, 1.075, 1.125])
    ax.set_yticklabels(
        ["L Reward", "L Choice", "Ignored", "R Choice", "R Reward"]
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_foraging_multisession_inner(ax, plot, df):
    """
    plot just one metric
    ax, axis to plot on
    plot, metric to plot, must be a column of df
    df, multisession dataframe
    """
    # Set up axis
    ax.set_ylabel(plot)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # some metrics have special formatting
    # otherwise we just plot the metric
    if plot == "side_bias":
        ax.plot(df["multisession_trial"], df["side_bias"], label="bias")
        lower = [x[0] for x in df["side_bias_confidence_interval"]]
        upper = [x[1] for x in df["side_bias_confidence_interval"]]
        ax.fill_between(
            np.arange(0, len(df)), lower, upper, color="gray", alpha=0.25
        )
        ax.axhline(0, linestyle="--", color="k", alpha=0.25)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Side Bias")
    elif plot == "reward_probability":
        ax.plot(
            df["multisession_trial"],
            df["reward_probabilityL"],
            color="b",
            label="L",
        )
        ax.plot(
            df["multisession_trial"],
            df["reward_probabilityR"],
            color="r",
            label="R",
        )
        ax.set_ylabel("Reward Probability")
    elif plot == "lickspout_position":
        ax.axhline(0, linestyle="--", color="k", alpha=0.25)
        ax.plot(
            df["multisession_trial"],
            df["lickspout_position_z"] - df["lickspout_position_z"].values[0],
            "k",
            label="z",
        )
        ax.plot(
            df["multisession_trial"],
            df["lickspout_position_y1"]
            - df["lickspout_position_y1"].values[0],
            "r",
            label="y1",
        )
        ax.plot(
            df["multisession_trial"],
            df["lickspout_position_y2"]
            - df["lickspout_position_y2"].values[0],
            "m",
            label="y2",
        )
        ax.plot(
            df["multisession_trial"],
            df["lickspout_position_x"] - df["lickspout_position_x"].values[0],
            "b",
            label="x",
        )
        ax.set_ylabel("$\\Delta$ lickspout")
    elif plot in df:
        ax.plot(df["multisession_trial"], df[plot], label=plot)
        ax.axhline(0, linestyle="--", color="k", alpha=0.25)
    else:
        print("Unknown plot element: {}".format(plot))

    ax.legend(loc="upper left")


def find_multiday_breaks(df):
    """
    annotates if the gap between sessions is greater than a day
    """
    dates = [
        datetime.strptime(x.split("_")[1], "%Y-%m-%d")
        for x in np.sort(df["ses_idx"].unique())
    ]
    time_delta = []
    for dex, val in enumerate(dates):
        time_delta.append((dates[dex] - dates[dex - 1]) > timedelta(hours=24))
    time_delta.append(False)

    return time_delta
