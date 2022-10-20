import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import filtfilt, butter
from acconeer.exptool.a121.algo import interpolate_peaks, find_peaks, APPROX_BASE_STEP_LENGTH_M


def estimate_distance(frame, sensor_config):
    """Estimates the distance between the sensor and the underlying floor."""
    (B, A) = butter(N=2, Wn=0.3)

    mean_sweep = np.abs(frame).mean(axis=0)
    mean_sweep = filtfilt(B, A, mean_sweep)

    found_peaks = find_peaks(mean_sweep, np.zeros_like(mean_sweep))

    peak_index = found_peaks
    (estimated_distances, _) = interpolate_peaks(
        mean_sweep,
        peak_index,
        sensor_config.start_point,
        sensor_config.step_length,
        APPROX_BASE_STEP_LENGTH_M
        )

    return estimated_distances


def calculate_variance_at_fixed_distance(frame):
    """Calculates the variance across multiple sweeps at a given distance from the sensor."""
    distance = 28
    return np.var(frame[:,distance])


def plot_feature_by_file_number(df, feature, num_bins):
    """Plots the distribution of the given feature."""

    # Range in histogram plot.
    upper_plot_limit = np.percentile(df[feature], 97)
    lower_plot_limit = np.percentile(df[feature], 1)

    plt.figure(num=1, figsize=(20, 5))
    axs1 = plt.gca()
    plt.figure(num=2, figsize=(20, 5))
    axs2 = plt.gca()

    for file_num in range(df['file_num'].max() + 1):
        floor_idx = (df['labels'] == 'floor') & (df['file_num'] == file_num)
        carpet_idx = (df['labels'] == 'carpet') & (df['file_num'] == file_num)

        if floor_idx.any():
            df[floor_idx].plot.hist(
                column=feature,
                label='asd',
                range=[lower_plot_limit, upper_plot_limit],
                bins=num_bins,
                grid=True,
                ax=axs1,
                alpha=0.5,
                title='floor'
                )
            plt.legend()

        if carpet_idx.any():
            df[carpet_idx].plot.hist(
                column=feature,
                range=[lower_plot_limit, upper_plot_limit],
                bins=num_bins,
                grid=True,
                ax=axs2,
                alpha=0.5,
                title='carpet'
                )
            plt.legend()
        plt.legend()
