import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch, find_peaks, savgol_filter
from BaselineRemoval import BaselineRemoval


class FeatureTransform:
    """
    A class which accepts a time series signal and converts those into frequency domain to extract features for ML models.
    There are several transformation available:
    get_ffts: provides frequencies and intensities.
    get_psds: provides frequencies and power spectral denisity values
    get_autocorrs: provides autocorreltaion values.
    get_first_n_features: removes baselines, applies a Svazisky-Golay filter, detect peaks and stores user defined first n
    frequencies and intensities.
    """

    def __init__(self, signal_ints, total_time=2.56, freq=50, signal_length=128):
        self.signal_ints = signal_ints
        self.total_time = total_time
        self.freq = freq
        self.signal_length = signal_length
        self.period = total_time / signal_length
        self.sample_rate = 1.0 / freq

    def normalize_data(self, signal_ints):
        self.signal_ints = signal_ints
        signal_ints = (signal_ints - np.min(signal_ints)) / np.ptp(signal_ints)
        return signal_ints

    def get_ints(self, signal_ints):
        self.signal_ints = signal_ints
        time = [self.sample_rate * i for i in range(0, len(self.signal_ints))]
        return time, signal_ints

    def get_ffts(self, signal_ints):
        self.signal_ints = signal_ints
        fft_freqs = np.linspace(0.0, 1.0 / (2.0 * self.period), self.signal_length // 2)
        fft_ints_ = fft(signal_ints)
        fft_ints = 2.0 / self.signal_length * np.abs(fft_ints_[0:self.signal_length // 2])
        fft_ints = fft_ints - np.mean(fft_ints)
        #         fft_ints = self.normalize_data(fft_ints)
        return fft_freqs, fft_ints

    def get_psds(self, signal_ints):
        self.signal_ints = signal_ints
        freqs, psd_ints = welch(signal_ints, fs=self.freq)
        #         psd_ints = self.normalize_data(psd_ints)
        return freqs, psd_ints

    def autocorr(self, x):
        result = np.correlate(x, x, mode='full')
        return result[len(result) // 2:]

    def get_autocorrs(self, signal_ints):
        self.signal_ints = signal_ints
        autocorr_ints = self.autocorr(signal_ints)
        #         autocorr_ints = self.normalize_data(autocorr_ints)
        freqs = np.array([self.period * j for j in range(0, self.signal_length)])
        return freqs, autocorr_ints

    def get_first_n_features(self, x, y, n_peaks, filter_order=5, filter_poly=2):
        baseObj = BaselineRemoval(y)
        y = baseObj.ModPoly(filter_poly)
        y = savgol_filter(y, filter_order, filter_poly)
        peaks, _ = find_peaks(y)
        first_n_freq = x[peaks][np.argsort(-y[peaks])][0:n_peaks]
        first_n_int = y[peaks][np.argsort(-y[peaks])][0:n_peaks]
        zero_pad_freq = [0] * n_peaks
        zero_pad_int = [0] * n_peaks
        zero_pad_freq[:len(first_n_freq)] = first_n_freq
        zero_pad_int[:len(first_n_int)] = first_n_int
        freq_int = list(zero_pad_freq) + list(zero_pad_int)
        return freq_int

    def get_feature_matrix(self, data, labels, n_peaks):
        all_features = []
        all_labels = []
        for signal_no in range(0, len(data)):
            features = []
            all_labels.append(labels[signal_no])
            for signal_comp in range(0, data.shape[2]):
                signal = data[signal_no, :, signal_comp]
                features += self.get_first_n_features(*self.get_psds(signal), n_peaks=n_peaks)
                features += self.get_first_n_features(*self.get_ffts(signal), n_peaks=n_peaks)
                features += self.get_first_n_features(*self.get_autocorrs(signal), n_peaks=n_peaks)
            all_features.append(features)
        return np.array(all_features), np.array(all_labels)
