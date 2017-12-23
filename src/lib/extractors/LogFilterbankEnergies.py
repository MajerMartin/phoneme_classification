import numpy as np
from .BaseExtractor import BaseExtractor


class LogFilterbankEnergies(BaseExtractor):
    """
    Compute log filterbank energies.
    """

    def __init__(self, filters_count=26, **kwargs):
        """
        Initialize log filterbank energies.
        :param filters_count: (int) number of filters
        :param kwargs: (dict) keyword arguments passed to inherited classes
        """
        super(LogFilterbankEnergies, self).__init__(**kwargs)
        self.filters_count = filters_count
        self.feature_type = "log filterbank energies"
        self.fft_size = 512

    def _hz_to_mel(self, hz):
        """
        Convert Hz to Mel.
        :param hz: (float) value in Hz
        :return: (float) value in Mel
        """
        return 2595 * np.log10(1 + hz * 1.0 / 700)

    def _mel_to_hz(self, mel):
        """
        Convert Mel to Hz.
        :param mel: (float) value in Mel
        :return: (float) value in Hz
        """
        return 700 * (10 ** (mel * 1.0 / 2595) - 1)

    def _get_power_spectrum(self, frames):
        """
        Calculate periodogram estimate of power spectrum for each frame.
        :param frames: (ndarray) signal split into frames
        :return: (ndarray) power spectrum for each frame
        """
        # np.square is element-wise
        return 1.0 / self.fft_size * np.square(np.abs(np.fft.rfft(frames, self.fft_size)))

    def _get_filterbanks(self, rate):
        """
        Assemble Mel-filterbanks.
        :param rate: (int) frame rate of audio signal
        :return: (ndarray) filterbanks
        """
        low_freq = 0
        high_freq = rate // 2

        # convert Hz to Mel
        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)

        # calculate filter points linearly spaced between lowest and highest frequency
        mel_points = np.linspace(low_mel, high_mel, self.filters_count + 2)

        # convert points back to Hz
        hz_points = self._mel_to_hz(mel_points)

        # round frequencies to nearest fft bin
        fft_bin = np.floor((self.fft_size + 1) * hz_points / rate)

        # first filterbank will start at the first point, reach its peak at the second point
        # then return to zero at the 3rd point. The second filterbank will start at the 2nd
        # point, reach its max at the 3rd, then be zero at the 4th etc.
        filterbanks = np.zeros([self.filters_count, self.fft_size // 2 + 1])

        for i in range(self.filters_count):
            # from left to peak
            for j in range(int(fft_bin[i]), int(fft_bin[i + 1])):
                filterbanks[i, j] = (j - fft_bin[i]) / (fft_bin[i + 1] - fft_bin[i])
            # from peak to right
            for j in range(int(fft_bin[i + 1]), int(fft_bin[i + 2])):
                filterbanks[i, j] = (fft_bin[i + 2] - j) / (fft_bin[i + 2] - fft_bin[i + 1])

        return filterbanks

    def _get_log_filterbank_energies(self, frames, rate):
        """
        Compute log Mel-filterbank energies for every frame.
        :param frames: (ndarray) signal split into frames
        :param rate: (int) frame rate of audio signal
        :return: (ndarrays) log filterbank energies for every frame
        """
        power_spectrum = self._get_power_spectrum(frames)
        filterbanks = self._get_filterbanks(rate)

        # weighted sum of the fft energies around filterbank frequencies
        filterbank_energies = np.dot(power_spectrum, filterbanks.T)

        # replace zeroes with machine epsilon to prevent errors in log operation
        filterbank_energies = np.where(filterbank_energies == 0, np.finfo(float).eps, filterbank_energies)

        return np.log(filterbank_energies)

    def _extract_features(self, frames, rate):
        """
        Compute log Mel-filterbank energies for every frame.
        :param frames: (ndarray) signal split into frames
        :param rate: (int) frame rate of audio signal
        :return: (ndarrays) log filterbank energies for every frame
        """
        return self._get_log_filterbank_energies(frames, rate)
