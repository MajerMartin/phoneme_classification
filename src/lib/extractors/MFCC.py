from scipy.fftpack import dct
from LogFilterbankEnergies import LogFilterbankEnergies


class MFCC(LogFilterbankEnergies):
    """
    Compute Mel-frequency cepstral coefficients.
    """

    def __init__(self, filters_count=26, coeffs_count=13):
        """
        Initialize MFCC.
        :param filters_count: (int) number of filters
        :param coeffs_count: (int) number of coefficients to keep
        """
        self.filters_count = filters_count
        self.coeffs_count = coeffs_count
        self.feature_type = "mfcc"
        self.fft_size = 512

    def _extract_features(self, frames, rate):
        """
        Compute Mel-frequency cepstral coefficients for every frame.
        :param frames: (ndarray) signal split into frames
        :param rate: (int) frame rate of audio signal
        :return: (ndarrays) MFCCs for every frame
        """
        log_filterbank_energies = self._get_log_filterbank_energies(frames, rate)
        mfcc = dct(log_filterbank_energies, type=2, axis=1, norm='ortho')[:, :self.coeffs_count]

        return mfcc
