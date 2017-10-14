from scipy.fftpack import dct
from LogFilterbankEnergies import LogFilterbankEnergies


class MFCC(LogFilterbankEnergies):
    """
    Compute Mel-frequency cepstral coefficients.
    """

    def __init__(self, coeffs_count=13, **kwargs):
        """
        Initialize MFCC.
        :param coeffs_count: (int) number of coefficients to keep
        :param kwargs: (dict) keyword arguments passed to inherited classes
        """
        super(MFCC, self).__init__(**kwargs)
        self.coeffs_count = coeffs_count
        self.feature_type = "mfcc"

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
