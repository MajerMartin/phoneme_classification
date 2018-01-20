from itertools import groupby, chain
from .BaseCTCModel import BaseCTCModel


class BatchCTCModel(BaseCTCModel):
    """
    Compile model, train and predict on data provided by feeder.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize batched CTC model.
        :param kwargs: (list) positional arguments passed to inherited classes
        :param kwargs: (dict) keyword arguments passed to inherited classes
        """
        super(BatchCTCModel, self).__init__(*args, **kwargs)

    def predict(self):
        """
        Predict on test set.
        :return: (tuple) ndarray of predicted phonemes per utterance, list of ndarrays with transcriptions
        """
        predictions, transcriptions = super(BatchCTCModel, self).predict()

        predictions_by_utterance_tmp = self.feeder.split_by_utterance(predictions, "test")
        predictions_by_utterance = []

        for prediction in predictions_by_utterance_tmp:
            predictions_by_utterance.append([k for k, g in groupby(list(chain(*prediction)))])

        return predictions_by_utterance, transcriptions
