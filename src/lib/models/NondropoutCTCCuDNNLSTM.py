from keras.layers import CuDNNLSTM, Dense, TimeDistributed, Bidirectional
from .BaseCTCModel import BaseCTCModel


class NondropoutCTCCuDNNLSTM(BaseCTCModel):
    def __init__(self, *args, **kwargs):
        super(NondropoutCTCCuDNNLSTM, self).__init__(*args, **kwargs)

    def _get_prediction_layer(self, mask):
        inner = Bidirectional(CuDNNLSTM(self.cells, return_sequences=True, kernel_initializer="he_normal"),
                              name="bilstm1")(mask)
        inner = Bidirectional(CuDNNLSTM(self.cells, return_sequences=True, kernel_initializer="he_normal"),
                              name="bilstm2")(inner)
        y_pred = TimeDistributed(
            Dense(len(self.feeder.phonemes_map), activation="softmax", kernel_initializer="he_normal"),
            name=self.output_layer)(inner)

        return y_pred
