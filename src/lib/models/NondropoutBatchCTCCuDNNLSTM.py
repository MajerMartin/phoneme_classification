from keras.layers import CuDNNLSTM, Dense, TimeDistributed
from .BatchCTCModel import BatchCTCModel


class NondropoutBatchCTCCuDNNLSTM(BatchCTCModel):
    def __init__(self, *args, **kwargs):
        super(NondropoutBatchCTCCuDNNLSTM, self).__init__(*args, **kwargs)

    def _get_prediction_layer(self, input_data):
        inner = CuDNNLSTM(self.cells, return_sequences=True, kernel_initializer="he_normal", name="lstm1")(input_data)
        inner = CuDNNLSTM(self.cells, return_sequences=True, kernel_initializer="he_normal", name="lstm2")(inner)
        y_pred = TimeDistributed(
            Dense(len(self.feeder.phonemes_map), activation="softmax", kernel_initializer="he_normal"),
            name=self.output_layer)(inner)

        return y_pred
