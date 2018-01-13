from keras.layers import Masking, LSTM, Dense, TimeDistributed, Bidirectional
from .BaseCTCModel import BaseCTCModel


class DropoutCTCLSTM(BaseCTCModel):
    def __init__(self, *args, **kwargs):
        super(DropoutCTCLSTM, self).__init__(*args, **kwargs)

    def _get_prediction_layer(self, input_data):
        mask = Masking(mask_value=0., name="mask")(input_data)
        inner = Bidirectional(LSTM(self.cells, dropout=0.25, recurrent_dropout=0.25, return_sequences=True,
                                   kernel_initializer="he_normal"), name="bilstm1")(mask)
        inner = Bidirectional(LSTM(self.cells, dropout=0.25, recurrent_dropout=0.25, return_sequences=True,
                                   kernel_initializer="he_normal"), name="bilstm2")(inner)
        y_pred = TimeDistributed(
            Dense(len(self.feeder.phonemes_map), activation="softmax", kernel_initializer="he_normal"),
            name=self.output_layer)(inner)

        return y_pred
