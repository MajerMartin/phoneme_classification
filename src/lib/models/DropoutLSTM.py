from keras.models import Sequential
from keras.layers import LSTM, Dense
from .BaseModel import BaseModel


class DropoutLSTM(BaseModel):
    def __init__(self, *args, **kwargs):
        super(DropoutLSTM, self).__init__(*args, **kwargs)

    def _compile_model(self):
        model = Sequential()
        model.add(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True, input_shape=self.input_shape))
        model.add(LSTM(128, dropout=0.25, recurrent_dropout=0.25))
        model.add(Dense(self.output_shape, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model
