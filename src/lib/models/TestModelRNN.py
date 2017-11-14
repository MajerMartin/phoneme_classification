from keras.models import Sequential
from keras.layers import LSTM, Dense
from BaseModel import BaseModel


class TestModelRNN(BaseModel):
    def __init__(self, *args, **kwargs):
        super(TestModelRNN, self).__init__(*args, **kwargs)

    def _compile_model(self):
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=self.input_shape))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(self.output_shape, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

        return model
