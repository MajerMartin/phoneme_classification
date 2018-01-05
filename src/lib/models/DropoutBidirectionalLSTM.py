from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras.optimizers import RMSprop
from .BaseModel import BaseModel


class DropoutBidirectionalLSTM(BaseModel):
    def __init__(self, *args, **kwargs):
        super(DropoutBidirectionalLSTM, self).__init__(*args, **kwargs)

    def _compile_model(self):
        model = Sequential()
        model.add(Bidirectional(
            LSTM(256, dropout=0.25, recurrent_dropout=0.25, return_sequences=True, input_shape=self.input_shape)))
        model.add(Bidirectional(LSTM(256, dropout=0.25, recurrent_dropout=0.25)))
        model.add(Dense(self.output_shape, activation="softmax"))

        if self.learning_rate:
            lr = self.learning_rate
        else:
            lr = 0.001

        rmsprop = RMSprop(lr=lr)

        model.compile(loss="categorical_crossentropy", optimizer=rmsprop, metrics=["accuracy"])

        return model
