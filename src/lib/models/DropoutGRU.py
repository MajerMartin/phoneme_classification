from keras.models import Sequential
from keras.layers import GRU, Dense
from BaseModel import BaseModel


class DropoutGRU(BaseModel):
    def __init__(self, *args, **kwargs):
        super(DropoutGRU, self).__init__(*args, **kwargs)

    def _compile_model(self):
        model = Sequential()
        model.add(GRU(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, input_shape=self.input_shape))
        model.add(GRU(128, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dense(self.output_shape, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

        return model
