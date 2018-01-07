from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense
from keras.optimizers import Adam
from .BaseModel import BaseModel


class NondropoutCuDNNLSTM(BaseModel):
    def __init__(self, *args, **kwargs):
        super(NondropoutCuDNNLSTM, self).__init__(*args, **kwargs)

    def _compile_model(self):
        model = Sequential()
        model.add(CuDNNLSTM(128, return_sequences=True, input_shape=self.input_shape))
        model.add(CuDNNLSTM(128))
        model.add(Dense(self.output_shape, activation="softmax"))

        if self.learning_rate:
            lr = self.learning_rate
        else:
            lr = 0.001

        adam = Adam(lr=lr)

        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model
