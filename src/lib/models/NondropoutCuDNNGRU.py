from keras.models import Sequential
from keras.layers import CuDNNGRU, Dense
from keras.optimizers import RMSprop
from .BaseModel import BaseModel


class NondropoutCuDNNGRU(BaseModel):
    def __init__(self, *args, **kwargs):
        super(NondropoutCuDNNGRU, self).__init__(*args, **kwargs)

    def _compile_model(self):
        model = Sequential()
        model.add(CuDNNGRU(128, return_sequences=True, input_shape=self.input_shape))
        model.add(CuDNNGRU(128))
        model.add(Dense(self.output_shape, activation="softmax"))

        if self.learning_rate:
            lr = self.learning_rate
        else:
            lr = 0.001

        rmsprop = RMSprop(lr=lr)

        model.compile(loss="categorical_crossentropy", optimizer=rmsprop, metrics=["accuracy"])

        return model
