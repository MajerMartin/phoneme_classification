from keras.models import Sequential
from keras.layers import Dense, Dropout
from .BaseModel import BaseModel


class PyramidDropoutMLP(BaseModel):
    def __init__(self, *args, **kwargs):
        super(PyramidDropoutMLP, self).__init__(*args, **kwargs)

    def _compile_model(self):
        model = Sequential()
        model.add(Dense(units=1028, input_dim=self.input_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=512, input_dim=1028, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=256, input_dim=512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=self.output_shape, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

        return model
