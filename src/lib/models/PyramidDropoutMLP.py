from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
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

        if self.learning_rate:
            lr = self.learning_rate
        else:
            lr = 0.01

        sgd = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=True)

        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        return model
