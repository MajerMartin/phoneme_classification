from keras.models import Sequential
from keras.layers import Dense, Dropout
from BaseModel import BaseModel


class BottleneckDropoutMLP(BaseModel):
    def __init__(self, *args, **kwargs):
        super(BottleneckDropoutMLP, self).__init__(*args, **kwargs)

    def _compile_model(self):
        model = Sequential()
        model.add(Dense(units=1028, input_dim=self.input_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=64, input_dim=1028, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=1028, input_dim=64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=self.output_shape, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

        return model
