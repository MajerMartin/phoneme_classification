from keras.models import Sequential
from keras.layers import Dense
from BaseModel import BaseModel


class TestModelMLP(BaseModel):
    def __init__(self, *args, **kwargs):
        super(TestModelMLP, self).__init__(*args, **kwargs)

    def _compile_model(self):
        model = Sequential()
        model.add(Dense(units=1024, input_dim=self.input_shape, activation="softmax"))
        model.add(Dense(units=64, input_dim=2048, activation="softmax"))
        model.add(Dense(units=1024, input_dim=64, activation="softmax"))
        model.add(Dense(units=self.output_shape, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

        return model
