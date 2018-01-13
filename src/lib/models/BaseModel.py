import os
from keras.backend import backend
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback, CSVLogger


class BaseModel(object):
    """
    Compile model, train and predict on data provided by feeder.
    """

    def __init__(self, feeder, epochs, batch_size, learning_rate=None, callbacks=[], cells=128):
        """
        Initialize base model.
        :param feeder: (object) feeder object
        :param epochs: (int) number of epochs
        :param batch_size: (int) batch size
        :param callbacks: (list) model callbacks to use
        :param cells: (list) cells per layer for recurrent neural networks
        """
        # TODO: support multiple feeders - train, validation, test
        self.feeder = feeder
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cells = cells

        self.initial_epoch = 0

        # define logging paths
        features_name = os.path.splitext(os.path.basename(self.feeder.features_path))[0]
        self.model_name = str(id(self)) + "_" + self.__class__.__name__ + "_" + str(self.feeder.left_context) + "_" + str(
            self.feeder.right_context) + "_" + str(self.feeder.time_steps)

        self.models_path_prefix = os.path.join("..", "weights", features_name)

        self.model_checkpoint_path = os.path.join(self.models_path_prefix, self.model_name + ".h5")
        self.metadata_path = os.path.join(self.models_path_prefix, self.model_name + "_metadata.txt")
        self.tensorboard_log_path = os.path.join("..", "logs", "tensorboard", features_name, self.model_name)
        self.csv_log_path = os.path.join("..", "logs", "csv", features_name, self.model_name + ".csv")

        # define input and output shapes
        if self.feeder.time_steps or self.feeder.ctc:
            self.input_shape = (feeder.get_dim("X", "train", 1), feeder.get_dim("X", "train", 2))
        else:
            self.input_shape = feeder.get_dim("X", "train", 1)

        if self.feeder.ctc:
            self.output_shape = None
        else:
            self.output_shape = feeder.get_dim("y", "train", 1)

        # build model
        self.callbacks = self._set_callbacks(callbacks)
        self.model = self._compile_model()

    def _create_dirs(self):
        """
        Create directories for model checkpoints and tensorboard log if they do not exists.
        """
        for filename in [self.model_checkpoint_path, self.tensorboard_log_path, self.csv_log_path]:
            dirname = os.path.dirname(filename)

            if not os.path.exists(dirname):
                os.makedirs(dirname)

    def _save_model_metadata(self):
        """
        Save model metadata into text file.
        """
        format_print = lambda k, v: "{:25}|\t{}\n".format(k, v)

        model_keys = ["epochs", "batch_size", "learning_rate", "model_checkpoint_path", "tensorboard_log_path",
                      "csv_log_path"]
        feeder_keys = ["features_path", "left_context", "right_context", "time_steps", "train_speakers", "val_speakers",
                       "test_speakers", "noise", "ctc"]

        with open(self.metadata_path, "w") as fw:
            for key in model_keys:
                fw.write(format_print(key, self.__dict__[key]))

            for key in feeder_keys:
                fw.write(format_print(key, self.feeder.__dict__[key]))

            fw.write("\n")
            self.model.summary(print_fn=lambda x: fw.write(x + "\n"))

    def _set_callbacks(self, callbacks):
        """
        Set callbacks used by model.
        :param callbacks: (list) model callbacks to use
        :return: (list) initialized callbacks
        """
        batches_count = int(self.feeder.get_steps_per_epoch(self.batch_size, "train"))

        def batch_print(batch, logs):
            if batch % 5000 == 0:
                print("\r  Batch: {}/{}".format(batch, batches_count - 1), end=" ")
            elif batch == batches_count - 1:
                print("\r  Batch: {}/{}\n".format(batch, batches_count - 1), end=" ")

        callbacks_init = {
            "modelCheckpoint": ModelCheckpoint(self.model_checkpoint_path, save_best_only=True, save_weights_only=False,
                                               verbose=1),
            "reduceLROnPlateau": ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001, verbose=1),
            "earlyStopping": EarlyStopping(patience=10, verbose=1),
            "CSVLogger": CSVLogger(self.csv_log_path),
            "batchPrint": LambdaCallback(on_batch_begin=batch_print)
        }

        if backend() == "tensorflow":
            from keras.callbacks import TensorBoard
            callbacks_init["tensorboard"] = TensorBoard(log_dir=self.tensorboard_log_path, write_graph=True)

        return [callbacks_init[cb] for cb in callbacks]

    def _compile_model(self):
        """
        Compile model - implemented in inherited classes.
        """
        raise NotImplementedError("Implemented in inherited classes.")

    def train(self):
        """
        Train model on train set and validate on validation set.
        """
        self._create_dirs()
        self._save_model_metadata()

        self.model.fit_generator(self.feeder.yield_batches(self.batch_size, "train"),
                                 steps_per_epoch=self.feeder.get_steps_per_epoch(self.batch_size, "train"),
                                 nb_epoch=self.epochs,
                                 verbose=2,
                                 callbacks=self.callbacks,
                                 validation_data=self.feeder.yield_batches(self.batch_size, "val"),
                                 validation_steps=self.feeder.get_steps_per_epoch(self.batch_size, "val"),
                                 initial_epoch=self.initial_epoch)

    def predict(self):
        """
        Predict on test set.
        :return: (tuple) ndarray of predicted phonemes per utterance, list of ndarrays with transcriptions
        """
        predictions = self.model.predict_generator(self.feeder.yield_batches(self.batch_size, "test", False),
                                                   steps=self.feeder.get_steps_per_epoch(self.batch_size, "test"),
                                                   verbose=1)

        predictions_by_utterance = self.feeder.split_by_utterance(predictions, "test")

        transcriptions = self.feeder.get_transcriptions("test")

        return predictions_by_utterance, [t for t in transcriptions]

    def evaluate(self):
        """
        Evaluate on test set.
        :return: (dict) scores of metrics
        """
        scores = self.model.evaluate_generator(self.feeder.yield_batches(self.batch_size, "test", False),
                                               steps=self.feeder.get_steps_per_epoch(self.batch_size, "test"))

        return {metric: score for score, metric in zip(scores, self.model.metrics_names)}

    def load_model(self, model_name, initial_epoch=0):
        """
        Load saved model.
        """
        self.initial_epoch = initial_epoch
        self.model = load_model(os.path.join(self.models_path_prefix, model_name))

    def __str__(self):
        """
        Override.
        :return: (string) class representation
        """
        output = ""

        for key, value in self.__dict__.items():
            if type(value) == str:
                value = "'{}'".format(value)
            output += "{{'{0}': {1}}}, ".format(key, value)

        return output[:-2]

    def __repr__(self):
        """
        Override.
        :return: (string) class representation
        """
        return self.__str__()
