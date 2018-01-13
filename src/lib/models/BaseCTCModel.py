import time
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from tensorflow.python.ops import ctc_ops as ctc
from keras.layers import Input, Masking, Lambda
from itertools import groupby
from .BaseModel import BaseModel


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """
    FROM KERAS - MODIFIED FOR BATCH SIZE OF ONE.
    Runs CTC loss algorithm on each batch element.
    # Arguments
        y_true: tensor `(samples, max_string_length)`
            containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_true`.
    # Returns
        Tensor with shape (samples,1) containing the
            CTC loss of each element.
    """
    label_length = tf.to_int32(tf.squeeze(label_length, axis=1))
    input_length = tf.to_int32(tf.squeeze(input_length, axis=1))
    sparse_labels = tf.to_int32(K.ctc_label_dense_to_sparse(y_true, label_length))

    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + K.epsilon())

    return tf.expand_dims(ctc.ctc_loss(inputs=y_pred,
                                       labels=sparse_labels,
                                       sequence_length=input_length), 1)


def ctc_lambda_func(args):
    """
    Function wrapper for CTC loss.
    :param args: (tensor, tensor, tensor, tensor) see function ctc_batch cost
    :return: (float) loss
    """
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


class LossHistory(Callback):
    """
    Custom callback for CTC loss logging.
    """

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))


class BaseCTCModel(BaseModel):
    """
    Compile model, train and predict on data provided by feeder.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize base CTC model.
        :param kwargs: (list) positional arguments passed to inherited classes
        :param kwargs: (dict) keyword arguments passed to inherited classes
        """
        # override epochs
        args = list(args)
        args[2] = 1

        # extract callbacks names
        self.used_callbacks = kwargs.get("callbacks", [])

        self.best_val_loss = np.inf
        self.flat_iterations = 0

        self.output_layer = "dense"

        super(BaseCTCModel, self).__init__(*args, **kwargs)

    def _get_prediction_layer(self, input_data):
        """
        Get prediction layer - implemented in inherited classes.
        :param input_data: (object) Keras input layer
        """
        raise NotImplementedError("Implemented in inherited classes.")

    def _compile_model(self):
        """
        Compile model.
        :return: (object) compiled model
        """
        input_data = Input(shape=(self.feeder.max_sequence_length, self.input_shape), dtype="float32",
                           name="the_input", )

        y_pred = self._get_prediction_layer(input_data)

        labels = Input(name="the_labels", shape=[self.feeder.max_labelling_length], dtype="float32")
        input_length = Input(name="input_length", shape=[1], dtype="int32")
        label_length = Input(name="label_length", shape=[1], dtype="int32")
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        if self.learning_rate:
            lr = self.learning_rate
        else:
            lr = 0.001

        adam = Adam(lr=lr)

        model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=adam)

        return model

    def _apply_callbacks(self, epoch, loss, val_loss):
        """
        Apply selected callbacks.
        :param epoch: (int) current epoch
        :param loss: (float) CTC loss on train set
        :param val_loss: (float) CTC loss on validation set
        :return: (boolean) apply early stopping
        """
        stop = False

        if "modelCheckpoint" in self.used_callbacks:
            self._simulate_modelCheckpoint(epoch, val_loss)

        if "earlyStopping" in self.used_callbacks:
            stop = self._simulate_earlyStopping(epoch, val_loss)

        if "CSVLogger" in self.used_callbacks:
            self._simulate_CSVLogger(epoch, loss, val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        return stop

    def _simulate_CSVLogger(self, epoch, loss, val_loss, write_header=False):
        """
        Simulate behaviour of CSVLogger callback from Keras.
        :param epoch: (int) current epoch
        :param loss: (float) CTC loss on train set
        :param val_loss: (float) CTC loss on validation set
        :param write_header: (boolean) write CSV header
        """
        with open(self.csv_log_path, "a") as fw:
            if write_header:
                fw.write("epoch,loss,val_loss")
            else:
                fw.write("{},{},{}".format(epoch, loss, val_loss))
            fw.write("\n")

    def _simulate_earlyStopping(self, epoch, val_loss, patience=0):
        """
        Simulate behaviour of earlyStopping callback from Keras.
        :param epoch: (int) current epoch
        :param val_loss: (float) CTC loss on validation set
        :param patience: (int) epochs to wait before stopping
        :return: (boolean) apply early stopping
        """
        if val_loss > self.best_val_loss:
            self.flat_iterations += 1
            print("Epoch {0:05d}: val_loss did not improve".format(epoch))
        else:
            self.flat_iterations = 0

        return self.flat_iterations == (patience + 1)

    def _simulate_modelCheckpoint(self, epoch, val_loss):
        """
        Simulate behaviour of modelCheckpoint callback from Keras.
        :param epoch: (int) current epoch
        :param val_loss: (float) CTC loss on validation set
        """
        if val_loss < self.best_val_loss:
            print("Epoch {0:05d}: val_loss improved from {1} to {2}, saving model to {3}".format(epoch,
                                                                                                 self.best_val_loss,
                                                                                                 val_loss,
                                                                                                 self.model_checkpoint_path))
            self.model.save(self.model_checkpoint_path)

    def train(self):
        """
                Train model on train set and validate on validation set.
                """
        self._create_dirs()
        self._save_model_metadata()

        if "CSVLogger" in self.used_callbacks:
            self._simulate_CSVLogger("dummy", "dummy", "dummy", write_header=True)

        for epoch in range(self.epochs):
            print("Epoch {0}/{1}".format(epoch + 1 + self.initial_epoch, self.epochs + self.initial_epoch))
            start_time = time.time()

            losses = []
            history = LossHistory()

            for inputs, outputs in self.feeder.yield_batches("train", shuffle=True):
                self.model.fit(inputs, outputs, batch_size=1, epochs=1, shuffle=False, verbose=0, callbacks=[history])
                losses.extend(history.losses)

            loss = np.mean(losses)
            val_loss = self.evaluate("val")["ctc"]

            stop = False
            if self._apply_callbacks(epoch + self.initial_epoch, loss, val_loss):
                stop = True

            print("  - {0}s - ctc: {1:0.4f} - val_ctc: {2:0.4f}".format(int(time.time() - start_time), loss, val_loss))

            if stop:
                print("Epoch {0:05d}: early stopping".format(epoch))
                break

    def predict(self):
        """
        Predict on test set.
        :return: (tuple) ndarray of predicted phonemes per utterance, list of ndarrays with transcriptions
        """
        pred_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(self.output_layer).output)

        predictions_by_utterance = []

        for inputs, outputs in self.feeder.yield_batches("test", shuffle=False):
            pred = [self.feeder.inverse_phonemes_map[index] for index in
                    np.argmax(pred_layer_model.predict_on_batch(inputs)[0, :, :], axis=1)]
            pred = [k for k, g in groupby(pred) if k != self.feeder.blank[0]]

            predictions_by_utterance.append(pred)

        transcriptions = self.feeder.get_transcriptions("test")

        return predictions_by_utterance, [t for t in transcriptions]

    def evaluate(self, split_type="test"):
        """
        Evaluate on selected set.
        :param split_type: (string) set to evaluate on (train/val/test)
        :return: (dict) scores of metrics
        """
        losses = []

        for inputs, outputs in self.feeder.yield_batches(split_type, shuffle=False):
            loss = self.model.evaluate(inputs, outputs, batch_size=1, verbose=0)
            losses.append(loss)

        return {"ctc": np.mean(losses)}
