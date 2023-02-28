from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from spektral.layers import GCNConv, GlobalSumPool
from tensorflow.keras.regularizers import l2
import parameters
import methods_gcng as methods


l2_reg = parameters.l2_reg

# Build model
class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.conv2 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.flatten = GlobalSumPool() 
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output


# Training function
@tf.function
def train_on_batch(inputs, target, model):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    loss = parameters.loss_fn(target, predictions) + sum(model.losses)
  gradients = tape.gradient(loss, model.trainable_variables)
  parameters.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, predictions


# Evaluation function
def evaluate(loader, model):  
  inputs = (loader[0][0], loader[0][1])
  target = loader[1]

  predictions = model((inputs), training=False)
  loss =  parameters.loss_fn(target, predictions)
  acc = tf.reduce_mean(binary_accuracy(target, predictions))
  sensitivity, specificity = methods.calc_sen_spe(target, predictions)

  return (loss, acc, predictions, sensitivity, specificity)



