import os, sys, datetime, shutil
import pickle
from tqdm import tqdm
os.environ['TF_ENABLE_NHWC']='1'
#os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"



import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
import numpy as np
from models import build_resnext50, get_classifier
from dataset import get_dataset



batch_size = 4
train_ds, val_ds = get_dataset(batch_size)

train_ds.repeat(-1)
val_ds.repeat(-1)

backbone = build_resnext50()
classifier = get_classifier(1000)

training_model = tf.keras.Sequential(layers=[backbone, classifier])
output = training_model(tf.zeros([1,224,224,3]))

num_elements_train = 1281167
num_elements_val   = 50000

num_batch_train = num_elements_train // batch_size
num_batch_val   = num_elements_val   // batch_size

num_epochs = 100

training_model.compile(
    optimizer=tf.keras.optimizers.Adam( 0.1 ),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = [ tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy()],
)

class CustomModelSaver(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0 and batch != 0:
            backbone.save_weights('./weights/backbone/backbone')
            classifier.save_weights('./weights/classifier/classifier')

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    CustomModelSaver(),
    tf.keras.callbacks.TensorBoard(histogram_freq = 10, update_freq=100, profile_batch = 0),
    tf.keras.callbacks.ReduceLROnPlateau( patience = 5, factor = 0.1, min_lr=1e-6)
]

try:
    backbone.load_weights('./weights/backbone/backbone')
    classifier.load_weights('./weights/classifier/classifier')
except Exception as e:
    print("Couldn't load weights.")

training_model.fit(
    train_ds,
    epochs=num_epochs,
    steps_per_epoch=num_batch_train,
    validation_data=val_ds,
    validation_steps=num_batch_val,
    callbacks = callbacks
)
