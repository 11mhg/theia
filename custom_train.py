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

total_train_steps = num_epochs * num_batch_train


schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [ int(np.floor( 0.5 * num_epochs * num_batch_train )), int(np.floor( 0.85 * num_epochs * num_batch_train )) ], 
    [ 0.5, 0.25, 0.01]
)
optim = tf.keras.optimizers.Adam( learning_rate = schedule )

@tf.function()
def train_step( img, label ):
    with tf.GradientTape() as gt:
        pred_label = training_model( img )
        loss = tf.keras.losses.categorical_crossentropy( label, pred_label )
    grads = gt.gradient( loss, training_model.trainable_variables )
    optim.apply_gradients( zip(grads, training_model.trainable_variables) )
    return pred_label, loss, grads

@tf.function()
def val_step( img, label ):
    pred_label = training_model( img )
    loss = tf.keras.losses.categorical_crossentropy( label, pred_label )
    return pred_label, loss

training_info = {}

if os.path.exists('./training_info.pkl'):
    with open('./training_info.pkl', 'rb') as f:
        training_info = pickle.load( f )
else:
    training_info = {
        'step': 0
    }
    with open('./training_info.pkl', 'wb') as f:
        pickle.dump( training_info, f )

try:
    backbone.load_weights('./weights/backbone/backbone')
    classifier.load_weights('./weights/classifier/classifier')
    print("Model weights found! Reloading!")
except Exception as e:
    backbone.save_weights('./weights/backbone/backbone')
    classifier.save_weights('./weights/classifier/classifier')
    print("Did not find weights. Resaving")


train_loss_mean = tf.keras.metrics.Mean(name='train_loss')
val_loss_mean   = tf.keras.metrics.Mean( name='val_loss')




train_accuracy = tf.keras.metrics.CategoricalAccuracy()
val_accuracy   = tf.keras.metrics.CategoricalAccuracy()


train_top_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(name='train_top_k_accuracy')
val_top_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(name='val_top_k_accuracy')



def reset_all_metrics():
    train_loss_mean.reset_state()
    val_loss_mean.reset_state()
    train_accuracy.reset_state()
    val_accuracy.reset_state()
    train_top_k_accuracy.reset_state()
    val_top_k_accuracy.reset_state()


log_dir="logs/"

train_writer = tf.summary.create_file_writer( log_dir + '/train/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
val_writer   = tf.summary.create_file_writer( log_dir + '/val/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )


step = training_info['step']

train_ds_iterator = iter(train_ds)
val_ds_iterator   = iter(val_ds)


pbar = tqdm( range(training_info['step'], total_train_steps ) )

try:
    for step in pbar:
        img, label = next(train_ds_iterator)
        pred_label, loss, grads = train_step( img, label )

        train_loss_mean.update_state( loss )
        train_accuracy.update_state( label, pred_label )
        train_top_k_accuracy.update_state( label, pred_label )

        if step % 100 == 0:
            backbone.save_weights('./weights/backbone/backbone')
            classifier.save_weights('./weights/classifier/classifier')
            with train_writer.as_default():
                tf.summary.scalar( 'loss', train_loss_mean.result(), step=step )
                tf.summary.scalar( 'accuracy', train_accuracy.result(), step=step )
                tf.summary.scalar( 'top_k_accuracy', train_top_k_accuracy.result(), step=step )


        if step % num_batch_train == 0 and step != 0:
            for ind in range(0, num_batch_val):
                img, label = next(val_ds_iterator)
                pred_label, loss = val_step( img, label )

                val_loss_mean.update_state( loss )
                val_accuracy.update_state( label, pred_label )
                val_top_k_accuracy.update_state( label, pred_label )
            
            with val_writer.as_default():
                tf.summary.scalar( 'loss', val_loss_mean.result(), step=step )
                tf.summary.scalar( 'accuracy', val_accuracy.result(), step=step )
                tf.summary.scalar( 'top_k_accuracy', val_top_k_accuracy.result(), step=step )
            reset_all_metrics()
        training_info['step'] = step
finally:
    backbone.save_weights('./weights/backbone/backbone')
    classifier.save_weights('./weights/classifier/classifier')
    with open('./training_info.pkl', 'wb') as f:
        pickle.dump( training_info, f)

