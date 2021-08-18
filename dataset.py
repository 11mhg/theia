import os, sys

os.environ['TFDS_DATA_DIR'] = '/mnt/d/Datasets/'
if 'win' in sys.platform:
    os.environ['TFDS_DATA_DIR'] = 'D:\Datasets'



import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf



def get_dataset(batch_size=32, image_size =  [ 224, 288, 352, 416 ] ):
    dataset_name = "imagenet2012"
    num_classes = 1000
    max_image_size = max( image_size )
    image_size_t = tf.convert_to_tensor( image_size )


    imagenet_datasets_builder = tfds.builder(
        dataset_name, 
    )
    manual_dataset_dir = "/mnt/d/Datasets/"

    if 'win' in sys.platform:
        manual_dataset_dir = 'D:\Datasets'

    imagenet_download_config = tfds.download.DownloadConfig(
                                                manual_dir = manual_dataset_dir)
    
    imagenet_datasets_builder.download_and_prepare(
        download_dir=manual_dataset_dir
    )


    imagenet_train = imagenet_datasets_builder.as_dataset(
        split=tfds.Split.TRAIN
    )

    imagenet_validation = imagenet_datasets_builder.as_dataset(
        split = tfds.Split.VALIDATION
    )

    def parse_single( data ):
        img   = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
        img   = tf.image.resize( img, (max_image_size, max_image_size), method = tf.image.ResizeMethod.BILINEAR )
        data['image'] = img
        return data['image'], data['label']

    imagenet_train      = imagenet_train.map( parse_single      , num_parallel_calls=tf.data.AUTOTUNE ).shuffle( batch_size*10 ).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    imagenet_validation = imagenet_validation.map( parse_single , num_parallel_calls=tf.data.AUTOTUNE ).shuffle( batch_size*10 ).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    def parse_batch( image, label ):
        label               = tf.one_hot( label-1, num_classes, dtype = tf.float32 )
        random_image_size   = tf.random.uniform((), minval=0, maxval = len(image_size), dtype=tf.int32)
        chosen_image_size   = image_size_t[ random_image_size]
        image               = tf.image.resize( image, (chosen_image_size, chosen_image_size), method = tf.image.ResizeMethod.BILINEAR )
        return image, label

    imagenet_train      = imagenet_train.map( parse_batch      , num_parallel_calls=tf.data.AUTOTUNE ).prefetch( tf.data.AUTOTUNE )
    imagenet_validation = imagenet_validation.map( parse_batch , num_parallel_calls=tf.data.AUTOTUNE ).prefetch( tf.data.AUTOTUNE )

    return imagenet_train, imagenet_validation


if __name__ == "__main__":
    train_ds, val_ds = get_dataset()

    import tqdm, time

    pbar = tqdm.tqdm()

    start = time.time()
    for image, label in train_ds.take(100):
        pbar.update(1)
    print("\nTook {}".format( time.time() - start ))

    start = time.time()
    i = 0
    iterator = iter(train_ds)
    while i < 100:
        image, label = next(iterator)
        i+=1
    print("\nTook {}".format( time.time() - start ))

    for image, label in train_ds:
        pbar.update(1)

        
