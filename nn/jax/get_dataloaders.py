import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataloaders(batch_size=64):
    def preprocess(data):
        image = tf.cast(data["image"], tf.float32) / 255.0
        image = tf.expand_dims(image, -1)  # add a channel dimension
        label = tf.one_hot(data["label"], 10)
        return image, label

    ds_train = tfds.load(
        "mnist", split="train", data_dir="../mnist", as_supervised=False
    )
    ds_train = (
        ds_train.map(preprocess)
        .cache()
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    ds_test = tfds.load("mnist", split="test", data_dir="../mnist", as_supervised=False)
    ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test
