import tensorflow as tf
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.optimizers.legacy import Adam
from settings import settings


def get_model_2d(
        model_filepath='models/2d_models/EfficientNetB3-saved-model-30-val_acc-1.00.hdf5',
):
    model_filepath = settings.storage_folder.joinpath(model_filepath)
    efficienb3 = efficientnet.EfficientNetB3(weights='imagenet',
                                             include_top=False,
                                             input_shape=(300, 300, 3),
                                             classes=5)

    efficienb3.trainable = True

    opt = Adam(learning_rate=5e-5)
    model = tf.keras.Sequential([
        efficienb3,
        tf.keras.layers.BatchNormalization(renorm=True),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.load_weights(model_filepath)

    return model
