import os
import tensorflow as tf
from data_preprocessing import get_data_generators
from model import create_model

train_dir = 'data/train'
test_dir = 'data/test'
model_path = 'models/saved_models/model.keras'

train_gen, test_gen = get_data_generators(train_dir, test_dir)
model = create_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
]

model.fit(
    train_gen,
    epochs=30,
    validation_data=test_gen,
    callbacks=callbacks
)

model.save(model_path)
