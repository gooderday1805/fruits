import tensorflow as tf
from model import build_model
from data_preprocessing import load_data

def train_model():
    train_data, val_data = load_data('data/train')

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=callbacks
    )

    model.save('models/saved_models/model.keras')
    print("✅ Model saved to models/saved_models/model.keras")

    return history
if __name__ == "__main__":
    train_model()
