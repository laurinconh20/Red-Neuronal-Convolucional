"""Entrena una CNN para clasificar frutas frescas y podridas."""
from pathlib import Path
import argparse
import tensorflow as tf

def find_directory(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob('*') if p.is_dir() and p.name.lower() == name]
    if not matches: raise FileNotFoundError(f"No se encontró la carpeta '{name}' dentro de {root}")
    return matches[0]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/modelo_frutas_final.keras'))
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    root = args.data
    if root is None:
        import kagglehub
        root = Path(kagglehub.dataset_download('sriramr/fruits-fresh-and-rotten-for-classification'))
    train_dir, test_dir = find_directory(root, 'train'), find_directory(root, 'test')
    size, batch, seed = (224, 224), 32, 123
    train = tf.keras.utils.image_dataset_from_directory(train_dir, validation_split=.2, subset='training', seed=seed, image_size=size, batch_size=batch)
    val = tf.keras.utils.image_dataset_from_directory(train_dir, validation_split=.2, subset='validation', seed=seed, image_size=size, batch_size=batch)
    test = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=size, batch_size=batch, shuffle=False)
    scale = tf.keras.layers.Rescaling(1 / 255.)
    prep = lambda ds: ds.map(lambda x, y: (scale(x), y)).cache().prefetch(tf.data.AUTOTUNE)
    train, val, test = prep(train), prep(val), prep(test)
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(.15), tf.keras.layers.RandomZoom(.15), tf.keras.layers.RandomContrast(.1)])
    model = tf.keras.Sequential([tf.keras.layers.Input((224, 224, 3)), aug, tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(64, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(128, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(.5), tf.keras.layers.Dense(len(train.class_names), activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    _, accuracy = model.evaluate(test, verbose=0)
    args.output.parent.mkdir(parents=True, exist_ok=True); model.save(args.output)
    print(f'Modelo guardado en: {args.output}'); print(f'Precisión en prueba: {accuracy * 100:.2f}%')

if __name__ == '__main__': main()
