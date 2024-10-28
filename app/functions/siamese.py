# -*- coding: utf-8 -*-
"""
Funciones de entrenamiento y prueba para un modelo de segmentación siamesa.
"""

import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from glob import glob 
from skimage.transform import resize # type: ignore
from imageio import imread # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

# Parámetros
data_dir = 'Imagenes/Endoscape/'
image_shape = (128, 128)
batch_size = 64

# Generador de datos
def generator(data_folder, image_shape, batch_size):
    image_paths = glob(os.path.join(data_folder, 'Imm', '*'))
    np.random.shuffle(image_paths)

    while True:
        for batch_i in range(0, len(image_paths), batch_size):
            images, mask_instruments, mask_backgrounds = [], [], []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                filename = os.path.basename(image_file).split('.')[0]
                image = resize(imread(image_file), image_shape, mode='reflect')

                mask_path2 = os.path.join(data_folder, 'Labelo', f'{filename}.png')
                mask_instrument = imread(mask_path2)
                mask_instrument = resize(mask_instrument, image_shape, mode='reflect')
                mask_instrument = (mask_instrument > 0.5).astype(np.float32)

                mask_background = 1 - mask_instrument
                mask_background = resize(mask_background, image_shape, mode='reflect')
                mask_background = (mask_background > 0.5).astype(np.float32)
                
                mask_instrument = np.expand_dims(mask_instrument, axis=-1)
                mask_background = np.expand_dims(mask_background, axis=-1)

                images.append(image)
                mask_instruments.append(mask_instrument)
                mask_backgrounds.append(mask_background)

            yield np.array(images), [np.array(mask_instruments), np.array(mask_backgrounds)]

# Métrica personalizada
def IOU_calc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1))
    return intersection / union

# Definición del modelo de segmentación siamesa
def siamese_segmentation_model(img_shape):
    inputs = Input(img_shape)
    # ... (Definición del modelo, igual que en el código proporcionado)
    return Model(inputs=inputs, outputs=[output_instrument, output_background])

# Función de entrenamiento
def entrenar_modelo(data_dir, image_shape, batch_size, epochs=25, model_path=None):
    # Cargar modelo o crear uno nuevo
    if model_path and os.path.exists(model_path):
        model = load_model(model_path, custom_objects={'IOU_calc': IOU_calc})
        print(f'Modelo cargado desde {model_path}')
    else:
        model = siamese_segmentation_model((128, 128, 3))
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[IOU_calc])

    # Generador de datos
    train_generator = generator(data_dir, image_shape, batch_size)
    
    # Configurar callbacks
    checkpoint_best = ModelCheckpoint('Mod/best_model_siamese_segmentation.h5', monitor='val_loss', save_best_only=True, verbose=1)
    checkpoint_all = ModelCheckpoint('Mod/model_siamese_segmentation_epoch_{epoch:02d}.h5', save_best_only=False, verbose=1)
    
    # Entrenar modelo
    model.fit(train_generator, steps_per_epoch=25, epochs=epochs, callbacks=[checkpoint_best, checkpoint_all])
    model.save('Mod/siames_org_inst.h5')

# Función de prueba
def probar_modelo(modelo, generador_prueba, batch_size=5):
    X_prueba, [Y_instrumento_real, Y_fondo_real] = next(generador_prueba)
    Y_instrumento_predicho, Y_fondo_predicho = modelo.predict(X_prueba)
    
    plt.figure(figsize=(15, 10))
    for i in range(batch_size):
        plt.subplot(batch_size, 4, 4*i+1)
        plt.imshow(X_prueba[i, ...])
        plt.title("Imagen Original")
        plt.axis('off')
        
        plt.subplot(batch_size, 4, 4*i+2)
        plt.imshow(Y_fondo_predicho[i, :, :, 0], cmap='gray')
        plt.title("Máscara Predicha Fondo")
        plt.axis('off')
        
        plt.subplot(batch_size, 4, 4*i+3)
        plt.imshow(Y_instrumento_predicho[i, :, :, 0], cmap='gray')
        plt.title("Máscara Predicha Instrumento")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Entrenar el modelo si es necesario
    entrenar_modelo(data_dir, image_shape, batch_size, epochs=25, model_path='Mod/model_siamese_segmentation_epoch_20.h5')
    
    # Probar el modelo
    modelo = load_model('Mod/siames_org_inst.h5', custom_objects={'IOU_calc': IOU_calc})
    generador_prueba = generator(data_dir, image_shape, batch_size)
    probar_modelo(modelo, generador_prueba)
