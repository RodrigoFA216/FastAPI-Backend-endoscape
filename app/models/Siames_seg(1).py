# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 22:44:45 2024

@author: rodar
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.transform import resize
from imageio import imread

# TensorFlow y Keras imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Add, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# Definición de la ruta de la carpeta donde están las imágenes
data_dir = 'Imagenes/Endoscape/'
image_shape = (128, 128)
batch_size = 64

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

                #mask_path = os.path.join(data_folder, 'Labeli', f'{filename}.png')
                #mask_background = imread(mask_path)
                mask_background = 1 - mask_instrument
                mask_background = resize(mask_background, image_shape, mode='reflect')
                mask_background = (mask_background > 0.5).astype(np.float32)
                
                mask_instrument = np.expand_dims(mask_instrument, axis=-1)
                mask_background = np.expand_dims(mask_background, axis=-1)

                images.append(image)
                mask_instruments.append(mask_instrument)
                mask_backgrounds.append(mask_background)

            yield np.array(images), [np.array(mask_instruments), np.array(mask_backgrounds)]

def IOU_calc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1))
    return intersection / union

def residual_block(x, filters):
    # Ajustar el shortcut para que tenga el mismo número de filtros
    shortcut = Conv2D(filters, (1, 1), padding='same')(x)  # Convolución 1x1 para ajustar dimensiones
    conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    return Add()([conv1, shortcut])  # Sumar el shortcut ajustado


def siamese_segmentation_model(img_shape):
    inputs = Input(img_shape)

    def shared_block(x):
        # Reducción en el número de capas y eliminación de Inception
        conv1 = residual_block(x, 64)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = residual_block(pool1, 128)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = residual_block(pool2, 256)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = residual_block(pool3, 512)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        return conv1, conv2, conv3, conv4, pool4

    def branch(conv1, conv2, conv3, conv4, shared, output_filters=1):
        up6 = UpSampling2D(size=(2, 2))(shared)
        up6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
        merge6 = concatenate([conv4, up6], axis=3)
        up7 = UpSampling2D(size=(2, 2))(merge6)
        up7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
        merge7 = concatenate([conv3, up7], axis=3)
        up8 = UpSampling2D(size=(2, 2))(merge7)
        up8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        merge8 = concatenate([conv2, up8], axis=3)
        up9 = UpSampling2D(size=(2, 2))(merge8)
        up9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        merge9 = concatenate([conv1, up9], axis=3)
        return Conv2D(output_filters, (1, 1), activation='sigmoid')(merge9)
    
    def branch2(conv1, conv2, conv3, conv4, shared, output_filters=1):
        up6 = UpSampling2D(size=(2, 2))(shared)
        up6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
        merge6 = concatenate([conv4, up6], axis=3)
        up7 = UpSampling2D(size=(2, 2))(merge6)
        up7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
        merge7 = concatenate([conv3, up7], axis=3)
        up8 = UpSampling2D(size=(2, 2))(merge7)
        up8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        merge8 = concatenate([conv2, up8], axis=3)
        up9 = UpSampling2D(size=(2, 2))(merge8)
        up9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        merge9 = concatenate([conv1, up9], axis=3)
        return Conv2D(output_filters, (1, 1), activation='sigmoid')(merge9)

    conv1, conv2, conv3, conv4, shared_output = shared_block(inputs)
    output_instrument = branch(conv1, conv2, conv3, conv4, shared_output)
    output_background = branch2(conv1, conv2, conv3, conv4, shared_output)

    return Model(inputs=inputs, outputs=[output_instrument, output_background])


if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import ModelCheckpoint
    
    # Cargar el modelo previamente guardado
    model_path = 'Mod/model_siamese_segmentation_epoch_20.h5'
    
    try:
        model = load_model(model_path, custom_objects={'IOU_calc': IOU_calc})  # Asegúrate de incluir cualquier métrica o función personalizada
        print(f'Modelo cargado desde {model_path}')
    except:
        print(f'No se pudo cargar el modelo desde {model_path}. Entrenando un nuevo modelo...')
        model = siamese_segmentation_model((128, 128, 3))
        model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=[IOU_calc])
    
    
    
    # Generador de datos
    train_generator = generator(data_dir, image_shape, batch_size)
    


    # Callback para guardar el mejor modelo basado en la métrica que elijas (por ejemplo, val_loss)
    checkpoint_best = ModelCheckpoint(
        'Mod/best_model_siamese_segmentation.h5',
        monitor='val_loss',  # Ajusta esto según la métrica que quieras monitorear
        save_best_only=True,
        verbose=1
    )
    
    # Callback para guardar el modelo en cada época
    checkpoint_all = ModelCheckpoint(
        'Mod/model_siamese_segmentation_epoch_{epoch:02d}.h5',  # Guarda con un nombre diferente por época
        save_best_only=False,
        verbose=1
    )
    
    # Entrenamiento del modelo con ambos callbacks
    model.fit(
        train_generator,
        steps_per_epoch=25 ,
        epochs=25,  # Puedes ajustar la cantidad de épocas si deseas continuar por más tiempo
        callbacks=[checkpoint_best, checkpoint_all]
    )
    
    # Guardar el modelo al final del entrenamiento
    model.save('Mod/siames_org_inst.h5')

    
    
    
    def ver_segmentacion(modelo, generador_prueba, batch_size=5):
        # Obtener un lote de datos de prueba
        X_prueba, [Y_instrumento_real, Y_fondo_real] = next(generador_prueba)
        
        # Realizar la predicción sobre el lote de imágenes
        Y_instrumento_predicho, Y_fondo_predicho = modelo.predict(X_prueba)
        
        # Visualizar resultados
        plt.figure(figsize=(15, 10))
        
        for i in range(batch_size):
            # Mostrar imagen original
            plt.subplot(batch_size, 4, 4*i+1)
            plt.imshow(X_prueba[i, ...])  # Imagen original
            plt.title("Imagen Original")
            plt.axis('off')
            
            # Mostrar máscara real del instrumento
            plt.subplot(batch_size, 4, 4*i+2)
            plt.imshow(Y_fondo_predicho[i, :, :, 0], cmap='gray')
            plt.title("Máscara Predicha Fondo")
            plt.axis('off')
            
            # Mostrar máscara predicha del instrumento
            plt.subplot(batch_size, 4, 4*i+3)
            plt.imshow(Y_instrumento_predicho[i, :, :, 0], cmap='gray')
            plt.title("Máscara Predicha Instrumento")
            plt.axis('off')
            
            
        
        plt.tight_layout()
        plt.show()
    
    # Llamar a la función para ver las segmentaciones en un lote de prueba
    ver_segmentacion(model, train_generator)
    
    
    
