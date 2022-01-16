import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

def main():

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA?: ", tf.test.is_built_with_cuda())

    train_ds = tf.keras.preprocessing.image_dataset_from_directory("data/train", labels='inferred', label_mode='binary', class_names=['male','female'], color_mode='grayscale', batch_size=100, image_size=(126,126))

    val_ds = tf.keras.preprocessing.image_dataset_from_directory("data/val", labels='inferred', label_mode='binary', class_names=['male','female'], color_mode='grayscale', batch_size=40, image_size=(126,126))
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory("data/test", labels='inferred', label_mode='binary', class_names=['male','female'], color_mode='grayscale', batch_size=40, image_size=(126,126))


    #print(train_ds.class_names)
    
    #for image_batch, labels_batch in train_ds:
    #    print(image_batch.shape)
    #    print(labels_batch.shape)


    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

       

    model = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(126,126, 1)),
        keras.layers.Conv2D(16, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(2)
    ])
   
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=75000,
    decay_rate=0.9) 
    
    opt = keras.optimizers.SGD(learning_rate=0.01)#lr_schedule)
    model.compile(optimizer=opt,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    model.summary()

    epochs=14
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    vis_acc_loss(history,epochs)    
 
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    print('\nTest accuracy:', test_acc) 

    #probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    #predictions = probability_model.predict(test_ds)

    #print(predictions) 

    model.save("gender_model")

def vis_acc_loss(history,epochs):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()



if __name__ == '__main__':

    main()
