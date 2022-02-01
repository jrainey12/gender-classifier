import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy.random import seed

seed(42)# keras seed fixing
tf.random.set_seed(42)# tensorflow seed fixing

def main(mode, model_name, learning_rate, batch_size, epochs):

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    print("CUDA?: ", tf.test.is_built_with_cuda())

    if mode == "train":
        train(model_name,learning_rate,batch_size,epochs)

    elif mode == "test":
        test(model_name,batch_size)

    else:
        print("Invalid mode selected.")

def train(model_name,learning_rate,batch_size,epochs):


    train_ds = tf.keras.preprocessing.image_dataset_from_directory("keyframe_data/sampled/train", labels='inferred', label_mode='binary', class_names=['male','female'], color_mode='grayscale', batch_size=batch_size, image_size=(126,126))

    val_ds = tf.keras.preprocessing.image_dataset_from_directory("keyframe_data/sampled/val", labels='inferred', label_mode='binary', class_names=['male','female'], color_mode='grayscale', batch_size=batch_size, image_size=(126,126))
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    #train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)#.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #train_ds_size = tf.data.experimental.cardinality(train_ds)*batch_size
    #val_ds_size = tf.data.experimental.cardinality(val_ds)*batch_size#tf.keras.backend.get_value

    #print(train_ds_size)
    #print(val_ds_size)

    train_ds = train_ds.cache()
    #train_ds = train_ds.shuffle(train_ds_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache()
    #val_ds = val_ds.shuffle(val_ds_size)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

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
  
    #decay_st = tf.keras.backend.get_value(tf.data.experimental.cardinality(train_ds))*15

    #print("LR Start: ", learning_rate)
    #print("LR Decay Steps: ", decay_st)

    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate=learning_rate,
    #decay_steps=decay_st,
    #decay_rate=0.1,
    #staircase=True) 
   
    opt = keras.optimizers.SGD(learning_rate=learning_rate)#lr_schedule)    
    model.compile(optimizer=opt,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    model.summary()

    fit_model = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    vis_acc_loss(fit_model,model_name,epochs)  

    model.save(model_name)  


def test(model,batch_size):
 
    test_ds = tf.keras.preprocessing.image_dataset_from_directory("keyframe_data/sampled/test", labels='inferred', label_mode='binary', class_names=['male','female'], color_mode='grayscale', batch_size=batch_size, image_size=(126,126))

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
   
    opt = keras.optimizers.SGD(learning_rate=0.01)#lr_schedule)
    model = keras.models.load_model(model)   
    model.compile(optimizer=opt,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    print('\nTest Loss:', test_loss)
    print('\nTest accuracy:', test_acc) 

    #probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    #predictions = probability_model.predict(test_ds)

    #print(predictions) 

  

def vis_acc_loss(fit_model,model_name,epochs):

    acc = fit_model.history['accuracy']
    val_acc = fit_model.history['val_accuracy']

    loss=fit_model.history['loss']
    val_loss=fit_model.history['val_loss']

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
    #plt.show()
    plt.savefig(model_name + "/acc_loss_graph.png")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train or test gender classifier.')
     
    parser.add_argument(
        'mode',
        choices=('train','test'),
        default='test',
        nargs='?',
        help="Execution mode: train or test. Default: test.")   
    
    parser.add_argument(
        '--model',
        type=str,
        help="Name of new model or Path of the model to be tested.")   

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help="Initial Learning rate for training.")

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help="Batch size for training and testing.")

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="Number of epochs for training.")

    args = parser.parse_args()

    main(args.mode, args.model, args.learning_rate, args.batch_size, args.epochs)
