import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy.random import seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

seed(42)# keras seed fixing
tf.random.set_seed(42)# tensorflow seed fixing

def main(mode, model_name, learning_rate, batch_size, epochs):
    """
    Train or test CNN for use with keyframes.
    Args:
        mode(train,test) - mode to run in.
        model_name(str) - Name for model to be saved or loaded for testing.
        learning_rate(float) - Starting learning rate.
        batch_size(int) - Size of batches.
        epochs(int) - Number of epochs to train for.
    """
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    print("CUDA?: ", tf.test.is_built_with_cuda())


    #Make directory for new model. 
    if not os.path.exists(model_name):
        os.mkdir(model_name)


    if mode == "train":
        train(model_name,learning_rate,batch_size,epochs)

    elif mode == "test":
        test(model_name,batch_size)

    else:
        print("Invalid mode selected.")


def train(model_name,learning_rate,batch_size,epochs):
    """
    Train CNN.
    Args:
        model_name(str) - Name for the model to be saved as.
        learning_rate(float) - Starting learning rate.
        batch_size(int) - Size of batches to use.
        epochs(int) - Number of epochs to train for.
    """


    tf.data.AUTOTUNE
    
    #Generate augmented data for training.
    train_datagen = ImageDataGenerator(rotation_range=10,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                rescale=1/255.,
                                fill_mode='nearest')
    
    #Load train data
    train_ds = train_datagen.flow_from_directory(
                "key_frame_data/sampled/train",
                target_size=(224, 224),
                batch_size=batch_size,
                color_mode='grayscale',
                shuffle = True,
                class_mode='binary')

    #Rescale val data.
    val_datagen = ImageDataGenerator(rescale=1/255.) 
    
    #Load val data.
    val_ds = val_datagen.flow_from_directory(
                "key_frame_data/sampled/val",
                target_size=(224, 224),
                batch_size=batch_size,
                color_mode='grayscale',
                shuffle = True,
                class_mode='binary')

    
    #Model definition.
    model = tf.keras.Sequential([
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(2)
    ])
  
    
    #LR scheduler to drop LR at set rate or at set iters
    epoch_iters = len(train_ds)
    print("LR Start: ", learning_rate)
    print("Iterations per epoch: ", epoch_iters)

    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate=learning_rate,
    #decay_steps=epoch_iters*25,
    #decay_rate=0.1,
    #staircase=True) 
  
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries = [epoch_iters*10],#,epoch_iters*40],
            values = [learning_rate, learning_rate/10])#, learning_rate/100])


    #SGD Optimizer
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=0.9)    
    model.compile(optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    model.build([None,224,224,1])

    model.summary()

    #Fit the model
    fit_model = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    #Visualise the accuracy and loss
    vis_acc_loss(fit_model,model_name,epochs)  

    #Save model
    model.save(model_name)  


def test(model,batch_size):
    """
    Test a previously trained model.
    Args:
        model(str) - Name of the model to test.
        batch_size(str) - Size of batch to use.
    """
    #Rescale test data
    test_datagen = ImageDataGenerator(rescale=1/255.,) 
    
    #Load test data
    test_ds = test_datagen.flow_from_directory(
                "key_frame_data/sampled/test",
                target_size=(224, 224),
                batch_size=batch_size,
                color_mode='grayscale',
                shuffle = True,
                class_mode='binary')


    #SGD optimiser
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    model = tf.keras.models.load_model(model)   
    model.compile(optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    #Evaluate model
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    print('\nTest Loss:', test_loss)
    print('\nTest accuracy:', test_acc) 
 
    #Get Predictions on test set.
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_ds)
    print(predictions)


def vis_acc_loss(fit_model,model_name,epochs):
    """
    Visualise the accuracy and loss using matplotlib.
    (Will be updated to use tensorboard at some point)
    Args:
        fit_model - Model that is being visualised.
        model_name(str) - Name of model.
        epochs(int) - Number of epochs.
    """
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
