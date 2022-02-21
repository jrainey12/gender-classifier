import tensorflow as tf
from tensorflow.keras import layers
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os 
from numpy.random import seed

class GenderModel():
    
    def __init__(self, input_dir, model_name):

        seed(42)
        tf.random.set_seed(42)

        self.input_dir = input_dir
        self.model_name = model_name

        self.model = tf.keras.Sequential([
                layers.Conv2D(16, 3, activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(2)
                ])
    

    def train(self, lr, bs, epochs, augs,LR_schedule):

        #Augment train data
        train_datagen = ImageDataGenerator(**augs)

        #Load train data
        train_ds = train_datagen.flow_from_directory(
                    self.input_dir + "/train",
                    target_size=(224,224),
                    batch_size=bs,
                    color_mode='grayscale',
                    shuffle=True,
                    class_mode='binary')

        
        #Rescale val data
        val_datagen = ImageDataGenerator(rescale=1/255.)

        #Load val data
        val_ds = val_datagen.flow_from_directory(
                    self.input_dir + "/val",
                    target_size=(224,224),
                    batch_size=bs,
                    color_mode='grayscale',
                    shuffle=True,
                    class_mode='binary')


        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        self.model.compile(optimizer=opt,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

        self.model.build([None,224,224,1])

        self.model.summary()

        #Tensorboard output
        log_dir = os.path.join(self.model_name.split('/')[0], "runs", os.path.basename(self.model_name))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


        #Train model
        self.model.fit(train_ds,
                       validation_data=val_ds,
                       epochs=epochs,
                       callbacks=[tensorboard_callback])

        #Save model
        self.model.save(self.model_name)


    def test(self, bs):

        #Rescale test data
        test_datagen = ImageDataGenerator(rescale=1/255.)

        #Load test data
        test_ds = test_datagen.flow_from_directory(
                    self.input_dir + "/test",
                    target_size=(224,224),
                    batch_size=bs,
                    color_mode='grayscale',
                    shuffle=True,
                    class_mode='binary')

        #Load model from model_name .
        model = tf.keras.models.load_model(self.model_name) 

        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

        model.compile(optimizer=opt,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

        test_loss, test_acc = model.evaluate(test_ds, verbose=1)
        
        print('\nTest Loss:', test_loss)
        print('\nTest Accuracy:', test_acc)

