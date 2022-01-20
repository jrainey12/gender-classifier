import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse

from numpy.random import seed
seed(42)# keras seed fixing
tf.random.set_seed(42)# tensorflow seed fixing


def main(image_file, model_name):
 
    model = keras.models.load_model(model_name)  

    img = keras.preprocessing.image.load_img(image_file, target_size = (126,126), color_mode='grayscale')
    img_array = keras.preprocessing.image.img_to_array(img)
    img_exp = np.expand_dims(img_array, axis = 0)


    opt = keras.optimizers.SGD(learning_rate=0.01)#lr_schedule)
    model.compile(optimizer=opt,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    predictions_single = probability_model(img_exp)


    print("Probabilities:", predictions_single)


    prediction = np.argmax(predictions_single[0])

    if prediction == 0:
        pred_class = "Male"
        prob = predictions_single[0][0]
    else:
        pred_class = "Female"
        prob = predictions_single[0][1]

    print("Prediction: ", prediction, pred_class)
    return pred_class, tf.keras.backend.get_value(prob)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train or test gender classifier.')
     
    parser.add_argument(
        'image',
        help="Image to be classified.")   
    
    parser.add_argument(
        '--model',
        help="Path of the model to be tested.")   

    args = parser.parse_args()

    main(args.image, args.model)
