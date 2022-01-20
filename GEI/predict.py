import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse

from numpy.random import seed
seed(42)# keras seed fixing
tf.random.set_seed(42)# tensorflow seed fixing

def main(image_file, model):
 
    model = keras.models.load_model(model)  

    img = keras.preprocessing.image.load_img(image_file, target_size = (126,126), color_mode='grayscale')
    img_array = keras.preprocessing.image.img_to_array(img)
    img_exp = np.expand_dims(img_array, axis = 0)

    opt = keras.optimizers.SGD(learning_rate=0.01)#lr_schedule)
    model.compile(optimizer=opt,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

   # test_loss, test_acc = model.evaluate(image, verbose=2)

   # print('\nTest Loss:', test_loss)
   # print('\nTest accuracy:', test_acc) 

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    predictions_single = probability_model.predict(img_exp)

    print("Probabilities:", predictions_single)


    prediction = np.argmax(predictions_single[0])

    if prediction == 0:
        pred_class = "Male"
        prob = predictions_single[0][0]
    else:
        pred_class = "Female"
        prob = predictions_single[0][1]

    print("Prediction: ", prediction, pred_class)
    return pred_class, prob




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Make predictions on gender classifier.')
     
    parser.add_argument(
        'image',
        help="Image to be classified.")   
    
    parser.add_argument(
        '--model',
        help="Path of the model to be tested.")   

    args = parser.parse_args()

    main(args.image, args.model)
