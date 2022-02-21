import tensorflow as tf
from gender_model import GenderModel
import argparse

def main(input_type, mode, model_name, lr, bs, epochs):
    """
    Wrapper with augmenetation and parameters for GEI training.
    Args:
        input_type(GEI,single,keyframe) - Type of input used.
        mode(train,test) - Execution mode.
        model_name(str) - Name of model to train or test.
        lr(float) - Initial learning rate for training.
        bs(int) - Size of batches to use.
        epochs(int) - Number of epochs to train for.
    """

    #Set input_dir, augs and lr_schedule for selected input
    if input_type == "GEI":

        input_dir = "GEI/GEI_data/sampled"
        
        augs = dict(rescale=1/255.,
                rotation_range=10,
                #width_shift_range=0.05,
                #height_shift_range=0.05,
                #shear_range=0.05,
                #zoom_range=0.05,
                #horizontal_flip=True
                )
        

        #learning rate schedule, list of epochs to drop at.(factor of 10 be default)
        lr_schedule = [60,100]


    elif input_type == "single":
        
        input_dir = "single_frame/single_frame_data/sampled"

        augs = dict(rescale=1/255.
                )
        
        lr_schedule = None
    
    elif input_type == "keyframe":

        input_dir = "keyframes/key_frame_data/sampled"

        augs = dict(rescale=1/255.,
                )

        lr_schedule = None



    #Initialise model
    GM = GenderModel(input_dir, model_name)
    
    
    if mode=="train":
        #Train then test model
        GM.train(lr, bs, epochs, augs, lr_schedule)

        GM.test(bs)

    elif mode=="test":
        #Test model
        GM.test(bs)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Wrapper for GEI training and testing.")

    parser.add_argument(
            'input_type',
            choices=('GEI','single','keyframe'),
            help="Type of input to use: GEI, single or keyframe.")

    parser.add_argument(
            'mode',
            choices=('train','test'),
            help="Execution mode: train or test.")

    parser.add_argument(
            'model',
            type=str,
            help="Name of the new model or Path of the model to be tested.")

    parser.add_argument(
            '-lr',
            type=float,
            default=0.0001,
            help="Initial Learning rate for training.")

    parser.add_argument(
            '-bs',
            type=int,
            default=64,
            help="Number of batches to be used.")

    parser.add_argument(
            '--epochs',
            type=int,
            default=1,
            help="Number of epochs to train for.")


    args = parser.parse_args()

    main(args.input_type,args.mode,args.model,args.lr,args.bs,args.epochs)

