from shutil import copyfile
from random import seed,choice
import random
import argparse
import os
from pathlib import Path


def main(base_dir, no_samples,sample_type, out_dir):
    """
    Sample a dataset and save the samples in a new folder.
    Args:
        base_dir(str): Directory containing data to be sampled.
        no_samples(int): Number of samples to save.
        sample_type(train,test,val): Type of data being sampled.
        out_dir(str): Directory to save sampled data.
    """

    #Set up male and female input paths
    b = Path(base_dir)
    m_in_dir = b / "male"
    f_in_dir = b / "female"

    #Set up male and female output paths
    o = Path(out_dir)
    m_out_dir = o / sample_type / "male"
    f_out_dir = o / sample_type / "female"

    #Sample for male and female
    sample(str(m_in_dir),no_samples,str(m_out_dir))
    sample(str(f_in_dir),no_samples,str(f_out_dir))


def sample(in_dir, no_samples, out_dir):
    """
    Randomly select a sample from the input data.
    Args:
        in_dir(str): Directory containing data to be sampled.
        no_samples(str): Number of samples to save.
        out_dir(str): Directory to save samples.
    """

    #TODO: Add a method of selecting/ignoring some varitions or angles.
    # e.g only selecting nm-01 to nm-04 and ignoring nm-05 and nm-06.
    # This would prevent having to move files.
    # Possibly after getting sils I could remove any containing selected strings.

    #TODO: Fix Path based approach, currently uses both Path and os, should use 
    # either one or the other.

    #Set seed to sample size to ensure data is consistent when
    # same size dataset is sampled on a set of data.
    random.seed(no_samples)

    out = Path(out_dir)
    if not out.exists():
        os.makedirs(out)
    
    indir = Path(in_dir)

    subjects = list(indir.glob("*"))
   
    lowest_dirs = []
    sils = []
    #get all files in dataset to be sampled 
    for root,dirs,files in os.walk(in_dir):
        if files and not dirs:
            for f in files:
                sils.append(root + "/"+ f)

    print(len(sils))

    #get samples
    samples = random.sample(sils, no_samples)
    #print(len(samples))

    #Copy samples to out directory
    for i,s in enumerate(samples):
        #print (i, s)
        outname = out / str("%03d.png" % (i + 1))
        copyfile(s, outname)


if __name__=='__main__':
 
    parser = argparse.ArgumentParser(description='Sample data for gender classification.')
     
    parser.add_argument(
        'base_dir',
        help="Directory that contains the data to be sampled.")   
    
    parser.add_argument(
        'no_samples',
        type=int,
        help="No of samples to save.")   
         
    parser.add_argument(
        'sample_type',
        help="Sample type: train, val or test.")   
    
    parser.add_argument(
        'out_dir',
        help="Directory to save samples.")

    args = parser.parse_args()

    main(args.base_dir, args.no_samples, args.sample_type, args.out_dir) 
