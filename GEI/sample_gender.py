from glob import glob
from os.path import exists, join
from shutil import copyfile
from random import choice
import argparse
import os

def main(base_dir, no_samples,sample_type):


    sample(join(base_dir,"male"),no_samples,join("data",sample_type,"male"),sample_type)

    sample(join(base_dir,"female"),no_samples,join("data",sample_type,"female"),sample_type)


def sample(in_dir, no_samples, out_dir, sample_type):

    if not exists(out_dir):

        os.makedirs(out_dir)
    

    subjects = glob(join(in_dir, "*")) 
    print(subjects)
    variations = ['nm-01', 'nm-02','nm-03','nm-04','nm-05','nm-06']
    
    angles = ['000','018','036','054','072','090','108','126','144','162','180']

    

    for x in range(no_samples):
        empty = True
        print(x)
        while empty:

            empty = False
        
            GEI = join(choice(subjects), choice(variations), choice(angles), "GEI.png")
            
            if not exists(GEI):
                empty = True
                continue
            print(GEI) 
        
        copyfile(GEI, join(out_dir,"%03d.png" % (x + 1)))



if __name__=='__main__':
 
    parser = argparse.ArgumentParser(description='Sample data for gender classification.')
     
    parser.add_argument(
        'base_dir',
        help="Directory that contains the GEIs.")   
    
    parser.add_argument(
        'no_samples',
        type=int,
        help="No of GEIs to sample.")   
         
    parser.add_argument(
        'sample_type',
        help="Sample type: train, val or test.")   
    
    args = parser.parse_args()

    main(args.base_dir, args.no_samples, args.sample_type) 
