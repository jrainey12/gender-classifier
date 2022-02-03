import argparse
import os
import cv2
from pathlib import Path


def main(base_dir, before):
    """
    Remove the partial silhouettes that are at the edge of the frame.
    Ensures only complete sils are present for later processing.
    Args: 
        base_dir - Directory tree of silhouette sequences.
        before - Bool flag for edge removal before alignment.
    """
    indir = Path(base_dir)

    subjects = list(indir.glob("*"))
   
    lowest_dirs = []
    
    #get all folders containing sil sequences 
    for root,dirs,files in os.walk(indir):
        if files and not dirs:
            lowest_dirs.append(root)
            

    print ("lowest_dirs: ", lowest_dirs)

    edge_sils = []
    #Find edge sils
    for d in lowest_dirs:
        edge_sils += find_edge_sils(d,before)
        #print(edge_sils) 
        
    #delete edge sils
    for sil in edge_sils:
        print (sil)
        if os.path.exists(sil):
            os.remove(sil)
        
def find_edge_sils(sil_dir,before):
    """
    Find the silhouettes at the edge of the frame.
    Args:
        sil_dir - Directory of silhouettes.
        before - flag for finding edges before alignment. 
    Return: edge_sils - list of file paths of edge silhouettes.
    """

    edge_sils = []
    for sil in Path(sil_dir).glob("*"):
        #print(sil)
        img = cv2.imread(str(sil),0)
        
        height,width = img.shape
        
        #find any white pixels in far left column of pixels
        #(pixel idx 1 is used instead of 0 for width as the left edge on 
        #the sil data has a 1 pixel gap for some reason.)
        for p in range(0,height-1):
            if img.item(p,1) == 255:
                print ("Left Edge: ", sil)
                edge_sils.append(sil)
                break

        #find any white pixel in far right column of pixels
        for p in range(0,height-1):
            if img.item(p,width-2) == 255:
                print ("Right Edge: ", sil)
                edge_sils.append(sil)
                break

    #Only look at top and bottom before alignments.
    #Doing so after aligment will remove all frames.

    if before:
        #find any white pixel in top row of pixels
        for p in range(0,width-1):
            if img.item(1,p) == 255:
                print ("Top Edge: ", sil)
                edge_sils.append(sil)
                break
        
        #find any white pixel in bottom row of pixels
        for p in range(0,width-1):
            if img.item(height-2,p) == 255:
                print ("Bottom Edge: ", sil)
                edge_sils.append(sil)
                break
        

    return edge_sils

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Find and remove silhouettes at the edge of a frame.')
     
    parser.add_argument(
        'base_dir',
        help="Base directory tree containing gait silhouette sequences.")   
   
    parser.add_argument(
        'before',
        choices=('True','False'),
        default='False',
        help="Before or after alignment.")

    args = parser.parse_args()

    main(args.base_dir,args.before)
 
