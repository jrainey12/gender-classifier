import argparse
import os
from pathlib import Path
import cv2

def main(base_dir):

    """
    Find the keyframes from a sequence of aligned gait silhouettes.
    A keyframe occurs at minimum and maximum stride lengths, in the middle of each phase of a gait cycle.

    Args:
        base_dir - Directory tree containing aligned gait silhouette sequences in the lowest directories.
        out_dir - Directory to save final keyframes (input dir tree layout will be recreated).
    """

    indir = Path(base_dir)

    subjects = list(indir.glob("*"))
   
    lowest_dirs = []
    
    #get all folders containing sil sequences 
    for root,dirs,files in os.walk(indir):
        if files and not dirs:
            lowest_dirs.append(root)

    for d in lowest_dirs:
        find_keyframes(d) 


def find_keyframes(sil_dir):
    """
    Find the keyframes using the stride lengths in each frame.
    
    Args;
        sil_dir - Directory containing silhouettes.
    """
    
    sil_dir = Path(sil_dir)
    
    sils = sorted(list(sil_dir.glob("*")))
    

    #get all stride widths
    stride_widths = []

    for sil in sils:
        stride_widths.append(get_stride_width(sil))
        

    print(stride_widths)
   #find min and max strides 
    #minima = find_minima(stride_widths)
    #maxima = find_maxima(stride_widths)


def find_minima(stride_widths):
    """
    Find minima in a sequence.
    
    Args:
        stride_widths - width of stride for each frame.
    """
   
    #TODO: Fix this method to correctly find all minimum values.
    #May need to be a recursive function to work correctly.
    
    minima = []
    mn = 1000
    for i,s in enumerate(stride_widths):
        
        if i == 0 and s < mn:
            mn = s
            continue

        elif s >= mn:
            minima.append([i,s])
            mn = 10000
    

def get_stride_width(frame):
    """
    Find the stride width in pixels.

    Args:
        frame - path of the frame to use.
    """

    img = cv2.imread(str(frame),0)
    height, width = img.shape
    
    #Slightly above bottom to cover all angles 
    bot = height - 5
    left = None
    right = None


    #print ("Height: ", height)
    #print ("Width: ", width)
    #print ("Bottom: ", bot)

    #Find leftmost white pixel
    for p in range(0,width-1):
        if left == None:
            if img.item(bot,p) == 255:
                left = p
                print ("Left: ", left)
                break

    #find rightmost white pixel
    for p in range(width-1,0,-1):
        if right == None:
            if img.item(bot,p) == 255:
                right = p
                print ("Right: ", right)
                break
    
    #calc width of stride    
    stride_width = right - left

    return stride_width


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Find keyframes from gait silhouette sequences.')
     
    parser.add_argument(
        'base_dir',
        help="Base directory tree containing gait silhouette sequences.")   
    
    args = parser.parse_args()

    main(args.base_dir)
    
