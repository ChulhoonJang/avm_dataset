# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:16:42 2017

@author: SRG
"""

from __future__ import print_function

import argparse

BATCH_SIZE = 10

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    print('parsing')
    return parser.parse_args()
                        
def main():
    """Create the model and start the training."""
    args = get_arguments()
    print(args.batch_size)
    
if __name__ == '__main__':
    main()