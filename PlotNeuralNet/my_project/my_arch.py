
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    #
    to_Conv('conv1', 48, 32, offset="(0,0,0)", to="(0,0,0)", height=48, depth=48, width=32),
    to_Pool('pool1', offset="(0,0,0)", to="(conv1-east)", height=24, depth=24, width=1),
    #
    to_Conv('conv2', 24, 64, offset="(0,0,0)", to="(pool1-east)", height=24, depth=24, width=64),
    # to_connection('pool1', 'conv2'), 
    to_Pool('pool2', offset='(0,0,0)', to='(conv2-east)', height=12, depth=12, width=1),
    #
    #how to plot reshape?
    #
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
