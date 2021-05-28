
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    ## (conv1 + pool1)
    to_Conv('conv1', 48, 32, offset="(0,0,0)", to="(0,0,0)", height=48, depth=48, width=32, caption='conv1'),
    to_Pool('pool1', offset="(0,0,0)", to="(conv1-east)", height=24, depth=24, width=1),
    ## (conv2 + pool2)
    to_Conv('conv2', 24, 64, offset="(1,0,0)", to="(pool1-east)", height=24, depth=24, width=64, caption='conv2'),
    to_Pool('pool2', offset='(0,0,0)', to='(conv2-east)', height=12, depth=12, width=1),
    ## <op reshape> -> (fc1 + fc2)
    to_FullyConnected('fc1', 128, offset='(0.75,0,0)', to='(pool2-east)', width=1, height=1, depth=128, caption='fc1'),
    to_FullyConnected('fc2', 5, offset='(0.75,0,0)', to='(fc1-east)', width=1, height=1, depth=5, caption='fc2'),
    ## soft max
    to_SoftMax('soft1', 5, offset='(0.50,0,0)', to='(fc2-east)', width=1, height=1, depth=5, caption='SOFT'),
    ## draw connections
    to_connection('pool1', 'conv2'),
    to_connection('pool2', 'fc1'),
    to_connection('fc1', 'fc2'),
    to_connection('fc2', 'soft1'),
    #
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
