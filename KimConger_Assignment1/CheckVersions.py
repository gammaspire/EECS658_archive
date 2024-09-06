#!/usr/bin/env python
# coding: utf-8

import sys
import scipy
import numpy as np
import pandas as pd
import sklearn

def print_version():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(np.__version__))
    print('pandas: {}'.format(pd.__version__))
    print('sklearn: {}'.format(sklearn.__version__))
    print(' ')
    print('Hello World!')

if __name__ == "__main__":
    print_version()
