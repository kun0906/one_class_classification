# -*- coding: utf-8 -*-
"""
    add 'parent path' to system path so that the script can call 'parent directory'
"""
import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath('../'))  # add 'parent path' to system path
print(sys.path)
