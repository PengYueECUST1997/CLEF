import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))


src_path = os.path.join(project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
