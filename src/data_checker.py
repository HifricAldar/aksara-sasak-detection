import os
import pandas as pd
from config import *

class DataChecker:
    @staticmethod
    def check_model_exists(model_path=MODEL_FILE):
        return os.path.exists(model_path)
    
    @staticmethod
    def check_data_exists(data_dir=OUTPUT_DIR):
        required_folders = ['train', 'valid', 'test']
        for folder in required_folders:
            if not os.path.exists(os.path.join(data_dir, folder)):
                return False
        return True
    
    @staticmethod
    def check_raw_data(json_paths, image_dirs):
        missing = []
        
        for path in json_paths:
            if not os.path.exists(path):
                missing.append(f"JSON: {path}")
        
        # Cek image directories
        for dir_path in image_dirs:
            if not os.path.exists(dir_path):
                missing.append(f"Images: {dir_path}")
        
        return len(missing) == 0, missing