import pandas as pd
import json
import os
import shutil

class DataProcessor:
    @staticmethod
    def load_coco_data(json_path):
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        images_df = pd.DataFrame(coco_data['images'])
        annotations_df = pd.DataFrame(coco_data['annotations'])
        categories_df = pd.DataFrame(coco_data['categories'])
        
        # Merge data
        annotations_with_cat = pd.merge(annotations_df, categories_df, 
                                       left_on='category_id', right_on='id')
        full_data = pd.merge(images_df, annotations_with_cat, 
                            left_on='id', right_on='image_id')
        
        return full_data[['file_name', 'name']]
    
    @staticmethod
    def prepare_folders(df, images_dir, output_dir, dataset_type):
        dataset_path = os.path.join(output_dir, dataset_type)
        os.makedirs(dataset_path, exist_ok=True)
        
        for category in df['name'].unique():
            os.makedirs(os.path.join(dataset_path, category), exist_ok=True)
        
        copied = 0
        for _, row in df.iterrows():
            src = os.path.join(images_dir, row['file_name'])
            dst = os.path.join(dataset_path, row['name'], row['file_name'])
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
        
        print(f" {dataset_type}: {copied} images")
        return copied