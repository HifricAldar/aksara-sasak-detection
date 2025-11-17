import os
import sys
import json

# Add current directory to path untuk import module lain di src
sys.path.append(os.path.dirname(__file__))

from data_checker import DataChecker
from data_loader import DataProcessor
from model_trainer import ModelTrainer
from config import *

def main():
    
    checker = DataChecker()
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # 1. Cek apakah model sudah ada
    if checker.check_model_exists():
        print("Model sudah ada. Skip training.")
        return
    
    # 2. Cek apakah data sudah diproses
    if checker.check_data_exists():
        print("Data sudah diproses. Langsung training...")
    else:
        print("Data belum diproses. Processing data...")
        
        # Cek data mentah tersedia - path RELATIF dari root project
        dataset_paths = {
            'train': {
                'json': "data/raw/Deteksi Tulisan Aksara.v2-dataset_aksara_sasak.coco/train/_annotations.coco.json",
                'images': "data/raw/Deteksi Tulisan Aksara.v2-dataset_aksara_sasak.coco/train"
            },
            'valid': {
                'json': "data/raw/Deteksi Tulisan Aksara.v2-dataset_aksara_sasak.coco/valid/_annotations.coco.json",
                'images': "data/raw/Deteksi Tulisan Aksara.v2-dataset_aksara_sasak.coco/valid"
            },
            'test': {
                'json': "data/raw/Deteksi Tulisan Aksara.v2-dataset_aksara_sasak.coco/test/_annotations.coco.json", 
                'images': "data/raw/Deteksi Tulisan Aksara.v2-dataset_aksara_sasak.coco/test"
            }
        }
        json_paths = [data['json'] for data in dataset_paths.values()]
        image_dirs = [data['images'] for data in dataset_paths.values()]

        raw_available, missing = checker.check_raw_data(json_paths, image_dirs)
        if not raw_available:
            print(f"Data mentah tidak lengkap: {missing}")
            return
        
        # Process data
        print("Memproses dataset...")
        
        for dataset_type, paths in dataset_paths.items():
            df = processor.load_coco_data(paths['json'])
            processor.prepare_folders(df, paths['images'], OUTPUT_DIR, dataset_type)
            print(f"{dataset_type}: {len(df)} images processed")

        NUM_CLASSES = len(df['name'].unique())

    # 3. Training model
    print("Training model...")
    model = trainer.create_model(
        input_shape=(*IMG_SIZE, 3),
        num_classes=NUM_CLASSES
    )
    history, class_indices = trainer.train(
        model, 
        TRAIN_DIR, 
        VAL_DIR,   
        epochs=EPOCHS
    )
    
    # 4. Save model - relative path ke folder models di root
    os.makedirs(MODELS_DIR, exist_ok=True) 
    model.save(MODEL_FILE)

    with open(JSON_FILE, 'w') as f:
        json.dump(class_indices, f)
    
    print(f"🎯 Model trained for {NUM_CLASSES} classes")
    print("PIPELINE SELESAI!")

if __name__ == "__main__":
    main()