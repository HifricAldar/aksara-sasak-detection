import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ModelTrainer:
    @staticmethod
    def create_model(input_shape=(224, 224, 3), num_classes = 18):
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.5),

            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),

            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    @staticmethod
    def train(model, train_dir, val_dir, epochs=20):
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            train_dir, target_size=(224,224), batch_size=32, class_mode='categorical'
        )
        
        val_gen = val_datagen.flow_from_directory(
            val_dir, target_size=(224,224), batch_size=32, class_mode='categorical'
        )
        
        history = model.fit(
            train_gen, validation_data=val_gen, epochs=epochs, verbose=2
        )
        
        return history, train_gen.class_indices