import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from End_to_End_DL_Project import logger
from End_to_End_DL_Project.config.configuration import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        # Changed from VGG16 to DenseNet121
        self.model = tf.keras.applications.DenseNet121(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, unfreeze_deep, learning_rate):
        # Fine-tuning logic: Unfreeze only the final dense block ('conv5') if True
        if unfreeze_deep:
            print("Fine-tuning enabled: Unfreezing DenseNet Block 5...")
            for layer in model.layers:
                # 'conv5' targets the final dense block in DenseNet121
                if 'conv5' in layer.name: 
                    layer.trainable = True
                else:
                    layer.trainable = False
        else:
            print("Feature Extraction only: Freezing entire backbone.")
            for layer in model.layers:
                layer.trainable = False

        # Adding custom classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        
        # Optional but highly recommended: Add a Dropout layer before Dense to prevent overfitting
        x = tf.keras.layers.Dropout(0.5)(x)
        
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
            # --- NEW: Explicit Weight Initialization ---
            kernel_initializer=tf.keras.initializers.GlorotUniform() 
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Using Adam optimizer with the lowered learning rate for safe fine-tuning
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            unfreeze_deep=self.config.params_unfreeze_deep, # Passes the new strategy
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)