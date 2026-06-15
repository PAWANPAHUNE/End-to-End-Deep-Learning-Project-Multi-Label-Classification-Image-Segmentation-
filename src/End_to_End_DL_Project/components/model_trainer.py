from End_to_End_DL_Project import logger
from End_to_End_DL_Project.config.configuration import TrainingConfig
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False 
        )
        
        # ADDED: Precision and Recall for robust medical evaluation
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical" 
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        valid_dir = os.path.join(self.config.training_data, "valid")

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=valid_dir,
            shuffle=False, 
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
            
        train_dir = os.path.join(self.config.training_data, "train")

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=train_dir,
            shuffle=True, 
            **dataflow_kwargs
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    # --- NEW FUNCTION: Generates and saves the training graphs ---
    def save_metrics_plot(self, history):
        # Create a figure with 2 subplots (Loss and Accuracy)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Plot Training and Validation Loss
        axes[0].plot(history.history['loss'], label='Train Loss', color='blue')
        axes[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Model Loss over Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 2. Plot Training and Validation Accuracy
        axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[1].set_title('Model Accuracy over Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Save the figure to the artifacts/training directory
        plot_path = os.path.join(self.config.root_dir, "training_history_graph.png")
        plt.savefig(plot_path)
        print(f"\n--- Metrics graph saved successfully at: {plot_path} ---")
        plt.show() # Displays it inline in your Jupyter Notebook as well

    def train(self):


        # --- UPDATED: High-Patience Callbacks ---
        
        # Early Stopping: Will now wait for 15 full epochs of NO improvement before giving up.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15, 
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce LR: Waits 6 epochs. If no improvement, it slashes the learning rate to 10% 
        # of what it was (factor=0.1) to force the model to take tiny, microscopic steps.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=6,
            min_lr=1e-6,
            verbose=1
        )

        # ----------------------------------------

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=[early_stopping, reduce_lr] 
        )

        self.save_metrics_plot(history)

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )