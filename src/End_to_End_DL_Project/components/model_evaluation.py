import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from End_to_End_DL_Project.entity.config_entity import EvaluationConfig
from End_to_End_DL_Project.utils.common import save_json
import os



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            # If you had a validation_split here previously, you can leave it or remove it,
            # it won't affect anything since we are pointing directly to the test folder.
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            # --- CHANGE 1: Point directly to the test folder ---
            directory=os.path.join(self.config.training_data, "test"), 
            
            class_mode="categorical",
            
            # --- CHANGE 2: CRITICAL for Confusion Matrix ---
            shuffle=False, 
            
            **dataflow_kwargs
        )
        
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        # Load with compile=False to avoid the optimizer bug, then recompile for evaluation
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        
        print("Evaluating basic metrics...")
        self.score = self.model.evaluate(self.valid_generator)
        
        print("Generating detailed predictions for Confusion Matrix & F1...")
        # Get raw probabilities and convert to class predictions
        y_pred_probs = self.model.predict(self.valid_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self.valid_generator.classes
        class_labels = list(self.valid_generator.class_indices.keys())

        # Calculate advanced metrics (weighted accounts for class imbalances)
        self.f1 = f1_score(y_true, y_pred, average='weighted')
        self.precision = precision_score(y_true, y_pred, average='weighted')
        self.recall = recall_score(y_true, y_pred, average='weighted')

        # Generate and save Confusion Matrix plot
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - 4 Class CT Scans')
        plt.tight_layout()
        
        self.cm_plot_path = "confusion_matrix.png"
        plt.savefig(self.cm_plot_path)
        plt.close()

        self.save_score()

    def save_score(self):
        # Save all advanced metrics locally
        scores = {
            "loss": self.score[0], 
            "accuracy": self.score[1],
            "f1_score": self.f1,
            "precision": self.precision,
            "recall": self.recall
        }
        save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            
            # Log all numerical metrics to MLflow
            mlflow.log_metrics({
                "loss": self.score[0], 
                "accuracy": self.score[1],
                "f1_score": self.f1,
                "precision": self.precision,
                "recall": self.recall
            })
            
            # Log the Confusion Matrix image so you can view it in the MLflow UI
            mlflow.log_artifact(self.cm_plot_path, "evaluation_plots")

            if tracking_url_type_store != "file":
                # Updated Model Name to reflect our new architecture
                mlflow.keras.log_model(self.model, "model", registered_model_name="DenseNet121Model")
            else:
                mlflow.keras.log_model(self.model, "model")