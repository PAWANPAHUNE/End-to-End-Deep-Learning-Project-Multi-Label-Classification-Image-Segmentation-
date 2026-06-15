import os
import sys

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import numpy as np
import gradio as gr
import tf_keras as keras
from tf_keras.layers import InputLayer
from tf_keras.preprocessing import image

class FixedInputLayer(InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        if 'batch_input_shape' not in kwargs and 'shape' not in kwargs:
            kwargs['shape'] = (224, 224, 3)
        super().__init__(**kwargs)

GLOBAL_MODEL_PATH = os.path.join("model", "model.h5")

loaded_network = keras.models.load_model(
    GLOBAL_MODEL_PATH,
    custom_objects={'InputLayer': FixedInputLayer}
)


def predict_image(img_path):
    if img_path is None:
        return "Please upload or select an image first."
    
    try:
        test_image = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(test_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        raw_predictions = loaded_network(img_array, training=False)
        result = np.argmax(raw_predictions.numpy(), axis=1)
        predicted_idx = int(result[0])
        
        class_mapping = {
            0: 'Adenocarcinoma',
            1: 'Large Cell Carcinoma',
            2: 'Normal',
            3: 'Squamous Cell Carcinoma'
        }
        
        return class_mapping.get(predicted_idx, "Unknown Class")
    except Exception as e:
        return f"Error during prediction: {str(e)}"

base_test_dir = "Data_Samples_for_UI/data/test"
categories = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]

def gather_samples_from_category(category_name):
    target_path = os.path.join(base_test_dir, category_name)
    found_images = []
    if os.path.exists(target_path):
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG'):
            found_images.extend(glob.glob(os.path.join(target_path, ext)))
    return found_images[:5]

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

* {
    font-family: 'JetBrains Mono', monospace !important;
}
.gradient-btn {
    background: linear-gradient(45deg, #f39c12, #d35400) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
}
.gradient-btn:hover {
    background: linear-gradient(45deg, #e67e22, #e67e22) !important;
    cursor: pointer;
}
.center-text {
    text-align: center;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"), css=custom_css) as demo:
    
    with gr.Row():
        with gr.Column(elem_classes="center-text"):
            gr.Markdown("# **End To End Deep Learning Project**")
            gr.Markdown("### *Developed by Pawan Pahune*")
            gr.HTML("<hr style='border: 0; height: 1px; background: #2a2a2a; margin-bottom: 20px;'>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### **1. Input CT Scan Layer**")
            input_image = gr.Image(type="filepath", label="Upload Chest CT Scan Specimen")
            predict_btn = gr.Button("Execute Prediction", elem_classes="gradient-btn")
            
        with gr.Column(scale=1):
            gr.Markdown("### **2. Diagnostic Network Output**")
            output_text = gr.Textbox(label="Model Decision Matrix Result", placeholder="Awaiting matrix classification...", interactive=False)
            
    gr.HTML("<br><hr style='border: 0; height: 1px; background: #2a2a2a; margin-bottom: 20px;'>")
    gr.Markdown("## **Data Samples Directory for Rapid Testing**")
    gr.Markdown("Select a target class to see actual sample specimens located inside your local project folders:")

    with gr.Tabs():
        for cat in categories:
            with gr.TabItem(label=cat.replace('.', ' ').title()):
                cat_samples = gather_samples_from_category(cat)
                if cat_samples:
                    gr.Examples(
                        examples=cat_samples,
                        inputs=input_image,
                        outputs=output_text,
                        fn=predict_image,
                        cache_examples=False
                    )
                else:
                    gr.Markdown(f"*No current data specimens found inside: `{base_test_dir}/{cat}/`*")

    predict_btn.click(fn=predict_image, inputs=input_image, outputs=output_text)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=None)
    demo.launch(server_name="0.0.0.0", server_port=7860)