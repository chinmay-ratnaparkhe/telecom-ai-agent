"""
Model Loader for Telecom KPI Analysis
Loads all available models (autoencoder, isolation forest, etc.) from the models directory on startup.
Provides unified inference interface for anomaly detection and KPI analysis.
"""

import os
import pickle
import torch
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

class ModelLoader:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self._load_all_models()

    def _load_all_models(self):
        for root, _, files in os.walk(self.model_dir):
            for file in files:
                path = os.path.join(root, file)
                if file.endswith('.pth'):
                    # PyTorch model
                    try:
                        model = torch.load(path, map_location='cpu')
                        self.models[file] = model
                    except Exception as e:
                        print(f"Failed to load PyTorch model {file}: {e}")
                elif file.endswith('.pkl'):
                    # scikit-learn or other pickle model
                    try:
                        with open(path, 'rb') as f:
                            model = pickle.load(f)
                        self.models[file] = model
                    except Exception as e:
                        print(f"Failed to load pickle model {file}: {e}")
                elif file.endswith('.joblib'):
                    # joblib model
                    try:
                        model = joblib.load(path)
                        self.models[file] = model
                    except Exception as e:
                        print(f"Failed to load joblib model {file}: {e}")

    def get_model(self, name):
        return self.models.get(name)

    def list_models(self):
        return list(self.models.keys())

    def infer(self, model_name, data):
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
        # Example: autoencoder anomaly score
        if hasattr(model, 'predict'):
            return model.predict(data)
        elif hasattr(model, 'forward'):
            with torch.no_grad():
                return model.forward(data)
        else:
            raise NotImplementedError(f"Inference not implemented for model {model_name}.")

# Usage example (for server setup):
# loader = ModelLoader()
# print(loader.list_models())
# result = loader.infer('autoencoder_model.pth', input_data)
