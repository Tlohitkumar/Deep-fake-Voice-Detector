Deepfake Audio Detector

This project provides a complete pipeline for detecting AI-generated (deepfake) audio using both machine-learning and deep-learning methods. It includes tools for audio preprocessing, feature extraction, dataset management, model training, and real-time inference through an API. The repository is structured to support experimentation with traditional models like XGBoost as well as neural networks built with PyTorch, making it easy to benchmark different approaches to deepfake audio detection.

The system can load audio files, extract informative acoustic features, train classifiers, and predict whether an audio sample is real or manipulated. It also includes a FastAPI interface for deploying the detector as a web service.
