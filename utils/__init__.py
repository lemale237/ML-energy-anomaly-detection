"""
Package utilitaires pour le projet de détection d'anomalies énergétiques
"""

from .data_preprocessing import generate_energy_data, preprocess_data
from .model_training import AnomalyDetectionModels
from .visualization import *

__version__ = "1.0.0"
__author__ = "Aubin KAMTSA & Sonia KOM"
