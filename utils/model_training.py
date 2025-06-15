"""
Utilitaires pour l'entraînement et l'évaluation des modèles de détection d'anomalies
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetectionModels:
    """
    Classe pour gérer les différents modèles de détection d'anomalies
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prépare les données pour l'entraînement
        """
        # Division temporelle pour respecter l'ordre chronologique
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Pour les modèles non supervisés, on utilise seulement les données normales
        normal_indices = y_train == 0
        X_train_normal = X_train[normal_indices]
        
        self.feature_names = X.columns.tolist()
        
        return X_train, X_test, y_train, y_test, X_train_normal
    
    def train_isolation_forest(self, X_train, contamination=0.1, random_state=42):
        """
        Entraîne un modèle Isolation Forest
        """
        print("🌲 Entraînement Isolation Forest...")
        
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0
        )
        
        model.fit(X_train)
        self.models['isolation_forest'] = model
        
        print("✅ Isolation Forest entraîné")
        return model
    
    def train_lof(self, X_train, contamination=0.1):
        """
        Entraîne un modèle Local Outlier Factor
        """
        print("🔍 Entraînement Local Outlier Factor...")
        
        model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=20,
            algorithm='auto',
            novelty=True
        )
        
        model.fit(X_train)
        self.models['lof'] = model
        
        print("✅ LOF entraîné")
        return model
    
    def train_one_class_svm(self, X_train, nu=0.1):
        """
        Entraîne un modèle One-Class SVM
        """
        print("🔧 Entraînement One-Class SVM...")
        
        model = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale'
        )
        
        model.fit(X_train)
        self.models['one_class_svm'] = model
        
        print("✅ One-Class SVM entraîné")
        return model
    
    def build_autoencoder(self, input_dim, encoding_dim=32):
        """
        Construit un autoencoder pour la détection d'anomalies
        """
        # Couche d'entrée
        input_layer = Input(shape=(input_dim,))
        
        # Encodeur
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        
        # Décodeur
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Modèle complet
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def train_autoencoder(self, X_train, epochs=100, batch_size=32, validation_split=0.2):
        """
        Entraîne un autoencoder
        """
        print("🧠 Entraînement Autoencoder...")
        
        # Construction du modèle
        autoencoder = self.build_autoencoder(X_train.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Entraînement
        history = autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.models['autoencoder'] = autoencoder
        
        print("✅ Autoencoder entraîné")
        return autoencoder, history
    
    def predict_anomalies(self, model_name, X_test, threshold=None):
        """
        Prédit les anomalies avec un modèle spécifique
        """
        model = self.models[model_name]
        
        if model_name == 'autoencoder':
            # Pour l'autoencoder, on calcule l'erreur de reconstruction
            X_pred = model.predict(X_test, verbose=0)
            mse = np.mean(np.square(X_test - X_pred), axis=1)
            
            if threshold is None:
                threshold = np.percentile(mse, 95)
            
            predictions = (mse > threshold).astype(int)
            scores = mse
            
        else:
            # Pour les autres modèles
            predictions = model.predict(X_test)
            # Conversion: -1 (anomalie) -> 1, 1 (normal) -> 0
            predictions = (predictions == -1).astype(int)
            
            # Scores de decision
            if hasattr(model, 'decision_function'):
                scores = -model.decision_function(X_test)  # Plus négatif = plus anormal
            else:
                scores = model.negative_outlier_factor_
        
        return predictions, scores
    
    def evaluate_model(self, model_name, y_true, y_pred, scores):
        """
        Évalue un modèle et stocke les résultats
        """
        # Métriques de classification
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC
        try:
            auc = roc_auc_score(y_true, scores)
        except:
            auc = np.nan
        
        # Stockage des résultats
        self.results[model_name] = {
            'classification_report': report,
            'confusion_matrix': cm,
            'auc': auc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'predictions': y_pred,
            'scores': scores
        }
        
        return self.results[model_name]
    
    def train_all_models(self, X_train, X_test, y_train, y_test, X_train_normal):
        """
        Entraîne tous les modèles et les évalue
        """
        print("🚀 Entraînement de tous les modèles...\n")
        
        # Isolation Forest
        self.train_isolation_forest(X_train_normal)
        pred_if, scores_if = self.predict_anomalies('isolation_forest', X_test)
        self.evaluate_model('isolation_forest', y_test, pred_if, scores_if)
        
        # Local Outlier Factor
        self.train_lof(X_train_normal)
        pred_lof, scores_lof = self.predict_anomalies('lof', X_test)
        self.evaluate_model('lof', y_test, pred_lof, scores_lof)
        
        # One-Class SVM
        self.train_one_class_svm(X_train_normal)
        pred_svm, scores_svm = self.predict_anomalies('one_class_svm', X_test)
        self.evaluate_model('one_class_svm', y_test, pred_svm, scores_svm)
        
        # Autoencoder
        self.train_autoencoder(X_train_normal)
        pred_ae, scores_ae = self.predict_anomalies('autoencoder', X_test)
        self.evaluate_model('autoencoder', y_test, pred_ae, scores_ae)
        
        print("\n✅ Tous les modèles entraînés et évalués")
        
        return self.results
    
    def get_results_summary(self):
        """
        Retourne un résumé des performances des modèles
        """
        summary = []
        
        for model_name, results in self.results.items():
            summary.append({
                'Modèle': model_name.replace('_', ' ').title(),
                'Précision': f"{results['precision']:.3f}",
                'Rappel': f"{results['recall']:.3f}",
                'F1-Score': f"{results['f1_score']:.3f}",
                'ROC-AUC': f"{results['auc']:.3f}" if not np.isnan(results['auc']) else 'N/A'
            })
        
        return pd.DataFrame(summary)
    
    def save_models(self, models_dir):
        """
        Sauvegarde tous les modèles entraînés
        """
        import os
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == 'autoencoder':
                model.save(f"{models_dir}/autoencoder_model.h5")
            else:
                joblib.dump(model, f"{models_dir}/{model_name}.pkl")
        
        print(f"✅ Modèles sauvegardés dans {models_dir}")
    
    def load_models(self, models_dir):
        """
        Charge les modèles sauvegardés
        """
        import os
        from tensorflow.keras.models import load_model
        
        # Chargement des modèles sklearn
        for model_name in ['isolation_forest', 'lof', 'one_class_svm']:
            model_path = f"{models_dir}/{model_name}.pkl"
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
        
        # Chargement de l'autoencoder
        ae_path = f"{models_dir}/autoencoder_model.h5"
        if os.path.exists(ae_path):
            self.models['autoencoder'] = load_model(ae_path)
        
        print(f"✅ Modèles chargés depuis {models_dir}")

def plot_model_comparison(results_dict, save_path=None):
    """
    Visualise la comparaison des performances des modèles
    """
    # Préparation des données
    models = list(results_dict.keys())
    metrics = ['precision', 'recall', 'f1_score', 'auc']
    
    data = []
    for model in models:
        for metric in metrics:
            value = results_dict[model][metric]
            if not np.isnan(value):
                data.append({
                    'Modèle': model.replace('_', ' ').title(),
                    'Métrique': metric.replace('_', ' ').title(),
                    'Score': value
                })
    
    df_metrics = pd.DataFrame(data)
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_metrics, x='Métrique', y='Score', hue='Modèle')
    plt.title('Comparaison des Performances des Modèles')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curves(results_dict, y_true, save_path=None):
    """
    Affiche les courbes ROC pour tous les modèles
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, results in results_dict.items():
        if not np.isnan(results['auc']):
            fpr, tpr, _ = roc_curve(y_true, results['scores'])
            plt.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC = {results['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Référence')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC - Comparaison des Modèles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Test des fonctions
    print("🧪 Test des utilitaires de modélisation...")
    
    # Génération de données de test
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], 1000, p=[0.9, 0.1]))
    
    # Test des modèles
    detector = AnomalyDetectionModels()
    X_train, X_test, y_train, y_test, X_train_normal = detector.prepare_data(X, y)
    
    print(f"📊 Données d'entraînement: {len(X_train)} lignes")
    print(f"📊 Données de test: {len(X_test)} lignes")
    print("✅ Tests réussis!")
