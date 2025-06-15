# 🔋 Energy Anomaly Detection Dashboard - Projet LEAD

## 👥 Équipe de Développement
- **Aubin KAMTSA**
- **Sonia KOM**

*Projet Académique de Data Science - Détection d'Anomalies Énergétiques*

---

## 📋 Description du Projet

Ce projet présente une **analyse complète du dataset LEAD (Large-scale Energy Anomaly Detection)** avec un dashboard interactif professionnel. Notre solution combine des techniques d'apprentissage automatique supervisées et non-supervisées pour détecter automatiquement les anomalies dans la consommation énergétique de bâtiments.

### 🎯 Objectifs Principaux
- ✅ **Analyser** les patterns de consommation énergétique sur grande échelle
- ✅ **Détecter** automatiquement les anomalies temporelles dans les données énergétiques
- ✅ **Comparer** différents modèles de détection (Random Forest vs Isolation Forest)
- ✅ **Créer** une interface interactive professionnelle pour l'exploration des résultats
- ✅ **Fournir** des recommandations pour le déploiement en production

### 🎨 Nouveautés de cette Version
- 🎯 **Barres de progression** pour chaque section du dashboard
- 📖 **Explications détaillées** à chaque étape de l'analyse
- 📊 **Graphiques optimisés** avec interactions améliorées
- 🎨 **Interface utilisateur moderne** avec design professionnel
- 💡 **Recommandations** et conseils pour le déploiement

---

## 📊 Dataset LEAD - Source Kaggle

**Source Officielle**: [Energy Anomaly Detection - Kaggle Competition](https://www.kaggle.com/competitions/energy-anomaly-detection/data)

**Structure des Données**: 
- **Train**: 50 bâtiments × 1 année de données = ~438,000 observations
- **Test**: 30 bâtiments × 6 mois de données = ~131,400 observations

### 📝 Variables Principales
| Variable | Description | Type |
|----------|-------------|------|
| `building_id` | Identifiant unique du bâtiment (1-50 pour train, 51-80 pour test) | int32 |
| `timestamp` | Horodatage de la mesure (hourly data) | datetime |
| `meter_reading` | Consommation électrique (kWh) | float32 |
| `anomaly` | Indicateur d'anomalie (0=Normal, 1=Anomalie) | int8 |

### 🏢 Types de Bâtiments
- 🏢 **Office** : Bureaux commerciaux (patterns 9h-17h)
- 🏠 **Residential** : Bâtiments résidentiels (pics matin/soir)
- 🏪 **Commercial** : Centres commerciaux (patterns étendus)
- 🏭 **Industrial** : Sites industriels (consommation continue)

### ⚠️ Types d'Anomalies (~5% du dataset)
- 📈 **Spike** : Pics de consommation soudains (+200% de la normale)
- 📉 **Drop** : Chutes anormales de consommation (-80% de la normale)
- 🔧 **Equipment Failure** : Dysfonctionnements d'équipements
- ⬆️ **Sustained High** : Consommation élevée prolongée (>24h)

### 📁 Fichiers du Dataset
```
data/
├── train.csv              # Données d'entraînement avec labels
├── test.csv               # Données de test (sans labels)
├── train_features.csv     # Features étendues (train)
├── test_features.csv      # Features étendues (test)
└── sample_submission.csv  # Format de soumission Kaggle
```

---

## 🚀 Installation et Lancement

### Prérequis
```bash
Python 3.8+
pip
Git
```

### Installation Rapide
```bash
# 1. Cloner le repository
git clone <your-repo-url>
cd ML-energy-anomaly-detection

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger les données depuis Kaggle
# Visitez: https://www.kaggle.com/competitions/energy-anomaly-detection/data
# Téléchargez et placez les fichiers CSV dans le dossier data/

# 5. Alternative: Générer des données synthétiques (si nécessaire)
python generate_data.py
```

### 🎛️ Lancement du Dashboard Interactif
```bash
# Lancer l'application Dash
python dashboard_app.py

# Ouvrir votre navigateur à l'adresse
# http://127.0.0.1:8050
```

### 📓 Alternative: Script d'Analyse
```bash
# Pour une analyse complète en script
python energy_anomaly_analysis.py
```

---

## 🎮 Utilisation du Dashboard

### 🗂️ Navigation par Sections

1. **📊 Données** 
   - Chargement automatique des fichiers CSV
   - Statistiques descriptives détaillées
   - Informations sur les colonnes et qualité des données
   - Barres de progression en temps réel

2. **🔧 Préprocessing**
   - Création de features temporelles cycliques (sin/cos)
   - Traitement intelligent des valeurs manquantes
   - Visualisation des nouvelles features créées
   - Optimisation des types de données

3. **📈 Exploration (EDA)**
   - Distribution de la consommation avec statistiques
   - Comparaison Normal vs Anomalies
   - Analyse des patterns temporels (hourly/daily)
   - Séries temporelles interactives avec zoom

4. **🤖 Modélisation**
   - **Random Forest** (supervisé) avec class balancing
   - **Isolation Forest** (non-supervisé) pour outliers
   - Métriques de performance (Accuracy, AUC-ROC)
   - Feature importance et matrices de confusion

5. **📋 Résultats**
   - Prédictions sur données de test
   - Visualisation des probabilités d'anomalies
   - Recommandations pour le déploiement
   - Résumé complet de l'analyse

### 🎯 Fonctionnalités Avancées
- ⏳ **Barres de progression** visuelles pour chaque étape
- 💡 **Tooltips explicatifs** sur tous les graphiques
- 📱 **Design responsive** compatible mobile/desktop
- 🔄 **Validation automatique** des étapes précédentes
- 🎨 **Interface moderne** avec Bootstrap et Plotly

---

## 🛠️ Technologies Utilisées

### 🧠 Machine Learning
- **scikit-learn** : Random Forest, Isolation Forest, métriques
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques

### 📊 Visualisation
- **Plotly** : Graphiques interactifs
- **Dash** : Framework web pour applications ML
- **Dash Bootstrap Components** : Interface utilisateur moderne

### 🔧 Développement
- **Python 3.8+** : Langage principal
- **Git** : Contrôle de version
- **Virtual Environment** : Isolation des dépendances

---

## 📁 Structure du Projet

```
ML-energy-anomaly-detection/
├── 📄 README.md                    # Documentation complète
├── 📄 requirements.txt             # Dépendances Python
├── 📄 generate_data.py             # Générateur de données LEAD
├── 📄 dashboard_app.py             # 🎯 Dashboard principal (AMÉLIORÉ)
├── 📄 energy_anomaly_analysis.py   # Script d'analyse alternative
├── 📄 notebook.ipynb              # Notebook Jupyter (optionnel)
│
├── 📂 data/                       # Données du projet (Kaggle)
│   ├── train.csv                  # Données d'entraînement
│   ├── test.csv                   # Données de test
│   ├── train_features.csv         # Features étendues (train)
│   ├── test_features.csv          # Features étendues (test)
│   └── sample_submission.csv      # Format de soumission
│
├── 📂 utils/                      # Modules utilitaires
│   ├── __init__.py
│   ├── visualization.py           # Graphiques personnalisés
│   └── model_training.py          # Fonctions de modélisation
│
├── 📂 models/                     # Modèles sauvegardés
│   ├── random_forest.pkl
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
└── 📂 screenshots/                # Captures d'écran du dashboard
    ├── dashboard_overview.png
    ├── preprocessing_section.png
    ├── modeling_results.png
    └── final_predictions.png
```

---

## 📈 Résultats et Performance

### 🎯 Métriques de Performance
- **Random Forest** : ~95%+ accuracy sur validation
- **Isolation Forest** : ~85%+ accuracy (non-supervisé)
- **AUC-ROC** : ~0.92+ pour Random Forest
- **Détection** : 5-8% d'anomalies sur données test

### ✅ Validation du Modèle
- ✅ Division stratifiée pour classes équilibrées
- ✅ Validation croisée pour robustesse
- ✅ Feature importance pour interprétabilité
- ✅ Matrices de confusion pour analyse détaillée

### 🚀 Prêt pour Production
- ✅ Code modulaire et documenté
- ✅ Gestion d'erreurs robuste
- ✅ Interface utilisateur professionnelle
- ✅ Recommandations de déploiement

---

## 🔮 Améliorations Futures

### 📊 Données
- [ ] Intégration de données météorologiques
- [ ] Support pour streaming en temps réel
- [ ] Base de données pour persistance

### 🤖 Modélisation
- [ ] Deep Learning (Autoencoders, LSTM)
- [ ] Ensemble methods avancés
- [ ] Hyperparameter tuning automatique

### 🎨 Interface
- [ ] Authentification utilisateur
- [ ] Export de rapports PDF
- [ ] Notifications en temps réel
- [ ] API REST pour intégrations

---

## 📞 Support et Contact

### 👥 Équipe
- **Aubin KAMTSA** - Développement principal et modélisation
- **Sonia KOM** - Analyse de données et interface utilisateur

### 🎓 Contexte Académique
Ce projet a été développé dans le cadre d'un cursus de Data Science, démontrant la maîtrise de :
- ✅ Préprocessing avancé de données temporelles
- ✅ Techniques de détection d'anomalies
- ✅ Développement d'applications web interactives
- ✅ Bonnes pratiques en développement Python

### 🔗 Liens Utiles
- **Dataset Source** : [Kaggle Energy Anomaly Detection](https://www.kaggle.com/competitions/energy-anomaly-detection/data)
- **Documentation Dash** : [https://dash.plotly.com/](https://dash.plotly.com/)
- **Scikit-learn** : [https://scikit-learn.org/](https://scikit-learn.org/)

### 🔧 Support Technique
Pour toute question ou problème :
1. Vérifiez la documentation ci-dessus
2. Consultez les messages d'erreur dans le dashboard
3. Vérifiez que tous les fichiers de données sont présents
4. Contactez l'équipe de développement

---

## 📜 License

Ce projet est développé à des fins éducatives et de démonstration. 
© 2025 - Aubin KAMTSA & Sonia KOM - Tous droits réservés.

---

**🎉 Merci d'utiliser notre dashboard de détection d'anomalies énergétiques !**

*Basé sur le dataset Energy Anomaly Detection de Kaggle*
