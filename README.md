# Language Detection
A comprehensive natural language processing (NLP) repository containing automated data scraping pipelines, traditional machine learning classifiers, deep learning architectures, and deployment scripts developed during coursework at NUS.

---

## Repository Structure

```text
├── Deep Learning model.ipynb                          # ANN model architecture trained on character trigrams
├── Machine Learning Model.ipynb                       # Multinomial Naive Bayes classification pipeline
├── Spoken_Language_Identification_Using_Deep_Learning # Paper refered for literature review
├── Text Scrapping.ipynb                               # Automated web scraper collecting multi-language Wikipedia data
└── Deployment code.py                                 # Python/Flask script for model serving and inference
```
## Components Overview

1. **Data Scraping (`Text Scrapping.ipynb`):** Implements multi-language text collection pipelines using BeautifulSoup and requests to gather raw corpora from Wikipedia.
2. **Machine Learning Model (`Machine Learning Model.ipynb`):** Utilizes a bag-of-words text representation (`CountVectorizer`) paired with a Multinomial Naive Bayes classifier, achieving high multi-language classification accuracy (~98.4%) across 17 languages.
3. **Deep Learning Models (`Deep Learning model.ipynb` & Spoken Language Script):** Implements a multi-layer Feedforward Neural Network (ANN) utilizing dense layers ('relu' and 'softmax'). Processes text using character-level trigrams to capture granular sub-word linguistic features, achieving accuracies exceeding 98.8% to 99.4%.
4. **Deployment (`Deployment code.py`):** Houses the server-side inference and Flask-based routing code designed to serve model predictions dynamically.

---

## Tech Stack & Libraries
1. **Language:** Python
2. **Data Processing & Web Scraping:** Pandas, NumPy, BeautifulSoup, Requests
3. **Deployment:** Flask

(Note: Specific machine learning and deep learning frameworks are detailed in the model comparison table below).

---
| Feature / Aspect | Machine Learning Model (`Machine Learning Model.ipynb`) | Deep Learning Model (`Deep Learning model.ipynb`) |
| :--- | :--- | :--- |
| **Algorithm Type** | **Multinomial Naive Bayes** (`MultinomialNB`) | **Feedforward Neural Network (ANN)** (Multi-Layer Perceptron with `Dense` layers) |
| **Frameworks & Libraries** | `scikit-learn`, `pandas`, `numpy`, `pickle` | `TensorFlow / Keras`, `scikit-learn`, `pandas`, `numpy` |
| **Feature Extraction** | **Word-level Bag-of-Words** (`CountVectorizer` on full text tokens) | **Character-level Trigrams** (`CountVectorizer` with `analyzer='char'`, `ngram_range=(3, 3)`) |
| **Target Encoding** | Label Encoding (`LabelEncoder`) | One-Hot Encoding (`to_categorical` / `np_utils`) |
| **Scope & Languages** | **17 languages** (e.g., English, French, Spanish, Russian, Arabic, etc.) | **6 languages** (`deu`, `eng`, `fra`, `ita`, `por`, `spa`) |
| **Model Architecture** | Probabilistic frequency-based classifier | Multi-layer sequential neural network (`relu` hidden layers, `softmax` output layer) |
| **Exact Test Accuracy** | **98.40%** (`0.98404`) | **98.79%** (standard test set) to **99.49%** (secondary test set: `testweird.csv`) |
| **Artifacts & Deployment** | Saved as `model.pkl` and `transform.pkl` for the Flask web application | Saved as model weights (`weights.h5`) for deep learning experiments |

## Usage Note
This repository serves as a portfolio archive showcasing an end-to-end machine learning and deep learning workflow; ranging from web scraping and feature engineering to model building, evaluation, and backend web deployment.

**Data & Artifact Availability:** This repository focuses on providing the complete, reproducible code pipeline. Users can execute the included notebooks to scrape the required data from scratch, train the models, and generate the necessary artifacts locally for deployment.
