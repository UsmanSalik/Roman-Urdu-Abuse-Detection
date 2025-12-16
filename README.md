Roman Urdu Abuse Detection

This project is a machine learning–based Roman Urdu abuse detection system that predicts whether a given comment is abusive or non-abusive. The models are trained from scratch on a custom Roman Urdu dataset containing over 170,000 comments.

The system focuses on handling the informal nature of Roman Urdu, including slang, spelling variations, and abusive expressions commonly found in online text.

Features

Detects abusive vs non-abusive Roman Urdu comments

Uses a custom-built Roman Urdu corpus (170k+ comments)

Trained from scratch using classical ML models:

Logistic Regression

Support Vector Machine (SVM)

Random Forest

Includes text preprocessing and feature engineering

Simple and user-friendly prediction interface via app.py

Project Structure
├── app.py                     # Run predictions on new input
├── Abusive_classify.ipynb     # Training, preprocessing, experiments
├── models/
│   ├── svm_model.pkl
│   ├── logistic_model.pkl
│   ├── random_forest.pkl
│   └── roman_urdu_corpus.txt
├── requirements.txt
└── README.md

Dataset

Language: Roman Urdu

Size: 170,000+ comments

Type: Custom curated dataset

Labels:

1 → Abusive

0 → Non-abusive

The dataset was preprocessed and used to train all models from scratch.

Preprocessing & Training

The following steps were applied during training:

Text cleaning and normalization

Tokenization

Roman Urdu–specific preprocessing

Feature extraction

Model training and evaluation

All experiments and training workflows are documented in Abusive_classify.ipynb.

Models Used

Logistic Regression – baseline linear classifier

Support Vector Machine (SVM) – high-accuracy classifier

Random Forest – ensemble-based model for robustness

All models are saved and reused for inference.

How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Run the application
python app.py

3. Predict a comment

Enter a Roman Urdu comment when prompted.
The system will output whether the comment is Abusive or Non-Abusive.

Example

Input:

tum bohat gandi baat karte ho


Output:

Prediction: Abusive

Future Improvements

Deep learning models (LSTM / Transformers)

Multilingual and code-mixed support

REST API or web interface

Improved contextual understanding

Author

Developed as a practical NLP and machine learning project focused on low-resource language abuse detection.