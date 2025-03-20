# TRI-NIT-hackathon
**Explainable Sexual Harrassment Classfifcation**
This is an ML Model that categorises various forms of sexual harassment shared on online forums, facilitating a faster response and action by authorities and concerned organisations. SafeCity dataset containing narratives of sexual harassment incidents shared on online forums. The model is be capable of handling all types of textual reports (eg. incidents shared on Twitter, Instagram etc) Also included a visualisation module for visualising the key factors leading to particular forms of harassment based on the XAI results. Also Included NLP to pre process the text (Narratives) for this project. Sexual harassment incidents are frequently reported on online platforms, but analyzing and categorizing them effectively is a challenge. This project leverages Natural Language Processing (NLP) and Explainable AI (XAI) to classify harassment cases, enabling organizations and authorities to take prompt action.

**Features:**
-> Multi-class Classification – Categorizes different types of harassment (e.g., commenting, staring, touching/groping).
-> NLP Pipeline – Preprocessing techniques including tokenization, stopword removal, and lemmatization.
-> Machine Learning Models – Uses Logistic Regression, Decision Trees, K-Nearest Neighbors, and SVM.
-> Explainability with SHAP – Highlights key factors influencing classification.
-> Real-World Data – Trained on the SafeCity dataset (crowdsourced reports of sexual harassment).
-> Scalable & Extensible – Can be adapted for analyzing reports from social media platforms (Twitter, Instagram, etc.).

**Tech Stack:**
Programming Language: Python 
Machine Learning: Scikit-learn
NLP: spaCy, TF-IDF, CountVectorizer
XAI (Explainability): SHAP (SHapley Additive exPlanations)
Visualization: Matplotlib, Seaborn

**Dataset:**
This project utilizes the SafeCity dataset, which includes textual reports of harassment cases categorized into various forms such as:
Commenting (verbal harassment)
Ogling / Facial Expressions / Staring (non-verbal harassment)
Touching / Groping (physical harassment)

**Installation & Usage**
1.Clone the repository
git clone https://github.com/yourusername/Explainable-Sexual-Harassment-Classification.git
cd Explainable-Sexual-Harassment-Classification

2.Install Dependencies
pip install -r requirements.txt

3.Run the model
python main.py

**Model Training & Evaluation**
1.Data Preprocessing
->Removed Emojis, hashtags, mentions, and URLs.
->Applied text normalization: lowercasing, tokenization, stopword removal and lemmatization
2.Training Machine Learning Models
The following models were trained and evaluated using TF-IDF feature extraction:
-Logistic Regression
-Decision Tree Classifier
-K Nearest Neighbors(KNN)
-Support Vector Machine
-Naive Bayes(NB)

3.Performance Metrics
**Hamming Score  ~71%**
Classification Report
Precision:correctness
Recall:sensitivity to actual cases
F1-score:balancing precision and recall

**Explanability with SHAP:**
To enhance transparency, we used SHAP values to explain model predictions. The visualization module highlights words that significantly impact classification, aiding in interpretability.

import shap
shap.summary_plot(shap_values, x_test_vectorized, plot_type='bar')

**Results & Insights:**
The Logistic Regression model provided a good balance between accuracy and interpretability.
Touching/Groping incidents were easier to classify, whereas Ogling/Staring had lower recall.
SHAP analysis showed that terms related to verbal threats significantly influenced predictions.

**Contributing:**
Fork the repository
Create a new branch: git checkout -b feature-branch
Commit changes: git commit -m "Add new feature"
Push to GitHub: git push origin feature-branch
Create a Pull Request
