# DevelopersHub Advanced Internship – AI/ML Engineering Tasks

This repository contains the completed tasks for the **DevelopersHub Corporation AI/ML Engineering – Advanced Internship**.  
Completed tasks in this repository:

- **Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API**
- **Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data**
- **Task 4: Context-Aware Chatbot Using LangChain or RAG**

---


---

# Task 2 – End-to-End ML Pipeline (Telco Churn Prediction)

### Objective
Build a production-ready machine learning pipeline to predict customer churn using scikit-learn Pipeline API.

### Methodology
- Data preprocessing using `ColumnTransformer`
- Model training using Logistic Regression and Random Forest
- Hyperparameter tuning using `GridSearchCV`
- Pipeline export using `joblib`

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1 Score
- Confusion Matrix

### Output
- Final pipeline saved as `churn_pipeline.joblib`

---

# Task 3 – Multimodal ML (Housing Price Prediction)

### Objective
Predict housing prices using both tabular data and house images.

### Methodology
- CNN-based image feature extraction
- Combine image features with tabular features
- Train regression model
- Evaluate using MAE and RMSE

### Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

# Task 4 – Context-Aware Chatbot (RAG + LangChain)

### Objective
Build a context-aware chatbot that remembers conversation history and retrieves information from a knowledge base using RAG.

### Methodology
- Create document corpus (`docs/`)
- Build vector store using embeddings
- Implement context memory using LangChain
- Deploy chatbot using Streamlit

### Features
- Context memory (chat history)
- Document retrieval via vector search
- Streamlit-based UI

---
