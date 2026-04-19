# Student Score Predictor App 🎓

This project is an end-to-end Machine Learning web application that predicts a student's `Math Score` based on demographic details, test preparation, and other test scores. It uses a tuned `RandomForestRegressor` and is deployed beautifully using Streamlit and custom dark-mode CSS.

**Dataset Link:** [Students Performance in Exams (Kaggle)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)  
**GitHub Repository:** [GitHub Link](https://github.com/SohaibAamir28/Student-Score-Predictor)  
**Live Demo:** [Live Streamlit Demo](https://student-score-predictor-app-ml.streamlit.app)

## File Overview

1. `StudentsPerformance.csv`: The core dataset.
2. `model.py`: Script to process the data, perform `GridSearchCV` on a Random Forest Pipeline, and export to `.pkl`.
3. `rf_model_pipeline.pkl`: The serialized inference pipeline and analytics data.
4. `app.py`: Streamlit dashboard code.
5. `requirements.txt`: Necessary dependencies.

## Local Setup

To run this locally, clone this folder and install the requirements:

```bash
pip install -r requirements.txt
python model.py  # to recreate the .pkl file if desired
streamlit run app.py
```

## Cloud Deployment Guide (Streamlit Community Cloud)

You can deploy this repository to the public internet for free using Streamlit Community Cloud. Follow these steps:

### 1. Push to GitHub
1. Create a free account on [GitHub](https://github.com/).
2. Create a new public repository and name it (e.g., `student-score-predictor`).
3. Commit and push the following files to your new GitHub repository:
   - `StudentsPerformance.csv`
   - `rf_model_pipeline.pkl` *(Make sure this file is uploaded!)*
   - `app.py`
   - `requirements.txt`

*(Note: `.pkl` files might be blocked by some firewalls, but Github/Streamlit handle them fine as long as they are pushed).*

### 2. Deploy on Streamlit
1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and create an account (you can log in directly with your GitHub account).
2. Click on **"New app"**.
3. Streamlit will ask you for permission to access your GitHub repositories. Grant it.
4. Fill in the deployment form:
   - **Repository:** Select the repository you just created (`your-username/student-score-predictor`).
   - **Branch:** Select the main branch (usually `main` or `master`).
   - **Main file path:** Type `app.py`.
5. Click **"Deploy!"**

### 3. Let It Bake 🚀
Streamlit will now spawn a remote server, install the packages from your `requirements.txt`, and start running `app.py`. Within 1-2 minutes, you will receive a public, live URL that you can share with the world!
