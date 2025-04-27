import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, precision_score, recall_score 

def main():
    st.title("🍄 Mushroom Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Predict whether your mushrooms are **edible or poisonous**! 🍄")
    st.sidebar.markdown("Predict whether your mushrooms are **edible or poisonous**! 🍄")
    
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        y_pred = model.predict(x_test)
        
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=ax)
            st.pyplot(fig)
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            if hasattr(model, "decision_function"):
                y_scores = model.decision_function(x_test)
            else:
                y_scores = model.predict_proba(x_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label="ROC Curve")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            st.pyplot(fig)
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            if hasattr(model, "decision_function"):
                y_scores = model.decision_function(x_test)
            else:
                y_scores = model.predict_proba(x_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label="Precision-Recall Curve")
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            st.pyplot(fig)

    def user_input_features(x_train, df):
        st.sidebar.subheader("Enter Mushroom Measurements")
        user_data = {}
        for col in x_train.columns:
            options = list(df[col].unique())
            user_data[col] = st.sidebar.selectbox(f"{col}", options)
        features = pd.DataFrame(user_data, index=[0])
        return features

    # --- main flow ---
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['Edible', 'Poisonous']

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (kernel coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100,500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest",100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The Maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap sample when building trees",(True,False), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        
    # Train the model when button pressed
    if st.sidebar.button("Classify", key=classifier):
        if classifier == "Support Vector Machine (SVM)":
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        elif classifier == "Logistic Regression":
            model = LogisticRegression(C=C, max_iter=max_iter)
        elif classifier == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
        
        model.fit(x_train, y_train)
        st.session_state.model = model  # Save the model
        
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.subheader(f"{classifier} Results")
        st.write("Accuracy: ", round(accuracy,2))
        st.write("Precision: ", round(precision_score(y_test, y_pred),2))
        st.write("Recall: ", round(recall_score(y_test, y_pred),2))
        plot_metrics(metrics, model, x_test, y_test, class_names)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

    # 🆕 User input for prediction
    st.subheader("Predict Your Own Mushroom 🍄")
    user_data = user_input_features(x_train, df)
    
    if st.button("Predict"):
        if "model" in st.session_state and st.session_state.model is not None:
            # Need to make sure user data columns match training columns
            user_data = user_data[x_train.columns]
            prediction = st.session_state.model.predict(user_data)
            if prediction[0] == 0:
                st.success("🎉 Your mushroom is **Edible**!")
            else:
                st.error("⚠️ Your mushroom is **Poisonous**! Be careful!")
        else:
            st.warning("⚠️ Please train the model first by clicking 'Classify'!")
    st.sidebar.title("Author: Suresh Tamang")

if __name__ == '__main__':
    main()
