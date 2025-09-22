import streamlit as st
import joblib
import spacy

# --- Load model và vectorizer ---
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("logistic_regression_model.pkl")

st.title("📰 Fake News Detector")

user_text = st.text_area("Nhập nội dung bài báo:")

if st.button("Kiểm tra"):
    if user_text.strip():
        X_new = vectorizer.transform([clean_user_text])
        prediction = model.predict(X_new)[0]
        st.success("Kết quả: " + ("FAKE" if prediction == 1 else "REAL"))
    else:
        st.warning("Vui lòng nhập nội dung!")

