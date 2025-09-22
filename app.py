import streamlit as st
import joblib
import spacy


# --- Load model và vectorizer ---
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("logistic_regression_model.pkl")
nlp = spacy.load("en_core_web_sm")

def clean_data(text: str) -> str:
    text = text.lower()
    doc = nlp(text)
    return " ".join([
        token.text if token.like_url else token.lemma_
        for token in doc
        if not (token.is_punct or token.is_space or token.is_stop)
    ])

st.title("📰 Fake News Detector")

user_text = st.text_area("Nhập nội dung bài báo:")

if st.button("Kiểm tra"):
    if user_text.strip():
        clean_user_text = clean_data(user_text)
        X_new = vectorizer.transform([clean_user_text])
        prediction = model.predict(X_new)[0]
        st.success("Kết quả: " + ("FAKE" if prediction == 1 else "REAL"))
    else:
        st.warning("Vui lòng nhập nội dung!")




