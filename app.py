import streamlit as st
import joblib
import nltk, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://c.pxhere.com/photos/45/ba/old_newspaper_newspaper_the_1960s_retro_sepia_old_nowiny_gliwickie_information-965064.jpg!s2");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

h1 {
    text-align: center;
}

div[data-testid="stTextArea"] label p {
    color: white;
    font-size: 20px;
    font-weight: bold;
}

textarea {
    min-height: 250px !important;
    width: 100% !important;
}

div.stButton > button {
    background-color: #005f99;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    cursor: pointer;
}
div.stButton > button:hover {
    background-color: #0077b3;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("Decision_Tree_model.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

st.markdown("<h1>📰 Fake News Detector</h1>", unsafe_allow_html=True)

user_text = st.text_area("Nhập nội dung bài báo:")

col1, col2, col3 = st.columns([1,1,1])
with col2:
    check_btn = st.button("Kiểm tra", use_container_width=True)

if check_btn:
    if user_text.strip():
        user_text = user_text.lower()
        user_text = re.sub(r'[^a-zA-Z]', ' ', user_text)
        clean_user_text = preprocess(user_text)

        X_new = vectorizer.transform([clean_user_text])
        prediction = model.predict(X_new)[0]

        if prediction == 1:
            result_text = "REAL"
            bg_color = "#4CAF50" 
        else:
            result_text = "FAKE"
            bg_color = "#FF4B4B"  

        st.markdown(
            f"""
            <div style="
                text-align: center;
                border: 2px solid {bg_color};
                padding: 10px;
                border-radius: 10px;
                background-color: {bg_color};
                width: 60%;
                margin: 0 auto;
                font-size: 20px;
                font-weight: bold;
                color: white;">
                {result_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

        word_counter = Counter(clean_user_text.split())
        top_words = word_counter.most_common(20)
        words = [w for w, _ in top_words]
        counts = [c for _, c in top_words]

        plt.figure(figsize=(10, 6))
        plt.bar(words, counts, color='green')
        plt.xticks(rotation=45, ha='right')
        plt.title("Top 20 từ xuất hiện nhiều nhất")
        plt.ylabel("Tần số")
        plt.tight_layout()
        st.pyplot(plt)

    else:
        st.markdown(
            """
            <div style="
                text-align: center;
                border: 2px solid orange;
                padding: 10px;
                border-radius: 10px;
                background-color: orange;
                width: 60%;
                margin: 0 auto;
                font-size: 18px;
                font-weight: bold;
                color: white;">
                Vui lòng nhập nội dung!
            </div>
            """,
            unsafe_allow_html=True
        )
