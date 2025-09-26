import streamlit as st
import joblib
import nltk, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')

# ==================== Background & Style ====================
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

# ==================== Load model & vectorizer ====================
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("Decision_Tree_model.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

# ==================== Session State ====================
if "history" not in st.session_state:
    st.session_state.history = []  # lưu [(text, prediction)]

# ==================== Tabs ====================
tab1, tab2 = st.tabs(["📰 Kiểm tra", "📜 Lịch sử"])

# -------------------- Tab 1: Kiểm tra --------------------
with tab1:
    st.markdown("<h1>📰 Fake News Detector</h1>", unsafe_allow_html=True)
    user_text = st.text_area("Nhập nội dung bài báo:")

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        check_btn = st.button("Kiểm tra", use_container_width=True)

    if check_btn:
        if user_text.strip():
            user_text_clean = re.sub(r'[^a-zA-Z]', ' ', user_text.lower())
            clean_user_text = preprocess(user_text_clean)

            X_new = vectorizer.transform([clean_user_text])
            prediction = model.predict(X_new)[0]

            # Lưu vào lịch sử
            st.session_state.history.append((user_text, prediction))

            # Hiển thị kết quả
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
            # -------------------- Biểu đồ lý do FAKE/REAL --------------------
            # Lấy độ quan trọng từ Decision Tree cho các từ xuất hiện trong bài
            from sklearn.tree import _tree
            import numpy as np
            st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
            def get_word_contribution(model, vectorizer, text_vector):
                fi = model.feature_importances_  # độ quan trọng
                words = vectorizer.get_feature_names_out()
                text_indices = text_vector.nonzero()[1]
                contrib_words = [(words[i], fi[i]) for i in text_indices]
                contrib_words = sorted(contrib_words, key=lambda x: x[1], reverse=True)
                return contrib_words[:5]  # top 10 từ quan trọng

            word_contrib = get_word_contribution(model, vectorizer, X_new)
            if word_contrib:
                words, importances = zip(*word_contrib)
                plt.figure(figsize=(10,6))
                plt.barh(words, importances, color=bg_color)
                plt.xlabel("Độ quan trọng")
                plt.title("Top từ ảnh hưởng tới dự đoán " + result_text)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                st.pyplot(plt)


            # Biểu đồ từ xuất hiện nhiều nhất
            st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
            word_counter = Counter(clean_user_text.split())
            top_words = word_counter.most_common(20)
            words = [w for w, _ in top_words]
            counts = [c for _, c in top_words]

            plt.figure(figsize=(10, 6))
            plt.bar(words, counts, color= bg_color)
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


# -------------------- Tab 2: Lịch sử --------------------
with tab2:
    st.markdown("<h1>📜 Lịch sử kiểm tra</h1>", unsafe_allow_html=True)
    if st.session_state.history:
        for i, (text, pred) in enumerate(reversed(st.session_state.history), 1):
            pred_text = "REAL" if pred == 1 else "FAKE"
            color = "#4CAF50" if pred == 1 else "#FF4B4B"
            st.markdown(
                f"""
                <div style="
                    border: 2px solid {color};
                    border-radius: 8px;
                    padding: 10px;
                    margin-bottom: 10px;
                    background-color: rgba(0, 0, 0, 0.3);
                    color: white;
                    width: 100%;
                    height: 200px;
                    overflow-y: auto;
                    margin-left: auto;
                    margin-right: auto;">
                    <b><span style="color:{color}">Bài báo #{i}:</span></b> {text}<br>
                    <b>Kết quả:</b> <span style="color:{color}; font-weight:bold;">{pred_text}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div style="
                text-align: center;
                border: 2px solid gray;
                padding: 10px;
                border-radius: 10px;
                background-color: gray;
                width: 60%;
                margin: 0 auto;
                font-size: 18px;
                font-weight: bold;
                color: white;">
                Chưa có lịch sử nào!
            </div>
            """,
            unsafe_allow_html=True
        )

