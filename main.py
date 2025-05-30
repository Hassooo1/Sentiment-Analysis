from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

import streamlit as st
import re
import nltk
import pickle

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)



def preprocess_text(text):
    # Make text lowercase and remove links, text in square brackets, punctuation, and words containing numbers
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', ' ', text)
    text = re.sub(r'\n', ' ', text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    stop_words.remove('not')
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words).strip()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    porter = PorterStemmer()
    lem_tokens = [porter.stem(lemmatizer.lemmatize(token)) for token in tokens]
    
    return ' '.join(lem_tokens)

def display_result(result):
    if result[0]=="Positive":
        st.subheader(result[0]+":smile:")
    elif result[0]=="Negative":
        st.subheader(result[0]+":pensive:")
    else:
        st.subheader(result[0]+":neutral_face:")



if __name__ == "__main":
    # ========================
    # App Configuration
    # ========================
    st.set_page_config(page_title="Sentimatic: Streamlining Customer Feedback Analysis", page_icon="💬", layout="centered")

    # ========================
    # Custom Styles
    # ========================
    st.markdown("""
        <style>
            .stTextInput>div>div>input {
                border-radius: 8px;
                padding: 10px;
            }
            .stButton>button {
                background-color: #4B0082;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                padding: 8px 16px;
            }
            .main {
                background-color: #f6f5f6;
            }
        </style>
    """, unsafe_allow_html=True)

    # ========================
    # Layout - Title & Description
    # ========================
    
    st.title("💬 Sentimatic")
    st.subheader("Streamlining Customer Feedback Analysis with Sentiment Intelligence")
    st.divider
    classifier = st.selectbox(
        "Which classifier do you want to use?",
        ["Logistic Regression", "Random Forest", "Gradient Boosting Classifier", "Naive Bayes", "Support Vector Machine (SVM)"])
    if classifier == 'Logistic Regression':
        st.write('You selected Logistic Regression')
    elif classifier == 'Naive Bayes':
        st.write('You selected Naive Bayes')
    elif classifier == 'Random Forest':
        st.write('You selected Random Forest')
    elif classifier == 'Gradient Boosting Classifier':
        st.write('You selected Gradient Boosting Classifier')
    else:
        st.write("You selected SVM")
    st.divider()
    
    with open("S1_model.p", 'rb') as mod:
            data = pickle.load(mod)
    vect = data['vectorizer']

    if classifier=="Logistic Regression":
        model = data["lg"]
    elif classifier=="Naive Bayes":
        model = data["nb"]
    elif classifier == 'Random Forest':
        model = data["rf"]
    elif classifier == 'Gradient Boosting Classifier':
        model = data["gbm"]
    else:
        model = data["svm"]


    st.subheader('Check sentiments of a single review:')
    single_review = st.text_area("Enter review:")
    if st.button('Check the sentiment!'):
        review = preprocess_text(single_review)
        inp_test = vect.transform([single_review])
        result = model.predict(inp_test)
        print(result)
        display_result(result)
        
    else:
        st.write('')