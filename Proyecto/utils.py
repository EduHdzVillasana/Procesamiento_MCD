# %%
import joblib
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
# %%
def load_model():
    """
    Loads the trained model from file.
    """    
    model = joblib.load("modelo_svm.pkl")
    return model
# %%
def process_text (text):
    lemmatizer = WordNetLemmatizer()
    stop_words_df = pd.read_excel("stop_words.xlsx")
    stop_words_list = list(stop_words_df["Words"].values)
    cleaned_text =re.sub('[^a-zA-Z]', ' ', text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.split()
    cleaned_text = [word for word in cleaned_text if word not in stop_words_list]
    cleaned_text = [lemmatizer.lemmatize(word) for word in cleaned_text]
    cleaned_text = ' '.join(cleaned_text)
    tfidf_vect = joblib.load("tfidf_vect.pkl")
    text_tfidf =  tfidf_vect.transform([cleaned_text])
    return text_tfidf
# %%
def model_predict(text):
    """
    Predicts using the loaded model.
    """

    text_features = process_text(text)
    model = load_model()
    prediction = model.predict(text_features)  # Fixed: Changed email to features
    # prediction = 1 if prediction == 1 else -1
    return prediction
# %%
