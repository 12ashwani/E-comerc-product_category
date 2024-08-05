import numpy as np
import re
import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from imblearn.pipeline import Pipeline

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

# Load the deep learning model and related components
with open('model_deep_nlp1.pkl', 'rb') as f:
    deep_model = pickle.load(f)

with open('label_encoder2.pkl', 'rb') as f:
    deep_label_encoder = pickle.load(f)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the SMOTE pipeline which includes vectorization and model
with open('model_log.pkl', 'rb') as f:
    smote_pipeline = pickle.load(f)

with open('label_encoder1.pkl', 'rb') as f:
    logistic_label_encoder = pickle.load(f)

def clean_text(text):
    """
    Clean and preprocess text using spaCy and regular expressions.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    if not text.strip():
        return ''
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def predict_deep_category(text):
    """
    Predict the category using the deep learning model.
    """
    cleaned_description = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_description])
    padded_sequence = pad_sequences(sequence, maxlen=100)  # Adjust maxlen as needed
    prediction = deep_model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)[0]
    predicted_category = deep_label_encoder.inverse_transform([predicted_label])[0]
    return predicted_category

def predict_logistic_category(text):
    """
    Predict the category using the traditional ML model within the pipeline.
    """
    cleaned_description = clean_text(text)
    input_vector = smote_pipeline.named_steps['tfidfvectorizer'].transform([cleaned_description])
    
    # Access the model from the pipeline
    model = smote_pipeline.named_steps['logisticregression']
    
    # Ensure feature size matches
    feature_size = input_vector.shape[1]
    expected_feature_size = model.n_features_in_
    print(f"Feature size of the vectorizer: {feature_size}")
    print(f"Number of features expected by the model: {expected_feature_size}")

    if feature_size != expected_feature_size:
        st.error(f"Feature size mismatch: Expected {expected_feature_size}, but got {feature_size}.")
        return None

    prediction = smote_pipeline.predict([cleaned_description])
    predicted_label = prediction[0]  # Get the predicted label
    predicted_category = logistic_label_encoder.inverse_transform([predicted_label])[0]
    n={0: 'Clothing ', 1: 'Jewellery ', 2: 'Footwear ', 3: 'Automotive ', 4: 'Mobiles & Accessories ', 5: 'Home Decor & Festive Needs ', 6: 'Kitchen & Dining ', 7: 'Computers ', 8: 'Watches ', 9: 'Tools & Hardware ', 10: 'Toys & School Supplies ', 11: 'Pens & Stationery ', 12: 'Baby Care ', 13: 'Bags, Wallets & Belts '}
    if predicted_category in n:
      predicted_category=n[predicted_category]
    return predicted_category

# Streamlit app
st.title("Product Categorization App")
st.write("Choose a model and enter a text description to predict its category.")

# Option to select model
model_option = st.selectbox("Select Model", ["Deep Learning Model", "Traditional ML Model"])

# Text input for product description
product_description = st.text_area("Product Description", "Type your product description here...")

if st.button("Predict Category"):
    if product_description.strip():
        if model_option == "Deep Learning Model":
            predicted_category = predict_deep_category(product_description)
        else:
            predicted_category = predict_logistic_category(product_description)
        if predicted_category:
            st.success(f"Predicted Category: {predicted_category}")
    else:
        st.write("Please enter a product description to categorize.")
