import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Load the model
model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

st.title("Sequence Classification")

# File upload
uploaded_file = st.file_uploader("Upload an Excel file", type=['xlsx'])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    
    candidate_labels = ["transportations", "food", "bathroom", "guidance"]
    labels = []
    for index, row in data.iterrows():
        try:
            sequence_to_classify = row[0] # assuming the text to classify is in the first column
            output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
            labels.append(output['labels'][0])
        except Exception as e:
            st.error(f"An error occurred during classification for row {index}.")
            st.error(str(e))

    # Add classification results to the data frame
    data['label'] = labels
    
    # Display data
    st.dataframe(data)
