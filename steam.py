import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load the model
model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

st.title("Sequence Classification")

# Input
sequence_to_classify = st.text_input("Enter the sequence to classify:")
if st.button("Classify"):
    if sequence_to_classify:
        candidate_labels = ["transportations", "food", "bathroom", "guidance"]
        try:
            output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
            st.write("Classification:", output['labels'][0])
        except Exception as e:
            st.error("An error occurred during classification.")
            st.error(str(e))
    else:
        st.warning("Please enter a sequence to classify.")
