# preload_models.py
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification

print("Downloading BART summarization model...")
BartTokenizer.from_pretrained("facebook/bart-large-cnn")
BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

print("Downloading RoBERTa sentiment model...")
AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

print("Models downloaded.")
