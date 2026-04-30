import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# ------------------------------
# Load models once
# ------------------------------
@st.cache_resource
def load_models():
    bart_model_name = "facebook/bart-large-cnn"
    bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
    roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_name)

    sentiment_pipeline = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer)
    return bart_tokenizer, bart_model, sentiment_pipeline


bart_tokenizer, bart_model, sentiment_pipeline = load_models()

# ------------------------------
# Functions
# ------------------------------
def get_news_urls(stock_name, num_results=5):
    """Fetch latest news URLs from Google News"""
    query = stock_name + " stock news"
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if href and "http" in href:
            links.append(href)
    return links[:num_results]


def scrape_article(url):
    """Extract text content from a news article"""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=5)
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])
    return text[:2000]  # limit size


def summarize_text(text):
    inputs = bart_tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=200)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def analyze_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]  # limit length
    return result["label"], round(result["score"], 2)


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Stock Sentiment Analysis", page_icon="📈", layout="centered")

st.markdown("<h1 style='text-align: center; color: black;'>📈 Stock Sentiment Analysis</h1>", unsafe_allow_html=True)

stock_name = st.text_input("Enter Stock Name", placeholder="e.g. Tesla, Reliance, Samsung")

if st.button("Analyze"):
    if stock_name:
        with st.spinner("Fetching news and analyzing..."):
            urls = get_news_urls(stock_name)

            all_summaries, sentiments, scores = [], [], []
            for url in urls:
                try:
                    article = scrape_article(url)
                    if not article.strip():
                        continue
                    summary = summarize_text(article)
                    sentiment, score = analyze_sentiment(summary)

                    all_summaries.append(summary)
                    sentiments.append(sentiment)
                    scores.append(score)
                except Exception as e:
                    continue

        # ----------------- Results -----------------
        st.subheader("📊 Prediction")
        if sentiments.count("LABEL_2") > sentiments.count("LABEL_0"):
            st.success("The stock will go up! 🚀")
        else:
            st.error("The stock may go down 📉")

        st.subheader("📰 Articles")
        for u in urls:
            st.write(f"[{u}]({u})")

        if scores:
            st.subheader("📉 Average Sentiment Scores")
            st.json({
                "avg_confidence": sum(scores) / len(scores),
                "positive_articles": sentiments.count("LABEL_2"),
                "neutral_articles": sentiments.count("LABEL_1"),
                "negative_articles": sentiments.count("LABEL_0")
            })
    else:
        st.warning("Please enter a stock name.")
