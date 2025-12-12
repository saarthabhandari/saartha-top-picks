# sentiment_news.py
# Fetch tweets via snscrape (if installed) and compute sentiment using VADER and TextBlob.
# Also scrape Google News headlines as a fallback.
# Designed to be tolerant if snscrape is missing.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import numpy as np

analyzer = SentimentIntensityAnalyzer()

def fetch_tweets_snscrape_safe(query, limit=150):
    """
    Try to use snscrape to fetch tweets. If snscrape not installed or fails, returns empty list.
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except Exception:
        return []
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            tweets.append(tweet.content)
    except Exception:
        return []
    return tweets

def compute_sentiment_scores(texts):
    """
    Returns dictionary with average VADER compound and TextBlob polarity and simple counts.
    """
    if not texts:
        return {'vader_compound': None, 'textblob_polarity': None, 'n': 0}
    v_scores = []
    tb_scores = []
    for t in texts:
        try:
            v = analyzer.polarity_scores(t)['compound']
            v_scores.append(v)
            tb_scores.append(TextBlob(t).sentiment.polarity)
        except Exception:
            continue
    if not v_scores:
        return {'vader_compound': None, 'textblob_polarity': None, 'n': 0}
    return {'vader_compound': float(np.mean(v_scores)), 'textblob_polarity': float(np.mean(tb_scores)), 'n': len(v_scores)}

def fetch_google_news(company_name, limit=5):
    """
    Simple Google News scrape for headlines. Best-effort; may break if Google blocks.
    """
    try:
        query = company_name + " India"
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}&tbm=nws"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        items = soup.select("div.dbsr")[:limit]
        news = []
        for it in items:
            title = it.select_one("div.JheGif").get_text(strip=True) if it.select_one("div.JheGif") else ""
            snippet = it.select_one("div.Y3v8qd").get_text(strip=True) if it.select_one("div.Y3v8qd") else ""
            source = it.select_one("div.XTjfc").get_text(strip=True) if it.select_one("div.XTjfc") else ""
            news.append({'title': title, 'snippet': snippet, 'source': source})
        return news
    except Exception:
        return []

def sentiment_and_news_for_company(company_name, query_override=None):
    """
    Top-level helper: returns sentiment summary (vader/textblob), number of tweets, and top news headlines.
    """
    query = query_override if query_override else company_name
    tweets = fetch_tweets_snscrape_safe(query)
    sentiment = compute_sentiment_scores(tweets)
    news = fetch_google_news(company_name, limit=5)
    return {'tweets_count': sentiment['n'], 'vader': sentiment['vader_compound'], 'textblob': sentiment['textblob_polarity'], 'news': news}
