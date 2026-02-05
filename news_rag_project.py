# ================================
# Beginner News Aggregator with Simple RAG (Fixed Version)
# ================================

import feedparser
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Step 1: Fetch News from RSS Feed
# -------------------------------

def fetch_news():
    print("\nFetching latest news articles...\n")
    rss_url = "http://feeds.bbci.co.uk/sport/rss.xml"


    feed = feedparser.parse(rss_url)

    articles = []

    for entry in feed.entries[:50]:

        # Get summary safely
        summary = entry.get("summary", "")
        title = entry.get("title", "No Title")

        # Skip empty summaries
        if summary.strip() == "":
            continue

        articles.append({
            "title": title,
            "summary": summary
        })

    print("Total Articles Fetched:", len(articles))

    return articles


# -------------------------------
# Step 2: Compress Article Text
# -------------------------------

def compress_text(text, max_words=50):
    words = text.split()
    return " ".join(words[:max_words])

# -------------------------------
# Step 3: Simple Summarization (Beginner Friendly)
# -------------------------------

def summarize_article(text):
    print("\nGenerating simple summary...\n")

    sentences = text.split(".")
    
    # Take first 2 sentences as summary
    summary = ". ".join(sentences[:5]) + "."

    return summary


# -------------------------------
# Step 4: Topic Grouping (Keyword)
# -------------------------------

def group_by_topic(articles):
    topics = {
        "AI": [],
        "Cybersecurity": [],
        "Mobile": [],
        "Other": []
    }

    for article in articles:
        title = article["title"].lower()

        if "ai" in title:
            topics["AI"].append(article["title"])
        elif "cyber" in title or "hack" in title:
            topics["Cybersecurity"].append(article["title"])
        elif "phone" in title or "mobile" in title:
            topics["Mobile"].append(article["title"])
        else:
            topics["Other"].append(article["title"])

    return topics


# -------------------------------
# Step 5: Build Simple RAG Search
# -------------------------------

def build_rag_system(articles):

    docs = [a["summary"] for a in articles if a["summary"].strip() != ""]

    if len(docs) == 0:
        print("\nERROR: No valid news text found!")
        return None, None

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)

    return vectorizer, tfidf_matrix


def retrieve_news(query, articles, vectorizer, tfidf_matrix):

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)

    best_match = scores.argmax()

    return articles[best_match]


# -------------------------------
# Main Program
# -------------------------------

def main():
    print("======================================")
    print(" Beginner News Aggregator with Simple RAG ")
    print("======================================")

    # Fetch articles
    articles = fetch_news()

    if len(articles) == 0:
        print("No news articles found. Try another RSS feed.")
        return

    # Display titles
    print("\nLatest News Headlines:\n")

    for i, article in enumerate(articles):
        print(f"{i+1}. {article['title']}")

    # Topic grouping
    print("\n======================================")
    print(" Topic Clustering (Keyword Based) ")
    print("======================================")

    topics = group_by_topic(articles)

    for topic, news_list in topics.items():
        print(f"\nTopic: {topic}")
        if len(news_list) == 0:
            print("  No articles found.")
        else:
            for news in news_list:
                print(" -", news)

    # Build RAG system
    vectorizer, tfidf_matrix = build_rag_system(articles)

    if vectorizer is None:
        return

    # Ask Query
    print("\n======================================")
    print(" Ask Questions (Simple RAG Search) ")
    print("======================================")

    query = input("\nEnter your query (example: AI news today): ")

    result = retrieve_news(query, articles, vectorizer, tfidf_matrix)

    print("\n======================================")
    print(" Best Matching Article Found ")
    print("======================================")

    print("\nTitle:", result["title"])

    print("\nCompressed Summary:")
    print(compress_text(result["summary"]))

    # Summarization
    print("\n======================================")
    print(" AI Generated Short Summary ")
    print("======================================")

    short_summary = summarize_article(result["summary"])
    print("\nFinal Summary:", short_summary)

    print("\n======================================")
    print(" Project Finished Successfully ✅")
    print("======================================")


# Run program
if __name__ == "__main__":
    main()
