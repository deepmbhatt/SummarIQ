from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from googleapiclient.discovery import build
import os

def get_comments(video_id, api_key, max_results=100):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments

def classify_sentiment(comment):
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def generate_wordcloud(text_list, filename):
    if not os.path.exists("static"):
        os.makedirs("static")
    
    # Combine all comments into one string
    text = " ".join(text_list)
    
    # Generate word cloud with specific settings
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="tab10").generate(text)
    
    # Save to the static folder
    wordcloud.to_file(os.path.join("static", filename))

def analyze_comments(video_id, api_key):
    comments = get_comments(video_id, api_key)
    sentiment_data = {"positive": [], "negative": [], "neutral": [], "all": comments}

    # Classify sentiment for each comment and add to the respective category
    for comment in comments:
        sentiment = classify_sentiment(comment)
        sentiment_data[sentiment].append(comment)

    # Generate word clouds for each sentiment category
    generate_wordcloud(sentiment_data["positive"], "wordcloud_positive.png")
    generate_wordcloud(sentiment_data["negative"], "wordcloud_negative.png")
    generate_wordcloud(sentiment_data["neutral"], "wordcloud_neutral.png")
    generate_wordcloud(sentiment_data["all"], "wordcloud_all.png")

    return sentiment_data
