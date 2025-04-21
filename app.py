from flask import Flask, render_template, request, redirect, url_for
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from langchain import PromptTemplate, LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from urllib.parse import urlparse, parse_qs
from isodate import parse_duration
import json
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sentiment_analysis import analyze_comments, generate_wordcloud

# ========== CONFIG ========== #
YOUTUBE_API_KEY = "Your_Youtube_API_Key"
GEMINI_API_KEY = "Your_Gemini_API_Key"

app = Flask(__name__)

# ========== UTILITY ========== #
def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    if 'youtube.com' in parsed_url.netloc:
        return parse_qs(parsed_url.query)['v'][0]
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path[1:].split('?')[0]
    else:
        raise ValueError("Unsupported YouTube URL format")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    url = request.form.get("youtube_url")
    video_id = extract_video_id(url)

    # Fetch metadata
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    video_response = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    ).execute()
    video_data = video_response['items'][0]
    snippet = video_data['snippet']
    stats = video_data['statistics']
    content = video_data['contentDetails']

    metadata = {
        "title": snippet['title'],
        "description": snippet['description'],
        "publishedAt": snippet['publishedAt'],
        "duration": str(parse_duration(content['duration'])),
        "viewCount": stats.get('viewCount', 'N/A'),
        "likeCount": stats.get('likeCount', 'N/A'),
        "commentCount": stats.get('commentCount', 'N/A'),
        "tags": snippet.get('tags', []),
        "thumbnail": snippet['thumbnails']['high']['url'],
    }

    # Get transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        full_text = ""
        print("Transcript Error:", e)

    # Summarize
    if full_text:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
        prompt = PromptTemplate(
            input_variables=["transcript"],
            template="""You are an intelligent assistant. Summarize the following YouTube video transcript in a detailed and informative way:

            Transcript:
            {transcript}

            Summary:
            """
        )
        chain = LLMChain(prompt=prompt, llm=llm)
        summary = chain.run(full_text)
    else:
        summary = "Transcript not available."

    # Save report
    report = {
        "metadata": metadata,
        "summary": summary
    }
    with open("video_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Perform sentiment analysis and generate word clouds
    sentiment_results = analyze_comments(video_id, YOUTUBE_API_KEY)

    # Pass generated wordclouds to template (ensure they're saved in the static folder)
    wordcloud_files = [
        "wordcloud_positive.png",
        "wordcloud_negative.png",
        "wordcloud_neutral.png",
        "wordcloud_all.png"
    ]

    # Ensure wordclouds are saved in the 'static' folder
    for file in wordcloud_files:
        if not os.path.exists(os.path.join("static", file)):
            generate_wordcloud(sentiment_results[file.replace("wordcloud_", "").replace(".png", "")], file)

    return render_template(
        "dashboard.html",
        summary=summary,
        metadata=metadata,
        wordclouds=wordcloud_files
    )

if __name__ == "__main__":
    app.run(debug=True)
