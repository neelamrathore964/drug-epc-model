import streamlit as st
import requests

API_URL = "http://localhost:8000/predict_drug_review"

st.title("💊 Drug Review Sentiment Classifier")

# Ask for API key first
api_key_input = st.text_input("🔑 Enter your API Key", type="password")

if api_key_input:
    # Show input only if API key is entered
    review_text = st.text_area("💬 Enter your drug review here:")

    if st.button("🚀 Predict Sentiment"):
        headers = {"x-api-key": api_key_input}  # Matching FastAPI header
        payload = {"review": review_text}
        response = requests.post(API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            sentiment = response.json()["predicted_sentiment"]
            # Map sentiment to emoji
            emoji_map = {
                "positive": "😊",
                "neutral": "😐",
                "negative": "😞"
            }
            emoji = emoji_map.get(sentiment, "")

            st.success(f"🧠 Sentiment: {sentiment} {emoji}")
            #st.success(f"🧠 Sentiment: {sentiment}")
        elif response.status_code == 403:
            st.error("❌ Invalid API Key.")
        else:
            st.error("❌ Server error: " + response.text)
else:
    st.warning("Please enter the API Key to access the prediction interface.")
