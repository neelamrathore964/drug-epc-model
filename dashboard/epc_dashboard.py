# dashboard/epc_dashboard.py

import streamlit as st
import requests

API_URL = "http://localhost:8000/predict_epc_rating"

st.set_page_config(page_title="EPC Search & Rating", page_icon="🏠")
st.title("🔍 Search EPC Rating by Address or LMK_KEY")

api_key = st.text_input("🔑 Enter API Key", type="password")

if api_key:
    search_type = st.radio("Search by:", ["Address", "LMK_KEY"])

    user_input = ""
    if search_type == "Address":
        user_input = st.text_input("🏡 Enter part of the property address")
    else:
        user_input = st.text_input("🆔 Enter LMK_KEY")
    
    if st.button("🚀 Predict Potential Rating"):
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "address": user_input if search_type == "Address" else None,
            "lmk_key": user_input if search_type == "LMK_KEY" else None
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()["predicted_potential_energy_rating"]
                rating_emoji = {
                    "A": "🌟",
                    "B": "👍",
                    "C": "😊",
                    "D": "😐",
                    "E": "😕",
                    "F": "😟",
                    "G": "🔴"
                }
                emoji = rating_emoji.get(result, "🏠")
                st.success(f"🔋 Predicted Potential Rating: {result} {emoji}")
            elif response.status_code == 403:
                st.error("❌ Invalid API Key.")
            elif response.status_code == 404:
                st.warning("⚠️ Property not found.")
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"⚠️ Failed to connect to the API: {e}")
else:
    st.warning("Please enter your API key.")
