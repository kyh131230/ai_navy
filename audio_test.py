from gtts import gTTS
import streamlit as st
import os

if st.button("경로 안내 실행"):
    text = "1층 로비에서 2층 화장실까지 경로를 안내합니다."
    tts = gTTS(text=text, lang="ko")
    tts.save("route.mp3")
    st.audio("route.mp3", format="audio/mp3", autoplay=True)
