import os
import io
import re
import json
import streamlit as st
import networkx as nx
from gtts import gTTS
from difflib import get_close_matches
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
import time
from pathlib import Path

# ==== LLM (LangChain) ====
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==== OpenAI STT ====
from openai import OpenAI

# ===============================
# 환경 변수 로드 (.env)
# ===============================
# load_dotenv()


def get_secret(key: str, default: str = ""):
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")
STT_MODEL = get_secret("STT_MODEL", "gpt-4o-mini-transcribe")
OPENAI_TTS_MODEL = get_secret("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = get_secret("OPENAI_TTS_VOICE", "alloy")

# ===============================
# 설정 스위치
# ===============================
USE_LLM_NORMALIZE = True  # LLM 보조 정규화 사용
USE_SPEECH_TO_TEXT = True  # 음성(STT) 사용


# ===============================
# 공통 유틸
# ===============================

st.set_page_config(
    page_title="남양주도시공사 AI 기반 실내 네비게이션",
    page_icon="🏢",
    layout="wide",  # ✅ 화면 폭 넓히기
    initial_sidebar_state="collapsed",
)

# 메인 컨테이너의 최대 폭 확장
st.markdown(
    """
<style>
/* 기본 block-container 폭을 넓힘 (원하는 값으로 조절) */
.block-container {max-width: 1200px;}
@media (min-width: 1600px){ .block-container {max-width: 1400px;} }
</style>
""",
    unsafe_allow_html=True,
)


def stream_markdown(text: str, chunk: int = 6, delay: float = 0.02):
    """
    채팅 말풍선 내부에서 '타자 효과'를 내는 간단 함수.
    현재 컨텍스트(예: with st.chat_message(...)) 안에서 호출해야 말풍선 안에 렌더됨.
    """
    placeholder = st.empty()
    buf = ""
    for i in range(0, len(text), chunk):
        buf += text[i : i + chunk]
        placeholder.markdown(buf)
        time.sleep(delay)


def fuzzy_match(user_text, poi_map):
    if not user_text:
        return None
    candidates = list(poi_map.keys())
    matches = get_close_matches(user_text, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def tts_advance(llm, text):
    if llm is None:
        return text
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 길 안내를 하기 위해 적어진 {text}에 대해 친절하게 번역하고 알려주는 네비게이션 입니다.

        - 규칙
        1. "→" 된 모양은 화살표라고 번역하지말고, 일반 자동차 네비게이션이 번역하듯이 자연스럽게 번역할 것.
        2. 전체적으로 문맥이 자연스러운지 한번 더 확인하고, 맞춤법 그리고 띄어쓰기 마지막으로 부드럽게 읽혀지는지 확인할 것.
        3. 문장은 전체적으로 친절할 것
        4. 요청한 문구에 없는 내용들은 함부로 지어내지 말 것.
        5. 남자와 여자가 구분되어 있는 샤워실, 화장실의 경우 확실하게 남자와 여자 구분지어서 잘 들릴 수 있도록 구성할 것.
        6. 엘리베이터가 연속으로 1층 -> 2층 -> 3층 이런식으로 이어질 때는 엘리베이터 1층~3층과 같이 한번에 안내할것.
        """
    )

    chain = prompt | llm | StrOutputParser()

    advanced_tts = chain.invoke({"text": text})

    return advanced_tts


def speak_openai(llm, client, text, filename="route.mp3"):
    out = Path(filename)
    with st.status("🔊 안내 음성 생성 중...", expanded=True) as status:
        status.update(label="📝 음성 대답을 위해 대답을 정리하는 중…")
        polished = tts_advance(llm, text)
        status.update(label="🎤 음성 합성 중…")

        # 👇 format 인자 제거!
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=polished,
        ) as resp:
            resp.stream_to_file(out)

        status.update(label="▶️ 재생합니다", state="complete")

    st.audio(str(out), format="audio/mp3", autoplay=True)


# ===============================
# 데이터 로드 및 그래프 구성
# ===============================
with open("indoor_graph.json", "r", encoding="utf-8") as f:
    graph_data = json.load(f)["buildings"][0]

G = nx.Graph()
for e in graph_data["edges"]:
    G.add_edge(e["from"], e["to"], via=e["via"])

poi_map = {}
for floor in graph_data["floors"]:
    for p in floor["pois"]:
        name = p["name_ko"]
        poi_map[name] = p["id"]
        poi_map[name.replace(" ", "")] = p["id"]  # 공백 제거 키도 매핑


def id_to_name(node_id):
    return next(
        p["name_ko"]
        for floor in graph_data["floors"]
        for p in floor["pois"]
        if p["id"] == node_id
    )


def safe_find_route(start_name, end_name):
    if start_name not in poi_map:
        fm = fuzzy_match(start_name.replace(" ", ""), poi_map)
        if fm:
            start_name = fm
        else:
            return None, f"출발지 '{start_name}' 는 등록된 위치가 아닙니다."

    if end_name not in poi_map:
        fm = fuzzy_match(end_name.replace(" ", ""), poi_map)
        if fm:
            end_name = fm
        else:
            return None, f"도착지 '{end_name}' 는 등록된 위치가 아닙니다."

    try:
        path = nx.shortest_path(G, poi_map[start_name], poi_map[end_name])
        route = [id_to_name(node) for node in path]
        return route, None
    except nx.NetworkXNoPath:
        return None, f"{start_name}에서 {end_name}까지 경로를 찾을 수 없습니다."


# ===============================
# 간단 POI 추출기 (LLM 없이)
# ===============================
def extract_poi_candidate(text):
    if text in poi_map:
        return text
    t0 = text.replace(" ", "")
    if t0 in poi_map:
        return t0
    for token in re.split(r"[ ,.!?/~·()\[\]{}\"'“”’]+", text):
        token = token.strip()
        if not token:
            continue
        if token in poi_map:
            return token
        t1 = token.replace(" ", "")
        if t1 in poi_map:
            return t1
        fm = fuzzy_match(t1, poi_map)
        if fm:
            return fm
    return fuzzy_match(text.replace(" ", ""), poi_map)


# ===============================
# LLM 인스턴스
# ===============================
llm = None
if USE_LLM_NORMALIZE:
    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY가 설정되지 않았습니다. LLM 정규화를 비활성화합니다.")
        USE_LLM_NORMALIZE = False
    else:
        llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)


# ===============================
# (보조) LLM 정규화
# ===============================
def normalize_with_llm_fallback(text):
    cand = extract_poi_candidate(text)
    if cand:
        return cand
    if not USE_LLM_NORMALIZE or llm is None:
        return "알 수 없음"

    poi_list = "\n".join(f"- {p}" for p in poi_map.keys())
    prompt = ChatPromptTemplate.from_template(
        """
        사용자가 "{user_text}" 라고 말했습니다.

        실제 후보 POI 목록:
        {poi_list}

        규칙:
        - 오타/띄어쓰기/발음이 비슷해도 '후보 목록' 중 가장 알맞은 '한 항목'만 정확히 출력하세요.
        - 후보 목록에 없는 표현은 절대 내지 마세요.
        - 정말로 매칭이 어렵다면 "알 수 없음"만 출력하세요.
        - 출력은 순수 텍스트로만 (따옴표/마크다운/JSON 금지).
        """
    )
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"user_text": text, "poi_list": poi_list}).strip()
        if response in poi_map:
            return response
        fm = fuzzy_match(response.replace(" ", ""), poi_map)
        return fm or "알 수 없음"
    except Exception:
        return "알 수 없음"


# ===============================
# 음성(STT) - OpenAI
# ===============================
oa_client = None
if USE_SPEECH_TO_TEXT:
    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY가 설정되지 않아 STT가 비활성화됩니다.")
        USE_SPEECH_TO_TEXT = False
    else:
        oa_client = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    if not USE_SPEECH_TO_TEXT or oa_client is None:
        return ""
    bio = io.BytesIO(wav_bytes)
    bio.name = "speech.wav"
    result = oa_client.audio.transcriptions.create(
        model=STT_MODEL,
        file=bio,
        response_format="text",
        language="ko",
    )
    return (result or "").strip()


# ===============================
# UI
# ===============================

st.markdown(
    """
<style>
/* 제목 줄바꿈 & 단어깨짐 제어 */
h1, .stMarkdown h1 {
  white-space: normal !important;   /* 한 줄 고정 방지 */
  overflow-wrap: anywhere;          /* 너무 긴 단어도 적절히 줄바꿈 */
  word-break: keep-all;             /* 한국어 음절 단위 깨짐 방지 */
  text-wrap: balance;               /* (지원 브라우저) 보기 좋게 줄 균형 */
}
</style>
""",
    unsafe_allow_html=True,
)

# st.title 대신:
st.markdown("# 🏢 남양주도시공사 AI 기반 실내 네비게이션")

st.caption(
    "1. 현재 위치를 물어본 뒤, 목적지를 묻는 순서인 안내 절차에 맞게 따라주세요."
)
st.caption("2. 해당 건물 안내가 아닌 질문을 할 경우, 답변을 드릴 수 없습니다.")

# 음성 입력
st.subheader("🎙️ 음성 입력")
audio = mic_recorder(
    start_prompt="눌러서 녹음 시작",
    stop_prompt="눌러서 녹음 정지",
    just_once=False,
    use_container_width=True,
    format="wav",
)

spoken_text = None
if USE_SPEECH_TO_TEXT and audio and "bytes" in audio and audio["bytes"]:
    st.audio(audio["bytes"], format="audio/wav")
    with st.spinner("음성 인식 중…"):
        try:
            spoken_text = transcribe_wav_bytes(audio["bytes"])
            if spoken_text:
                st.success(f"📝 인식 결과: {spoken_text}")
        except Exception as e:
            st.error(f"음성 인식 실패: {e}")
            spoken_text = None

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "phase" not in st.session_state:
    st.session_state.phase = "ask_start"  # ask_start -> ask_end
if "current_start" not in st.session_state:
    st.session_state.current_start = None
if "current_end" not in st.session_state:
    st.session_state.current_end = None
if "asked_start_prompt" not in st.session_state:
    st.session_state.asked_start_prompt = False


# 단계별 프롬프트
def phase_prompt():
    if st.session_state.phase == "ask_start":
        return "현재 어디에 계신가요? (예: 정문 로비, 체육관 입구 등)"
    elif st.session_state.phase == "ask_end":
        return "어디로 가고 싶으신가요? (목적지를 알려주세요)"
    return "메시지를 입력하세요."


# 기존 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 어시스턴트 인사 1회 출력
if st.session_state.phase == "ask_start" and not st.session_state.asked_start_prompt:
    start_prompt = "현재 어디에 계신가요? 근처에 보이는 시설물이나 계신 위치를 자세히 말씀해 주세요. (예: 1층 안내 데스크)"
    st.session_state.messages.append({"role": "assistant", "content": start_prompt})
    with st.chat_message("assistant"):
        stream_markdown(start_prompt)
    st.session_state.asked_start_prompt = True

# 입력창
user_text_input = st.chat_input(
    "질문에 답을 여기에 입력해 주세요."
)  # st.chat_input(phase_prompt())
user_input = spoken_text or user_text_input

# 입력 처리
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ========= 1단계: 현재 위치 =========
    if st.session_state.phase == "ask_start":
        start_candidate = normalize_with_llm_fallback(user_input)
        if start_candidate and start_candidate != "알 수 없음":
            st.session_state.current_start = start_candidate
            st.session_state.phase = "ask_end"
            st.session_state.asked_start_prompt = False  # 다음 라운드 인사 리셋

            reply = (
                f"출발지를 '{start_candidate}'로 설정했어요. 이제 목적지를 알려주세요."
            )
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                stream_markdown(reply)

        else:
            reply = "죄송해요, 현재 위치를 잘 못 들었어요. 등록된 위치명으로 다시 말씀해 주세요. (예: 정문 로비)"
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                stream_markdown(reply)

    # ========= 2단계: 목적지 =========
    elif st.session_state.phase == "ask_end":
        end_candidate = normalize_with_llm_fallback(user_input)
        if end_candidate and end_candidate != "알 수 없음":
            st.session_state.current_end = end_candidate

            start = st.session_state.current_start
            end = st.session_state.current_end

            route, error = safe_find_route(start, end)
            if error:
                reply = error
            else:
                route_text = " → ".join(route)
                reply = (
                    f"{start}에서 출발하여 {end}까지 이동 경로는 {route_text} 입니다."
                )

            # 초기화 후 ask_start로 복귀
            st.session_state.phase = "ask_start"
            st.session_state.current_start = None
            st.session_state.current_end = None
            st.session_state.asked_start_prompt = False

            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                stream_markdown(reply)
            if "경로는" in reply:
                speak_openai(llm, oa_client, reply)

        else:
            reply = "죄송해요, 목적지를 잘 못 들었어요. 등록된 목적지로 다시 말씀해 주세요. (예: 화장실 남(1층))"
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                stream_markdown(reply)
