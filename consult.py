import os
import io
import datetime as dt
import random
import csv
import streamlit as st
from google import generativeai as genai

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="마인드셋 코치 챗봇 (Gemini 2.5 Flash)",
    page_icon="💬",
    layout="centered",
)

# 일정/시간
KST = dt.timezone(dt.timedelta(hours=9))
def now_kst() -> dt.datetime:
    return dt.datetime.now(KST)

def local_date_str(d: dt.datetime | None = None) -> str:
    d = d or now_kst()
    return d.strftime("%Y-%m-%d")

TODAY = local_date_str()

# 안전 고지 (정신건강 관련)
SAFETY_NOTICE = (
    "이 챗봇은 동기부여/학습 마인드셋을 돕는 도구이며 의료적 진단/치료를 대체하지 않습니다. "
    "자/타해 위험이 있거나 심각한 정서적 고통이 지속되면 전문기관에 즉시 상담하세요. "
    "한국생명의전화 1588-9191, 정신건강위기상담전화 1577-0199."
)

# -----------------------------
# API 설정 (secrets로만)
# -----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("🔑 API 키가 없습니다. .streamlit/secrets.toml에 GEMINI_API_KEY를 설정하세요.")
    st.stop()

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("Gemini 설정 중 오류가 발생했습니다. secrets 구성 및 키를 다시 확인하세요.")
    st.stop()

# -----------------------------
# 멘탈 코치 페르소나(시스템 지시문)
# -----------------------------
PERSONA = """
당신은 친절하고 실용적인 학습/마인드셋 코치입니다.
원칙:
- 공감 → 구체적 피드백 → 아주 작은 다음 행동(Next step) 제안(1~3개)을 한국어로.
- 과장/정신의학적 진단 금지. 필요 시 전문기관 안내.
- 대학 1학년 CS 전공자에게 맞춘 생산적 습관/학습 루틴/시간관리/감정 라벨링을 돕기.
- 톤: 따뜻하고 담백, 과한 칭찬·설교 금지.
출력 형식(가능한 한 간결):
1) 요약: (사용자 상황 한 줄 요약)
2) 코칭: (핵심 팁 2~4개, 불릿)
3) 오늘의 한 걸음: (실행 1~3개, 체크박스 이모지 포함)
"""

MODEL_NAME = "gemini-2.5-flash"
MAX_HISTORY = 30  # 대화 길이 제한

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 오늘도 차근차근 같이 가봅시다. 무엇이든 편하게 적어주세요."}
    ]

if "mood_record" not in st.session_state:
    st.session_state.mood_record = {}  # { "YYYY-MM-DD": {"mood": int, "note": str} }

if "daily_done" not in st.session_state:
    st.session_state.daily_done = set()  # {"YYYY-MM-DD"} : 오늘 긍정 멘트 생성 플래그

if "display_name" not in st.session_state:
    st.session_state.display_name = "친구"

# -----------------------------
# 도우미 함수
# -----------------------------
def is_morning() -> bool:
    hour = now_kst().hour
    return 5 <= hour <= 11

def daily_positive_lines(name_hint: str = "친구") -> list[str]:
    """날짜 고정 시드로 매일 같은 긍정 멘트."""
    seed = int(TODAY.replace("-", ""))
    random.seed(seed)
    templates = [
        f"{name_hint}, 오늘은 '완벽'이 아니라 '전진'이면 충분해요. 1%만 성장해봅시다.",
        "딱 25분만 집중 + 5분 휴식(포모도로) 2세트면 탄성이 붙어요.",
        "비교 대신 기록: 어제의 나와 오늘의 나만 비교해요.",
        "뇌는 시작하면 따라옵니다. 2분만 착수 규칙으로 시동을 걸어봐요.",
        "불안은 행동으로만 줄어듭니다. 너무 작아 보이는 일부터 체크 ✔",
    ]
    k = random.choice([2, 3])
    return random.sample(templates, k=k)

def render_history():
    for m in st.session_state.messages:
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["content"])

def convert_to_gemini_history(messages: list[dict]) -> list[dict]:
    """Streamlit 메시지를 Gemini 대화 이력 스키마로 변환."""
    hist = []
    for m in messages[-MAX_HISTORY:]:
        role = "user" if m["role"] == "user" else "model"
        hist.append({"role": role, "parts": [m["content"]]})
    return hist

def stream_gemini_reply(prompt: str, mood: int | None, mood_note: str | None) -> str:
    """스트리밍 응답(가능하면), 실패 시 일반 응답."""
    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=PERSONA
    )
    # 컨텍스트: 최근 대화 + 오늘 기분
    context_prefix = ""
    if mood is not None:
        context_prefix += f"[오늘의 기분: {mood}/10]\n"
    if mood_note:
        context_prefix += f"[메모: {mood_note}]\n"
    full_user_prompt = context_prefix + prompt

    try:
        chat = model.start_chat(history=convert_to_gemini_history(st.session_state.messages))
        with st.chat_message("assistant"):
            placeholder = st.empty()
            acc = ""
            try:
                for chunk in chat.send_message(full_user_prompt, stream=True):
                    if getattr(chunk, "text", None):
                        acc += chunk.text
                        placeholder.markdown(acc)
            except Exception:
                # 스트리밍 중간 실패 → 지금까지 출력, 남은 건 일반 요청으로 보강
                pass

            if acc.strip():
                return acc

        # 스트리밍에서 텍스트를 못 받았을 때 일반 모드 재시도
        resp = chat.send_message(full_user_prompt)
        text = getattr(resp, "text", None) or "(빈 응답)"
        with st.chat_message("assistant"):
            st.markdown(text)
        return text

    except Exception:
        try:
            resp = model.generate_content(full_user_prompt)
            text = resp.text or "(빈 응답)"
            with st.chat_message("assistant"):
                st.markdown(text)
            return text
        except Exception as ee:
            with st.chat_message("assistant"):
                st.error("응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
            return f"(오류) {ee}"

def trim_history():
    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

def export_mood_csv() -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "mood", "note"])
    for day, rec in sorted(st.session_state.mood_record.items()):
        writer.writerow([day, rec.get("mood", ""), rec.get("note", "")])
    return output.getvalue().encode("utf-8-sig")  # Excel 호환

# -----------------------------
# 사이드바: 상태/설정
# -----------------------------
with st.sidebar:
    st.subheader("🔐 설정 & 상태")
    st.caption("API 키는 `st.secrets`로만 사용합니다.")
    st.markdown(f"**모델**: `{MODEL_NAME}`")
    st.info(SAFETY_NOTICE)

    st.markdown("---")
    st.subheader("👤 사용자 설정")
    st.session_state.display_name = st.text_input("이름/별칭", value=st.session_state.display_name).strip() or "친구"

    st.markdown("---")
    st.subheader("🧠 오늘의 기분 체크")
    saved = st.session_state.mood_record.get(TODAY, {})
    mood = st.slider("기분(0=최저, 10=최고)", 0, 10, value=int(saved.get("mood", 6)))
    mood_note = st.text_input("한 줄 메모(선택)", value=saved.get("note", ""))

    cols_sb = st.columns(2)
    if cols_sb[0].button("기분 저장/업데이트"):
        st.session_state.mood_record[TODAY] = {"mood": mood, "note": mood_note}
        st.success("오늘 기분을 저장했습니다.")
    st.download_button(
        "기분 기록 CSV 다운로드",
        data=export_mood_csv(),
        file_name="mood_record.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🧹 관리")
    if st.button("대화 내역 초기화"):
        st.session_state.messages = [
            {"role": "assistant", "content": "대화 내역을 초기화했어요. 무엇이든 편하게 적어주세요."}
        ]
        st.success("초기화 완료!")

# -----------------------------
# 본문: 헤더 & 아침 긍정 멘트
# -----------------------------
st.title("💬 마인드셋 코치 챗봇 — Gemini 2.5 Flash")
st.caption("CS 새내기용 동기부여/학습 루틴 코칭 + 일반 Q&A. (스트리밍 응답 지원)")

if is_morning() and TODAY not in st.session_state.daily_done:
    with st.expander("☀️ 오늘의 아침 긍정 멘트", expanded=True):
        st.write("\n".join(f"- {line}" for line in daily_positive_lines(name_hint=st.session_state.display_name)))
        st.session_state.daily_done.add(TODAY)
else:
    with st.expander("☀️ 오늘의 아침 긍정 멘트", expanded=False):
        st.write("\n".join(f"- {line}" for line in daily_positive_lines(name_hint=st.session_state.display_name)))

# -----------------------------
# 빠른 코칭 카드: (클릭 시 자동 프롬프트)
# -----------------------------
cols = st.columns(3)
quick_prompts = {
    "25분 포모도로 계획": "오늘 해야 할 일을 3개로 압축하고 25분×2세트 계획을 짜줘. 휴식활동도 제안해줘.",
    "불안 ↓ 즉시 행동": "불안이 커져서 미루는 중이야. 지금 5분 안에 가능한 초소형 행동 3가지만 정해줘.",
    "시험 D-7 로드맵": "일주일 후 시험 대비 로드맵을 과목별 체크리스트 형식으로 만들어줘. 난이도는 대학교 1학년 CS 기준."
}
for i, (label, qp) in enumerate(quick_prompts.items()):
    if cols[i].button(label, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": qp})
        answer = stream_gemini_reply(
            qp,
            mood=st.session_state.mood_record.get(TODAY, {}).get("mood"),
            mood_note=st.session_state.mood_record.get(TODAY, {}).get("note"),
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        trim_history()

# -----------------------------
# 대화 히스토리 렌더링
# -----------------------------
st.markdown("### 대화")
render_history()

# -----------------------------
# 채팅 입력
# -----------------------------
user_text = st.chat_input("무엇이든 적어주세요. 예) 오늘 공부 계획 세워줘 / 마음이 무겁다 ...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    ans = stream_gemini_reply(
        user_text,
        mood=st.session_state.mood_record.get(TODAY, {}).get("mood"),
        mood_note=st.session_state.mood_record.get(TODAY, {}).get("note"),
    )
    st.session_state.messages.append({"role": "assistant", "content": ans})
    trim_history()
