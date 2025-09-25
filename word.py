# Streamlit Gemini Chatbot + Daily TOEIC Vocab (with CLI fallback & self-tests)
# ----------------------------------------------------------------------------
# What changed (bugfix):
# - Fix crash when Streamlit isn't installed (ModuleNotFoundError: streamlit)
#   -> Detect Streamlit at import-time. If missing, run a CLI fallback that:
#      * prints today's TOEIC words (deterministic by date, KST)
#      * optionally runs a minimal chat REPL if GEMINI_API_KEY + google-generativeai exist
#      * provides clear instructions to install Streamlit
# - Hardened chat streaming fallback (avoid unbound chat_obj on exceptions)
# - Added lightweight self-tests (run with: `python this_file.py --test` or RUN_TESTS=1)
# - Kept original Streamlit UI/behavior intact when Streamlit is available.
# ----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import time
import json
import random
import datetime as dt
import hashlib
from typing import List, Dict

# ----------------------------
# Optional UI dependency (Streamlit)
# ----------------------------
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except ModuleNotFoundError:
    st = None  # shim placeholder
    HAS_STREAMLIT = False

# ----------------------------
# Optional LLM SDK (google-generativeai)
# ----------------------------
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

# ----------------------------
# Configuration & Constants
# ----------------------------
MODEL_NAME = "gemini-2.5-flash"
APP_TITLE = "Gemini Chat + Daily TOEIC Trainer"
DAILY_WORD_COUNT = 12  # number of words to show per day

# Minimal seed TOEIC wordbank (extend as you like)
WORD_BANK: List[Dict] = [
    {"word": "allocate", "pos": "v.", "kr": "할당하다", "ex": "The manager allocated more resources to the project.", "ex_kr": "관리자는 그 프로젝트에 더 많은 자원을 할당했다."},
    {"word": "amendment", "pos": "n.", "kr": "수정, 개정", "ex": "The contract requires an amendment to extend the deadline.", "ex_kr": "마감 연장을 위해 계약서에 개정이 필요하다."},
    {"word": "appraisal", "pos": "n.", "kr": "평가", "ex": "Annual performance appraisals are scheduled for next week.", "ex_kr": "연간 성과 평가는 다음 주에 예정되어 있다."},
    {"word": "attain", "pos": "v.", "kr": "달성하다", "ex": "We attained our quarterly sales target.", "ex_kr": "우리는 분기 매출 목표를 달성했다."},
    {"word": "backlog", "pos": "n.", "kr": "미처리분, 밀린 일", "ex": "The team is working through a backlog of support tickets.", "ex_kr": "팀은 밀린 지원 티켓을 처리 중이다."},
    {"word": "benchmark", "pos": "n./v.", "kr": "기준, 기준으로 삼다", "ex": "We benchmarked our service against industry leaders.", "ex_kr": "우리는 업계 선도 기업을 기준으로 서비스를 비교했다."},
    {"word": "contingency", "pos": "n.", "kr": "우발 사태, 비상 대비", "ex": "Please prepare a contingency plan for potential delays.", "ex_kr": "잠재적 지연에 대비한 비상 계획을 준비해 주세요."},
    {"word": "deduct", "pos": "v.", "kr": "공제하다", "ex": "Taxes will be deducted from your paycheck.", "ex_kr": "세금은 급여에서 공제될 것이다."},
    {"word": "discrepancy", "pos": "n.", "kr": "불일치", "ex": "The audit found a discrepancy in the inventory count.", "ex_kr": "감사에서 재고 수량의 불일치가 발견되었다."},
    {"word": "feasible", "pos": "adj.", "kr": "실현 가능한", "ex": "The proposal is financially feasible.", "ex_kr": "그 제안은 재정적으로 실현 가능하다."},
    {"word": "incentive", "pos": "n.", "kr": "장려금, 유인책", "ex": "Employees received incentives for meeting deadlines.", "ex_kr": "직원들은 마감 준수에 대한 장려금을 받았다."},
    {"word": "liability", "pos": "n.", "kr": "책임, 부채", "ex": "The company has no liability for lost items.", "ex_kr": "회사는 분실물에 대한 책임이 없다."},
    {"word": "logistics", "pos": "n.", "kr": "물류, 운영 관리", "ex": "We need to finalize the event logistics.", "ex_kr": "행사 운영 계획을 마무리해야 한다."},
    {"word": "negotiate", "pos": "v.", "kr": "협상하다", "ex": "They negotiated better terms for the supplier.", "ex_kr": "그들은 공급업체와 더 나은 조건을 협상했다."},
    {"word": "overhead", "pos": "n.", "kr": "간접비", "ex": "Cutting overhead can improve profitability.", "ex_kr": "간접비를 줄이면 수익성이 개선될 수 있다."},
    {"word": "procurement", "pos": "n.", "kr": "조달", "ex": "The procurement team issued a request for proposals.", "ex_kr": "조달 팀이 제안요청서를 발행했다."},
    {"word": "redundant", "pos": "adj.", "kr": "불필요한, 중복의", "ex": "Some redundant processes were eliminated.", "ex_kr": "일부 중복된 프로세스가 제거되었다."},
    {"word": "reimburse", "pos": "v.", "kr": "상환하다, 변제하다", "ex": "Travel expenses will be reimbursed within a week.", "ex_kr": "여비는 일주일 내에 상환된다."},
    {"word": "retention", "pos": "n.", "kr": "유지, 보유", "ex": "Improving customer retention is our top priority.", "ex_kr": "고객 유지율 향상이 최우선 과제다."},
    {"word": "viable", "pos": "adj.", "kr": "실행 가능한", "ex": "Is the timeline viable for the team?", "ex_kr": "그 일정이 팀에 실행 가능한가요?"},
]

# ----------------------------
# Helper Functions
# ----------------------------

def _today_kst_date() -> dt.date:
    """Return today's date (system locale). Streamlit hosting often uses KST for the user here."""
    return dt.date.today()


def get_today_seed() -> int:
    """Deterministic daily seed based on date (e.g., 20250925)."""
    today = _today_kst_date()
    base = int(today.strftime("%Y%m%d"))
    h = int(hashlib.sha256(str(base).encode()).hexdigest(), 16) % (10 ** 8)
    return h


def pick_daily_words(bank: List[Dict], k: int) -> List[Dict]:
    rnd = random.Random(get_today_seed())
    if k >= len(bank):
        return bank.copy()
    return rnd.sample(bank, k)


def to_gemini_history(chat_history: List[Dict]) -> List[Dict]:
    """Convert our history [{role, content}] to Gemini format."""
    gh = []
    for msg in chat_history:
        role = "user" if msg.get("role") == "user" else "model"
        gh.append({"role": role, "parts": [msg.get("content", "")]})
    return gh


def _get_api_key_from_sources() -> str | None:
    """Return an API key from Streamlit secrets if available, else from env."""
    key = None
    if HAS_STREAMLIT and getattr(st, "secrets", None):
        # st.secrets supports key access like a dict
        key = st.secrets.get("GEMINI_API_KEY")
    if not key:
        key = os.environ.get("GEMINI_API_KEY")
    return key


# ----------------------------
# Streamlit-specific UI helpers
# ----------------------------

def _require_api_key_streamlit() -> None:
    """Streamlit-only guard for API key + SDK presence; stops app with a clear error if missing."""
    if not HAS_STREAMLIT:
        return
    api_key = _get_api_key_from_sources()
    if not api_key:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다. `.streamlit/secrets.toml` 또는 환경변수에 키를 추가하세요.")
        st.stop()
    if genai is None:
        st.error("`google-generativeai` 패키지가 설치되지 않았습니다. `pip install google-generativeai` 후 다시 실행하세요.")
        st.stop()
    genai.configure(api_key=api_key)


def _render_vocab_table_streamlit(words: List[Dict]) -> None:
    st.write("### 오늘의 TOEIC 핵심 단어")
    st.caption("매일 자동으로 새 목록이 선정됩니다 (로컬 날짜 기준). 다운로드로 기록해두세요!")
    rows = [{"단어": w["word"], "품사": w["pos"], "의미": w["kr"], "예문": w["ex"], "해석": w["ex_kr"]} for w in words]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # CSV download
    import io, csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["word","pos","kr","ex","ex_kr"])
    writer.writeheader()
    for w in words:
        writer.writerow(w)
    st.download_button("오늘의 단어 CSV 다운로드", buf.getvalue().encode("utf-8"), file_name="toeic_words_today.csv", mime="text/csv")


def _vocab_quiz_streamlit(words: List[Dict]) -> None:
    st.write("### 퀴즈: 뜻 고르기 (객관식)")
    with st.form("mc_quiz"):
        score = 0
        selections = {}
        rnd = random.Random(get_today_seed() + 7)
        meanings = [w["kr"] for w in words]
        for i, w in enumerate(words):
            pool = [m for m in meanings if m != w["kr"]]
            wrongs = rnd.sample(pool, k=min(3, max(0, len(pool))))
            opts = wrongs + [w["kr"]]
            rnd.shuffle(opts)
            choice = st.radio(
                label=f"{i+1}. '{w['word']}'의 의미는?",
                options=opts,
                key=f"mc_{i}",
                index=0,
                horizontal=False,
            )
            selections[i] = (choice, w["kr"])  # (chosen, answer)
        if st.form_submit_button("채점하기"):
            for _, (chosen, ans) in selections.items():
                if chosen == ans:
                    score += 1
            st.success(f"점수: {score} / {len(words)}")

    st.write("### 퀴즈: 예문 빈칸 채우기 (주관식)")
    with st.form("fill_quiz"):
        score2 = 0
        user_inputs = {}
        rnd2 = random.Random(get_today_seed() + 13)
        sample = rnd2.sample(words, k=min(5, len(words)))
        for i, w in enumerate(sample):
            blanked = w["ex"].replace(w["word"], "____") if w["word"] in w["ex"] else f"____ : {w['ex']}"
            ans = w["word"]
            val = st.text_input(f"{i+1}. {blanked}", key=f"fill_{i}")
            user_inputs[i] = (val.strip(), ans)
        if st.form_submit_button("채점하기"):
            for _, (typed, ans) in user_inputs.items():
                if typed.lower() == ans.lower():
                    score2 += 1
            st.success(f"점수: {score2} / {len(user_inputs)})")


def _init_chat_state_streamlit() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {role: "user"|"model", "content": str}


# ----------------------------
# Streamlit App (only if Streamlit is available)
# ----------------------------
if HAS_STREAMLIT:
    st.set_page_config(page_title=APP_TITLE, page_icon="📘", layout="wide")

    st.title("📘 Gemini Chat + Daily TOEIC Trainer")

    with st.sidebar:
        st.header("설정")
        st.markdown("**모델**: `gemini-2.5-flash`")
        st.caption("API 키는 .streamlit/secrets.toml 에 `GEMINI_API_KEY` 로 저장하거나 환경변수로 설정하세요.")
        st.divider()
        st.markdown("**대화 관리**")
        if st.button("대화 초기화"):
            st.session_state.chat_history = []
            st.success("대화를 초기화했습니다.")

    # Ensure key & init
    _require_api_key_streamlit()
    _init_chat_state_streamlit()

    # --- Columns: Chat | Vocab ---
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("💬 Gemini 상담형 챗봇")

        # Display history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

        # Chat input
        user_prompt = st.chat_input("메시지를 입력하세요…")
        if user_prompt:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("assistant"):
                placeholder = st.empty()
                buf: List[str] = []

                chat_obj = None
                try:
                    model = genai.GenerativeModel(MODEL_NAME)
                    chat_obj = model.start_chat(history=to_gemini_history(st.session_state.chat_history[:-1]))
                    stream = chat_obj.send_message(user_prompt, stream=True)
                    final_text = ""
                    for ev in stream:
                        chunk = getattr(ev, "text", None)
                        if chunk:
                            buf.append(chunk)
                            final_text += chunk
                            placeholder.markdown("".join(buf))
                    assistant_text = final_text.strip()
                except Exception:
                    try:
                        if chat_obj is None:
                            model = genai.GenerativeModel(MODEL_NAME)
                            chat_obj = model.start_chat(history=to_gemini_history(st.session_state.chat_history[:-1]))
                        resp = chat_obj.send_message(user_prompt)
                        assistant_text = getattr(resp, "text", "(응답 없음)")
                        placeholder.markdown(assistant_text)
                    except Exception as e:
                        assistant_text = f"오류: {e}"
                        placeholder.error(assistant_text)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

    with col2:
        st.subheader("🗓️ Daily TOEIC Trainer")
        today_words = pick_daily_words(WORD_BANK, DAILY_WORD_COUNT)
        _render_vocab_table_streamlit(today_words)
        st.divider()
        _vocab_quiz_streamlit(today_words)

    # ----------------------------
    # Footer / How-To
    # ----------------------------
    with st.expander("🔐 설정 방법 (필수)"):
        st.markdown(
            """
            1. 프로젝트 루트에 `.streamlit/secrets.toml` 파일을 만들고 아래처럼 저장:
               
               ```toml
               GEMINI_API_KEY = "YOUR_API_KEY"
               ```
               
            2. 라이브러리 설치:
               
               ```bash
               pip install streamlit google-generativeai
               ```
               
            3. 앱 실행:
               
            ```bash
            streamlit run streamlit_gemini_toeic_app.py
            ```
            """
        )

    with st.expander("🧩 팁: 단어장 확장하기"):
        st.markdown(
            """
            - 이 파일 상단의 `WORD_BANK` 리스트에 새 항목을 계속 추가하세요.
            - 각 항목은 `{word, pos, kr, ex, ex_kr}` 키를 포함합니다.
            - 더 큰 사전을 사용하려면 CSV/JSON을 읽어 `WORD_BANK` 를 대체해도 됩니다.
            - 날짜별 고정 시드를 사용하므로, 매일 새로운 랜덤 목록이 **일관되게** 생성됩니다.
            """
        )

# ----------------------------
# CLI fallback (when Streamlit is not installed)
# ----------------------------
else:
    def _cli_print_words(words: List[Dict]) -> None:
        print("\n[Daily TOEIC Words]")
        print("(Install Streamlit for full UI: pip install streamlit google-generativeai)\n")
        for w in words:
            print(f"- {w['word']} ({w['pos']}): {w['kr']}")
            print(f"  e.g., {w['ex']}")
        print()

    def _cli_chat_loop() -> None:
        api_key = _get_api_key_from_sources()
        if genai is None or not api_key:
            print("[Info] Chat disabled (missing `google-generativeai` or `GEMINI_API_KEY`).")
            print("       To enable: pip install google-generativeai && export GEMINI_API_KEY=...\n")
            return
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            chat = model.start_chat(history=[])
        except Exception as e:
            print(f"[Warn] Failed to initialize Gemini chat: {e}\n")
            return
        print("Type your message (or 'exit' to quit):")
        while True:
            try:
                user_in = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_in or user_in.lower() in {"exit", "quit"}:
                break
            try:
                resp = chat.send_message(user_in)
                print("Bot>", getattr(resp, "text", "(no text)"))
            except Exception as e:
                print("[Error]", e)

    def main_cli() -> None:
        print("[Running in CLI fallback mode — Streamlit not found]")
        print("Install UI deps with: pip install streamlit google-generativeai\n")
        words = pick_daily_words(WORD_BANK, DAILY_WORD_COUNT)
        _cli_print_words(words)
        _cli_chat_loop()

# ----------------------------
# Lightweight Self-Tests
# ----------------------------

def _run_tests() -> None:
    # Test 1: seed is deterministic (numeric, length ~8-digit modulo)
    s1 = get_today_seed()
    s2 = get_today_seed()
    assert isinstance(s1, int) and isinstance(s2, int), "Seed must be int"
    assert s1 == s2, "Seed must be stable within same date"

    # Test 2: daily words deterministic for same day & k
    sample1 = pick_daily_words(WORD_BANK, 5)
    sample2 = pick_daily_words(WORD_BANK, 5)
    assert sample1 == sample2 and len(sample1) == 5, "Daily sample must be deterministic and sized correctly"

    # Test 3: gemini history transform
    gh = to_gemini_history([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    assert gh[0]["role"] == "user" and gh[1]["role"] == "model", "Role mapping failed"
    assert gh[0]["parts"][0] == "hi" and gh[1]["parts"][0] == "yo", "Content mapping failed"

    # Test 4: CLI runs without Streamlit (does not raise)
    if not HAS_STREAMLIT:
        try:
            words = pick_daily_words(WORD_BANK, 3)
            assert isinstance(words, list) and len(words) == 3
        except Exception as e:
            raise AssertionError(f"CLI basic flow failed: {e}")

    print("All tests passed ✅")


if __name__ == "__main__":
    if "--test" in sys.argv or os.environ.get("RUN_TESTS") == "1":
        _run_tests()
        sys.exit(0)
    if not HAS_STREAMLIT:
        main_cli()
