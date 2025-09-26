# app.py — MarketLens (All-asset Live Context + SerpAPI-only News Evidence)

import os
import re
import time
import json
import tempfile
import datetime
from typing import List, Dict, Optional, Tuple

import requests
import streamlit as st

# LangChain community / core
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===============================
# Branding / Constants
# ===============================
APP_NAME = "MarketLens"
APP_TAGLINE = "투자 리서치 비서"
DISCLAIMER = "본 서비스의 정보는 참고용이며, 투자 결정 및 그 결과에 대한 책임은 이용자 본인에게 있습니다."

# cmc api key : 49527fc6-f1ce-4549-8cbe-f8f42db593ee

# ===============================
# UI helpers
# ===============================
def section_divider():
    st.markdown("---")

def info_card(title: str, body: str):
    st.markdown(
        f"""
<div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin:6px 0;background:#fafafa">
  <div style="font-weight:600">{title}</div>
  <div style="font-size:14px;line-height:1.6;margin-top:6px">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# ===============================
# Market Data: CMC (optional) → CoinGecko fallback
# ===============================
def get_price_cmc(symbol: str, cmc_key: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    CoinMarketCap Pro API (선택). 성공 시 (price_usd, percent_change_24h) 반환.
    """
    if not cmc_key:
        return None, None
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": cmc_key}
    params = {"symbol": symbol.upper(), "convert": "USD"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        # j["data"]는 심볼 키에 리스트가 들어올 수 있음
        data = j.get("data", {})
        arr = data.get(symbol.upper(), [])
        if not arr:
            return None, None
        item = arr[0]
        quote = item.get("quote", {}).get("USD", {})
        price = quote.get("price")
        chg = quote.get("percent_change_24h")
        return (float(price) if price is not None else None,
                float(chg) if chg is not None else None)
    except Exception:
        return None, None

def cg_search_symbol(symbol: str) -> Optional[str]:
    """
    CoinGecko 검색 → coin_id 반환. 심볼 완전일치 우선, 없으면 첫 후보.
    """
    try:
        r = requests.get("https://api.coingecko.com/api/v3/search", params={"query": symbol}, timeout=10)
        r.raise_for_status()
        j = r.json()
        coins = j.get("coins", [])
        exact = [c for c in coins if c.get("symbol", "").lower() == symbol.lower()]
        if exact:
            return exact[0].get("id")
        return coins[0].get("id") if coins else None
    except Exception:
        return None

def get_price_coingecko(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """
    CoinGecko: 심볼 → id 검색 → simple/price.
    반환: (price_usd, percent_change_24h)
    """
    coin_id = cg_search_symbol(symbol)
    if not coin_id:
        return None, None
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd", "include_24hr_change": "true"},
            timeout=10,
        )
        r.raise_for_status()
        j = r.json()
        data = j.get(coin_id, {})
        price = data.get("usd")
        chg = data.get("usd_24h_change")
        return (float(price) if price is not None else None,
                float(chg) if chg is not None else None)
    except Exception:
        return None, None

def get_live_price(symbol: str, cmc_key: Optional[str]) -> Tuple[Optional[float], Optional[float], str]:
    """
    우선 CMC → 실패 시 CoinGecko.
    반환: (price, chg24h, source_str)
    """
    px, chg = get_price_cmc(symbol, cmc_key)
    if px is not None:
        return px, chg, "CoinMarketCap"
    px, chg = get_price_coingecko(symbol)
    if px is not None:
        return px, chg, "CoinGecko"
    return None, None, "N/A"

def get_ohlc_binance(symbol_pair: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> List[Dict]:
    """
    Binance klines: 일부 자산은 미상장일 수 있으므로 실패해도 무시.
    """
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params={"symbol": symbol_pair, "interval": interval, "limit": limit}, timeout=10)
        r.raise_for_status()
        arr = r.json()
        ohlc = [
            {"t": k[0], "o": float(k[1]), "h": float(k[2]), "l": float(k[3]), "c": float(k[4]), "v": float(k[5])}
            for k in arr
        ]
        return ohlc
    except Exception:
        return []

def make_live_context(symbol: str, cmc_key: Optional[str], binance_pair: Optional[str], lookback: int = 50) -> Tuple[str, Optional[float]]:
    """
    자산 전용 현재가·최근 고저 컨텍스트 문자열과 현재가 반환.
    """
    px, chg, source = get_live_price(symbol, cmc_key)
    parts = []
    if px is not None:
        parts.append(f"{symbol.upper()} 현물 추정가(${source}): ${px:,.0f} (24h {chg:+.2f}%)" if chg is not None else f"{symbol.upper()} 현물 추정가(${source}): ${px:,.0f}")
    recent_high = recent_low = None
    if binance_pair:
        ohlc = get_ohlc_binance(binance_pair, "1h", max(lookback, 50))
        if ohlc:
            highs = [k["h"] for k in ohlc[-lookback:]]
            lows = [k["l"] for k in ohlc[-lookback:]]
            if highs and lows:
                recent_high = max(highs); recent_low = min(lows)
                parts.append(f"최근 {lookback}캔들 고가: ${recent_high:,.0f}, 저가: ${recent_low:,.0f}")
    if not parts:
        return "시장 컨텍스트: 불러오기 실패", None
    parts.append("지지·저항·리스크 언급 시 위 컨텍스트를 우선 적용(오래된 레벨 금지)")
    return " / ".join(parts), px

# ===============================
# SerpAPI: 항상 뉴스 근거로 사용
# ===============================
def make_web_search_tool() -> Tool:
    search = SerpAPIWrapper()

    def run_with_source(query: str, time_scope: str = "week") -> str:
        try:
            # 최신/오늘 포함 시 day, 기본 week
            lower_q = query.lower()
            scope = "day" if any(k in lower_q for k in ["today", "latest", "current", "오늘", "최신", "현재"]) else time_scope
            if hasattr(search, "params"):
                search.params["time_period"] = scope
            results = search.results(query)
            organic = results.get("organic_results", [])
        except Exception as e:
            return f"웹 검색 오류: {e}"

        formatted = []
        for r in organic[:5]:
            title = (r.get("title") or "").strip()
            link = r.get("link") or ""
            source = r.get("source") or ""
            snippet = r.get("snippet") or ""
            if link:
                formatted.append(f"- [{title}]({link}) ({source})\n  {snippet}")
            else:
                formatted.append(f"- {title} (출처: {source})\n  {snippet}")
        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."

    return Tool(
        name="web_search",
        func=run_with_source,
        description="SerpAPI 기반 구글 뉴스 검색(제목/링크/출처/요약). 모든 뉴스·거시 코멘트는 반드시 이 결과에 근거합니다."
    )

def collect_news_evidence(tool: Tool, queries: List[str], time_scope: str = "week") -> str:
    """
    여러 쿼리를 검색하고 상위 결과를 마크다운으로 합칩니다.
    """
    blocks = []
    for q in queries:
        text = tool.func(q, time_scope)
        if text and "검색 결과가 없습니다." not in text and "웹 검색 오류" not in text:
            blocks.append(f"### {q}\n{text}")
    return "\n\n".join(blocks) if blocks else "증거 없음"

# ===============================
# PDF Retriever Tool
# ===============================
def build_pdf_retriever_tool(files: List) -> Optional[Tool]:
    if not files:
        return None
    all_docs, tmp_paths = [], []
    try:
        for uf in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.read())
                tmp_paths.append(tmp.name)
        for p in tmp_paths:
            loader = PyPDFLoader(p)
            docs = loader.load()
            all_docs.extend(docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(all_docs)
        vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        retriever = vector.as_retriever(search_kwargs={"k": 6})
        return create_retriever_tool(
            retriever,
            name="pdf_search",
            description="업로드한 리서치/PDF 문서에서 관련 내용을 우선 검색합니다."
        )
    finally:
        for p in tmp_paths:
            try: os.remove(p)
            except Exception: pass

# ===============================
# Agent & Prompt (뉴스는 SerpAPI 근거만 사용)
# ===============================
def build_agent(tools: List[Tool], model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> AgentExecutor:
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    system_text = (
        "당신은 투자 리서치 비서입니다. 반드시 한국어로, 전문적이고 차분한 톤으로 답변합니다. "
        "가격·레벨·지표처럼 시간 민감도가 높은 항목은 '현재 시장 컨텍스트'를 우선합니다. "
        "뉴스/거시 코멘트는 반드시 `web_search`(SerpAPI) 결과에만 근거하여 작성하며, "
        "출처가 없거나 상충하면 해당 내용은 '언급 생략'합니다. 추측 금지. "
        "PDF 정보는 정적 참고자료로만 사용하고, 숫자 레벨은 최신 시세에 맞춰 재해석합니다. "
        "답변은 핵심 요약 → 근거(출처/계산 근거) → 리스크/한계 순으로 간결히 정리하고, 확정 어조를 피합니다. "
        f"마지막 줄에 디스클레이머를 포함: '{DISCLAIMER}'"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("placeholder", "{chat_history}"),
            ("human", "{input}\n\n[뉴스 증거]\n{news_evidence}\n\n지시: 뉴스/거시 코멘트는 위 증거에만 근거하여 작성. 출처 불명/상충 시 '언급 생략'."),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

def agent_respond(executor: AgentExecutor, user_input: str, history: ChatMessageHistory, news_evidence_md: str) -> str:
    try:
        res = executor({"input": user_input, "chat_history": history.messages, "news_evidence": news_evidence_md})
        return res.get("output", "").strip()
    except Exception as e:
        return f"에이전트 실행 오류: {e}"

# ===============================
# Scenario Builder with News Evidence
# ===============================
SCENARIO_PROMPT = """다음 자산에 대한 포지션 시나리오를 작성합니다.

[시장 컨텍스트]
{live_context}

[뉴스 증거(SerpAPI 결과)]
{news_evidence}

[사용자 질문/맥락]
{question}

규칙:
- 가격/지지/저항/리스크는 '시장 컨텍스트'의 현재가·최근 고저를 우선 적용
- 뉴스/거시 코멘트는 '뉴스 증거'에서 확인된 내용만 사용. 확인되지 않으면 '언급 생략'
- 확정 어조 금지. '~할 수 있다', '~로 보인다' 등 여지 남김
- 간결한 불릿 중심, 중복 회피

요청 출력(마크다운):
1) 롱을 고려하는 이유 (3~5개)
2) 숏을 고려하는 이유 (3~5개)
3) 관망을 고려하는 이유 (3~5개)
4) 핵심 레벨(지지/저항) 3~5개 — 반드시 현재가 기준으로 재계산/정리
5) 리스크 관리 포인트(손절/분할/레버리지 주의) 3~5개
마지막 줄에 디스클레이머 포함: {disclaimer}
"""

def build_scenario_with_news(llm: ChatOpenAI, live_context: str, news_evidence_md: str, question: str, disclaimer: str) -> str:
    prompt = SCENARIO_PROMPT.format(live_context=live_context, news_evidence=news_evidence_md, question=question, disclaimer=disclaimer)
    resp = llm.invoke(prompt)
    return resp.content

# ===============================
# Recency Guard
# ===============================
def price_sanity_check(answer_text: str, live_px: Optional[float], tolerance: float = 0.5) -> str:
    if not live_px:
        return answer_text
    nums = []
    for token in re.findall(r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)", answer_text):
        try:
            n = float(token.replace(",", ""))
            if n > 0:
                nums.append(n)
        except Exception:
            pass
    bad = [n for n in nums if (n < live_px * (1 - tolerance) or n > live_px * (1 + tolerance))]
    if len(bad) >= 3:
        warn = "\n\n[검증 메모] 일부 가격 레벨이 현재가와 괴리가 커 최신성이 떨어질 수 있습니다. 최근 캔들을 기준으로 재검토가 필요해 보입니다."
        return answer_text + warn
    return answer_text

# ===============================
# Report Builder
# ===============================
def build_research_report(title: str, items: Dict) -> str:
    lines = [
        f"# {title}",
        f"- 생성 일시: {items.get('date', '')}",
        "",
        "## 개요",
        items.get("summary", ""),
        "",
        "## 뉴스 요약",
        items.get("news", ""),
        "",
        "## Q&A 요약",
        items.get("qa", ""),
        "",
        "## 포지션 시나리오",
        items.get("scenario", ""),
        "",
        f"> {DISCLAIMER}",
    ]
    return "\n".join(lines)

# ===============================
# Streamlit App
# ===============================
def main():
    st.set_page_config(page_title=f"{APP_NAME} — {APP_TAGLINE}", layout="wide")

    # Header
    c1, c2 = st.columns([1, 3])
    with c1:
        if os.path.exists("./marketlens_logo.png"):
            st.image("./marketlens_logo.png", use_container_width=True)
        else:
            st.markdown(f"### {APP_NAME}")
            st.caption(APP_TAGLINE)
    with c2:
        st.markdown(
            f"**{APP_NAME}**는 PDF 리서치 + 최신 뉴스 증거(SerpAPI) + 실시간 시세 컨텍스트를 결합한 투자 서포트 챗봇입니다."
        )

    section_divider()

    # Sidebar
    with st.sidebar:
        st.subheader("환경 설정")
        openai_key = st.text_input("OpenAI API 키", type="password")
        serp_key = st.text_input("SerpAPI API 키", type="password")
        cmc_key = st.text_input("CoinMarketCap API 키 (선택)", type="password")
        model_name = st.selectbox("LLM 모델", ["gpt-4o-mini", "gpt-4o"], index=0)
        temperature = st.slider("창의성(temperature)", 0.0, 1.0, 0.2, 0.1)

        st.markdown("---")
        st.subheader("대상 자산")
        symbol = st.text_input("심볼 (예: BTC, ETH, SOL)", value="BTC").upper().strip()
        binance_pair = st.text_input("Binance 심볼(선택, 예: BTCUSDT — 미상장 자산은 비워두기)", value="BTCUSDT").strip()

        st.markdown("---")
        st.subheader("PDF 라이브러리")
        pdf_files = st.file_uploader("리서치/리포트 PDF 업로드 (복수 선택 가능)", type=["pdf"], accept_multiple_files=True)

        st.markdown("---")
        st.caption(DISCLAIMER)

    if not openai_key or not serp_key:
        st.warning("좌측에서 OpenAI / SerpAPI 키를 입력하세요.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["SERPAPI_API_KEY"] = serp_key

    # Session
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = ChatMessageHistory()
    if "qa_cache" not in st.session_state:
        st.session_state["qa_cache"] = []
    if "last_news" not in st.session_state:
        st.session_state["last_news"] = ""

    # Tools
    tools: List[Tool] = []
    pdf_tool = build_pdf_retriever_tool(pdf_files)
    if pdf_tool:
        tools.append(pdf_tool)
    web_tool = make_web_search_tool()
    tools.append(web_tool)

    # Agent & raw LLM
    agent = build_agent(tools, model_name=model_name, temperature=temperature)
    llm_raw = ChatOpenAI(model=model_name, temperature=temperature)

    # Live context for the chosen symbol
    live_ctx_str, ctx_price = make_live_context(symbol, cmc_key, binance_pair if binance_pair else None, lookback=50)
    st.info(live_ctx_str if ctx_price else f"{symbol} 실시간 컨텍스트를 불러오지 못했습니다. 계속 진행은 가능하지만 최신성에 유의하세요.")

    # Tabs
    tab_brief, tab_research, tab_pdf, tab_report = st.tabs(
        ["아침 브리핑", "리서치 Q&A", "PDF 리서치", "리포트 내보내기"]
    )

    # --- Tab 1: Morning Briefing ---
    with tab_brief:
        st.subheader("오늘의 시장 브리핑")
        colA, colB = st.columns([2, 1])

        with colA:
            if st.button("브리핑 업데이트"):
                queries = [
                    f"{symbol} latest news crypto",
                    "crypto market today highlights",
                    "FOMC rate decision outlook",
                    "미국 금리 인하 전망",
                ]
                st.session_state["last_news"] = collect_news_evidence(web_tool, queries, time_scope="day")
                time.sleep(0.2)

            if st.session_state["last_news"]:
                st.markdown(st.session_state["last_news"])
            else:
                info_card("안내", "버튼을 눌러 최신 브리핑을 불러오세요. (뉴스 헤드라인 요약)")

        with colB:
            info_card(
                "브리핑 구성",
                "- 자산별 최신 기사\n- 전체 시장 키워드\n- 금리/거시 키워드(SerpAPI 근거 기반)"
            )

    # --- Tab 2: Research Q&A (항상 뉴스 근거 포함)
    with tab_research:
        st.subheader("리서치 Q&A")
        qcol, scol = st.columns([3, 2])

        with qcol:
            user_q = st.text_area(
                "질문을 입력하세요 (예: 'ETH 스테이킹 동향과 리스크 요약')",
                height=120,
            )
            do_scenario = st.checkbox("포지션 시나리오 카드 함께 생성", value=True)

            if st.button("질문 실행", type="primary"):
                if user_q.strip():
                    # 1) 뉴스 증거 수집 (항상 SerpAPI)
                    queries = [
                        f"{symbol} latest news crypto",
                        f"{symbol} onchain news",
                        "crypto market today highlights",
                        "FOMC rate decision outlook",
                        "미국 금리 인하 전망",
                    ]
                    news_evidence_md = collect_news_evidence(web_tool, queries, time_scope="day")

                    # 2) 에이전트 입력(현재가 컨텍스트 + 뉴스 증거)
                    prompt = (
                        f"[시장 컨텍스트]\n{live_ctx_str}\n\n"
                        f"[질문]\n{user_q}\n\n"
                        "규칙: 오래된 가격/레벨 금지. 현재가·최근 캔들 기준. "
                        "뉴스·거시 코멘트는 '뉴스 증거'에만 근거. 출처 없으면 '언급 생략'."
                    )
                    answer = agent_respond(agent, prompt, st.session_state["chat_history"], news_evidence_md)
                    answer = price_sanity_check(answer, ctx_price)

                    st.session_state["qa_cache"].append((user_q, answer))
                    st.session_state["chat_history"].add_user_message(user_q)
                    st.session_state["chat_history"].add_ai_message(answer)

                    st.markdown("#### 답변")
                    st.markdown(answer)

                    if do_scenario:
                        st.markdown("#### 포지션 시나리오")
                        scenario = build_scenario_with_news(llm_raw, live_ctx_str, news_evidence_md, user_q, DISCLAIMER)
                        scenario = price_sanity_check(scenario, ctx_price)
                        st.markdown(scenario)
                        st.session_state["qa_cache"].append((f"[시나리오:{symbol}]", scenario))
                else:
                    st.info("질문을 입력해 주세요.")

        with scol:
            info_card(
                "동작 원리",
                "1) 현재가·최근 고저 컨텍스트 주입\n2) 뉴스/거시 근거는 항상 SerpAPI 결과만 사용\n3) PDF는 정적 참고(수치 재해석)"
            )
            section_divider()
            if st.session_state["qa_cache"]:
                st.markdown("**최근 Q&A**")
                for q, a in st.session_state["qa_cache"][-5:][::-1]:
                    with st.expander(q):
                        st.markdown(a)

    # --- Tab 3: PDF Research ---
    with tab_pdf:
        st.subheader("PDF 리서치 검색")
        if not pdf_tool:
            st.info("좌측에서 PDF를 업로드하면 PDF 리서치 도구가 활성화됩니다.")
        else:
            q = st.text_input("PDF에서 찾을 내용을 입력하세요 (예: '토큰 분배', '리스크 요인')")
            if st.button("PDF 검색"):
                # PDF 검색은 정적 정보 위주이므로 가격 레벨을 직접 언급하지 않도록 유도
                # 필요 시 최신 시세와 함께 재해석은 Research Q&A 탭에서 수행
                # 그래도 요약에는 신선도 가드 적용
                news_evidence_md = "증거 없음 (PDF 검색 모드)"
                prompt = (
                    "다음 내용을 PDF에서 찾아 간결히 요약하세요. 필요시 출처 페이지를 함께 표시하세요.\n\n"
                    "가격/레벨 언급은 피하고, 구조/정책/토큰노믹스/리스크 등 정적 정보를 위주로 요약하세요.\n\n"
                    f"질문: {q}"
                )
                ans = agent_respond(agent, prompt, st.session_state["chat_history"], news_evidence_md)
                ans = price_sanity_check(ans, ctx_price)
                st.session_state["chat_history"].add_user_message(prompt)
                st.session_state["chat_history"].add_ai_message(ans)
                st.markdown(ans)

    # --- Tab 4: Report Export ---
    with tab_report:
        st.subheader("리서치 리포트 내보내기")
        report_title = st.text_input("리포트 제목", value=f"{APP_NAME} 일일 리서치")
        summary_text = st.text_area("한 줄 개요/요약", height=80, value=f"{symbol} 및 시장 이슈 요약")
        news_md = st.text_area("뉴스 요약 (마크다운)", height=180, value=st.session_state.get("last_news", ""))
        qa_md = st.text_area(
            "Q&A 요약 (마크다운)",
            height=200,
            value="\n\n".join([f"### Q: {q}\n{a}" for q, a in st.session_state["qa_cache"][-5:]]),
        )
        scenario_md = st.text_area("포지션 시나리오 (마크다운)", height=200, value="")

        if st.button("리포트 생성"):
            payload = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "summary": summary_text,
                "news": news_md,
                "qa": qa_md,
                "scenario": scenario_md,
            }
            md = build_research_report(report_title, payload)
            st.success("리포트를 생성했습니다. 아래 버튼으로 다운로드하세요.")
            st.download_button(
                label="마크다운 다운로드 (.md)",
                data=md.encode("utf-8"),
                file_name=f"{report_title.replace(' ', '_')}.md",
                mime="text/markdown",
            )

if __name__ == "__main__":
    main()