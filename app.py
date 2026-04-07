import json
import os
import re
import textwrap
import urllib.parse
import urllib.request
from datetime import datetime

import streamlit as st
from langchain.agents.factory import create_agent
from langchain_classic.memory.buffer import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq
from langchain.tools import tool

# --- Utilities -----------------------------------------------------------------
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "history.json")
STOPWORDS = {
    "the", "and", "to", "of", "in", "a", "for", "is", "on", "with",
    "as", "by", "an", "from", "that", "it", "this", "at", "be", "are",
    "was", "has", "have", "will", "our", "its", "their", "or", "was",
}


def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(record: dict):
    history = load_history()
    history.append(record)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = {}
    for word in words:
        if word in STOPWORDS:
            continue
        freq[word] = freq.get(word, 0) + 1
    items = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    return [word for word, _ in items[:limit]]


def sentiment_label(text: str) -> str:
    positive = ["growth", "gain", "positive", "strong", "improve", "confidence", "lead", "up", "success", "benefit"]
    negative = ["risk", "drop", "negative", "decline", "loss", "concern", "weak", "fear", "fall", "challenge"]
    score = 0
    text_lower = text.lower()
    for token in positive:
        if token in text_lower:
            score += 1
    for token in negative:
        if token in text_lower:
            score -= 1
    if score > 0:
        return "Positive 😊"
    if score < 0:
        return "Negative 😡"
    return "Neutral 😐"


def shorter_summary(text: str) -> str:
    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    return " ".join(sentences[:2]) if sentences else text


def bullet_summary(text: str, count: int = 5) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", text.strip()) if s.strip()]
    if not sentences:
        words = text.strip().split()
        lines = [" ".join(words[i : i + 10]).strip() for i in range(0, min(len(words), count * 10), 10)]
        return "\n".join(f"• {line}" for line in lines[:count])
    bullets = []
    for sentence in sentences[:count]:
        words = sentence.split()
        short = " ".join(words[:10]).rstrip(".,;:")
        bullets.append(f"• {short}")
    if len(bullets) < count:
        remaining = text.strip().split()
        while len(bullets) < count and remaining:
            next_chunk = " ".join(remaining[:8]).rstrip(".,;:")
            bullets.append(f"• {next_chunk}")
            remaining = remaining[8:]
    return "\n".join(bullets[:count])


def detailed_summary(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", text.strip()) if s.strip()]
    if sentences:
        return " ".join(sentences[:3])
    return text[:200] if len(text) > 200 else text


def local_summarize(text: str, mode: str = "detailed") -> str:
    normalized = mode.lower()
    if "bullet" in normalized:
        return bullet_summary(text, count=5)
    if "short" in normalized:
        return shorter_summary(text)
    return detailed_summary(text)


def news_api_fetch(query: str) -> str:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return (
            "News API not configured. Set NEWSAPI_KEY to fetch live headlines, "
            "or paste article text directly."
        )
    q = urllib.parse.quote(query)
    url = f"https://newsapi.org/v2/everything?q={q}&pageSize=3&language=en&apiKey={api_key}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except Exception as exc:
        return f"News fetch failed: {exc}"
    if data.get("status") != "ok":
        return f"News API error: {data.get('message', 'unknown error')}"
    articles = data.get("articles", [])
    if not articles:
        return "No live news headlines were found for that query."
    lines = [f"{idx+1}. {item.get('title')} ({item.get('source', {}).get('name')})" for idx, item in enumerate(articles)]
    return "\n".join(lines)


# --- Agent tools ---------------------------------------------------------------


def summarize_text_fn(text: str, mode: str = "detailed") -> str:
    if not text.strip():
        return "No text provided for summarization."
    normalized = mode.lower()
    if "bullet" in normalized:
        prompt = (
            "You MUST follow these instructions strictly.\n\n"
            "Task: Convert the news into EXACTLY 5 bullet points.\n\n"
            "Rules:\n"
            "- Each point must be SHORT (max 10 words)\n"
            "- Do NOT copy sentences from input\n"
            "- Extract only key ideas\n"
            "- Use simple words\n\n"
            "Output format:\n"
            "• point 1\n"
            "• point 2\n"
            "• point 3\n"
            "• point 4\n"
            "• point 5\n\n"
            f"News:\n{text.strip()}"
        )
    elif "short" in normalized:
        prompt = (
            "You MUST follow these instructions strictly.\n\n"
            "Task: Summarize the news in ONLY 3 short lines.\n"
            "Keep it concise and avoid long sentences.\n\n"
            "Output format:\n"
            "line 1\n"
            "line 2\n"
            "line 3\n\n"
            f"News:\n{text.strip()}"
        )
    else:
        prompt = (
            "You MUST follow these instructions strictly.\n\n"
            "Task: Give a detailed summary in paragraph form.\n"
            "Focus on the key ideas and write a coherent summary.\n\n"
            f"News:\n{text.strip()}"
        )
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is not set. Please configure the environment variable to enable Groq summarization."
    model = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.2)
    response = model.invoke(prompt)
    return getattr(response, 'content', str(response))

summarize_text = tool(
    "summarize_text",
    description="Summarize news text into bullet points, short summary, or detailed summary.",
)(summarize_text_fn)


def word_counter_fn(text: str) -> str:
    words = re.findall(r"\b\w+\b", text)
    return f"Word count: {len(words)}"

word_counter_tool = tool(
    "word_counter",
    description="Count words in the text and compare before and after summarization.",
)(word_counter_fn)


def news_fetch_fn(query: str) -> str:
    return news_api_fetch(query)

news_fetch_tool = tool(
    "news_fetch",
    description="Fetch latest headlines from News API for a topic or query.",
)(news_fetch_fn)


@st.cache_resource
def build_agent() -> object | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    model = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.2)
    return create_agent(
        model,
        [summarize_text, word_counter_tool, news_fetch_tool],
        system_prompt=(
            "You are a news summarization agent. Use tools to process the user request. "
            "When given a news article or URL, choose the best tool and produce a clean final output. "
            "The final answer should be formatted for display and should not repeat internal reasoning."
        ),
    )


def init_memory() -> ConversationBufferMemory:
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = ConversationBufferMemory()
    return st.session_state.agent_memory


def current_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_result(summary: str, mode: str) -> str:
    if mode == "Bullet Points":
        lines = summary.strip().split('\n')
        formatted = []
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            if line_clean.startswith(('•', '-', '*')):
                formatted.append(line_clean if line_clean.startswith('•') else '•' + line_clean[1:])
            else:
                formatted.append(f'• {line_clean}')
        return '\n'.join(formatted)
    if mode == "Short":
        return summary
    return summary


# --- Streamlit UI -------------------------------------------------------------

st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="wide",
)

st.title("🧠 AI News Summarizer")
st.markdown(
    "Use the agentic AI system below to summarize articles, compare summaries, and store history. "
    "The agent decides which tool to use and Groq LLaMA3 handles the reasoning and summary generation."
)

col1, col2 = st.columns([2, 1])
with col1:
    article_text = st.text_area("Paste news article text", height=260)
    article_url = st.text_input("Article URL (optional)")
    fetch_query = st.text_input("Fetch live news query (optional)", help="Use News API to fetch headlines automatically.")
    mode = st.selectbox("Summary mode", ["Bullet Points", "Short", "Detailed"])
    use_groq = st.checkbox("Use Groq AI (requires GROQ_API_KEY)", value=True)
    analyze_keywords = st.checkbox("Extract keywords", value=True)
    analyze_sentiment = st.checkbox("Sentiment analysis", value=True)
    save_history_toggle = st.checkbox("Save summary history", value=True)
    submit = st.button("Summarize news")
    clear_memory = st.button("Clear agent memory")

with col2:
    st.markdown("### System Report")
    st.markdown(
        "**Architecture**\n"
        "User Input → LangChain Agent → Groq LLM → Tools (Summarizer / News API / Word Counter) → Memory → Output"
    )
    st.markdown(
        "**Algorithm**\n"
        "1. User enters news text or query.\n"
        "2. Agent receives input and decides which tool to use.\n"
        "3. Groq LLaMA3 processes the request.\n"
        "4. Summary generated and memory updated.\n"
        "5. Output displayed cleanly."
    )
    st.markdown("**Results & Analysis**")
    st.write("Speed: Fast\nAccuracy: High\nFlexibility: Good")

if clear_memory:
    if "agent_memory" in st.session_state:
        st.session_state.agent_memory.clear()
        st.success("Agent memory cleared.")
    if "conversation_history" in st.session_state:
        st.session_state.conversation_history = []

memory = init_memory()
conversation_history = st.session_state.get("conversation_history", [])
history = load_history()

summary_text = ""
summary_display = ""
sentiment = ""
keywords = []
before_count = after_count = ""
agent_message = ""

if submit:
    if not article_text and not article_url and not fetch_query:
        st.error("Please paste news text, enter a URL, or provide a live news query.")
    else:
        agent_input = article_text.strip() or ""
        if article_url:
            agent_input += f"\nURL: {article_url}"
        if fetch_query:
            agent_input += f"\nFetch query: {fetch_query}"
        agent_input = agent_input.strip()

        agent = build_agent()
        if agent is None or not use_groq:
            summary_text = local_summarize(agent_input, mode.lower())
        else:
            prompt = (
                "You are a news summarization agent. Use the summarization tool and return only the final summary.\n\n"
                f"Mode: {mode.lower()}\n\n"
                f"News:\n{agent_input}"
            )
            try:
                with st.spinner("Agent is analyzing the article and selecting tools..."):
                    summary_text = agent.run(prompt)
            except Exception as exc:
                st.error("Agent call failed. See details below.")
                st.exception(exc)
                summary_text = ""

        if summary_text:
            before_count = word_counter_fn(agent_input)
            after_count = word_counter_fn(summary_text)
            sentiment = sentiment_label(summary_text) if analyze_sentiment else "Not analyzed"
            keywords = extract_keywords(summary_text) if analyze_keywords else []

            memory.save_context({"input": agent_input}, {"output": summary_text})
            conversation_history.append({
                "time": current_datetime(),
                "input": agent_input,
                "summary": summary_text,
                "mode": mode,
            })
            st.session_state.conversation_history = conversation_history

            if save_history_toggle:
                save_history(
                    {
                        "timestamp": current_datetime(),
                        "mode": mode,
                        "input": agent_input,
                        "summary": summary_text,
                        "keywords": keywords,
                        "sentiment": sentiment,
                    }
                )

            summary_display = format_result(summary_text, mode)

if summary_display:
    st.markdown("### Final Summary")
    st.markdown(summary_display)
    st.markdown("---")
    st.markdown("### Debug Info")
    st.info(f"Summary length: {len(summary_text)} characters | {len(summary_text.split())} words")
    st.markdown("### Summary Details")
    st.markdown(f"**Mode:** {mode}")
    st.markdown(f"**Saved interactions:** {len(conversation_history)}")
    st.markdown(f"**Input word count:** {before_count}")
    st.markdown(f"**Summary word count:** {after_count}")
    st.markdown(f"**Sentiment:** {sentiment}")
    if keywords:
        st.markdown(f"**Keywords:** {', '.join(keywords)}")

if conversation_history:
    st.markdown("---")
    st.markdown("### Conversation Memory")
    for item in reversed(conversation_history[-5:]):
        st.write(f"**{item['time']} — Mode: {item['mode']}**")
        st.write(item["input"])
        st.write(item["summary"])

if history:
    with st.expander("Saved history records"):
        for item in reversed(history[-5:]):
            st.markdown(f"- {item.get('timestamp')} | {item.get('mode')} | {item.get('sentiment')}")

st.markdown("---")
st.markdown("### Case Study Example")
st.write(
    "Input: ‘India launches AI program…’\nProcessing: Agent → Tool → LLM\nOutput: AI initiative launched, Infrastructure focus, Economic growth"
)
