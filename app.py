"""
Mini chatbot (Streamlit + Groq).

Features: calculator tool, document Q&A (PDF/TXT/MD), history trimming,
auto-retry on rate limits, save/load chats as JSON, specialist personas.

Run locally:  streamlit run app.py
"""

import io
import json
import os
import time

import streamlit as st
from groq import APIConnectionError, APIStatusError, Groq

from calc import calculate

try:
    from pypdf import PdfReader
except ImportError:  # PDF upload disabled if pypdf is missing
    PdfReader = None

# ---------------------------------------------------------------- config
MODELS = {
    "GPT-OSS 120B (best quality, calculator)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (faster, calculator)": "openai/gpt-oss-20b",
    "Groq Compound (web search + code, no calculator tool)": "groq/compound",
}

PERSONAS = {
    "General assistant": "You are a helpful, accurate assistant. Answer any question clearly.",
    "Math tutor": (
        "You are a patient math tutor. Solve problems step by step, explain the reasoning, "
        "and verify the final answer."
    ),
    "Aviation maintenance & safety assistant": (
        "You are an assistant for aviation maintenance, MRO optimization and aviation safety. "
        "Be precise, separate established practice from opinion, and remind the user that "
        "approved maintenance data and regulations take precedence over your answers."
    ),
}

RULES = (
    "\n\nRules: format math with LaTeX ($...$ inline, $$...$$ for display). "
    "Say clearly when you are unsure instead of guessing."
)
TOOL_RULE = (
    " For any arithmetic beyond trivial, call the calculator tool and use its result "
    "(it supports + - * / ^ %, sqrt, sin, cos, tan, log, ln, exp, factorial, pi, e)."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression exactly, e.g. '12.5*sqrt(2)+3^4'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]

# ---------------------------------------------------------------- helpers
def get_secret_or_env(name):
    try:
        val = st.secrets.get(name)
        if val:
            return val
    except Exception:  # no secrets file locally
        pass
    return os.environ.get(name)


def call_with_retry(client, **kwargs):
    """Retry on rate limits, connection problems and 5xx errors (waits 2s, 4s, 8s)."""
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except APIConnectionError:
            if attempt == 3:
                raise
        except APIStatusError as e:
            retryable = e.status_code == 429 or e.status_code >= 500
            if not retryable or attempt == 3:
                raise
        time.sleep(2 ** (attempt + 1))


def ask(client, model, system, history, use_tools):
    """Returns (answer_text, calculator_log). Handles the tool-calling loop."""
    msgs = [{"role": "system", "content": system}] + history
    kwargs = {"model": model, "messages": msgs, "max_completion_tokens": 4096}
    if use_tools:
        kwargs["tools"] = TOOLS
    log = []
    for _ in range(6):
        resp = call_with_retry(client, **kwargs)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "", log
        msgs.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc.function.name == "calculator":
                result = calculate(args.get("expression", ""))
                log.append(result)
            else:
                result = "Error: unknown tool"
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        kwargs["messages"] = msgs
    return "I stopped after too many calculation steps. Try breaking the question into parts.", log


def extract_text(uploaded):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("PDF support needs: pip install pypdf")
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Mini Chatbot", page_icon="💬")
st.title("💬 Mini Chatbot")

# optional password gate (set APP_PASSWORD in secrets to protect a public app)
app_password = get_secret_or_env("APP_PASSWORD")
if app_password and not st.session_state.get("authed"):
    pw = st.text_input("Password", type="password")
    if pw:
        if pw == app_password:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("doc_text", "")
st.session_state.setdefault("doc_name", "")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    api_key = get_secret_or_env("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Get one at console.groq.com/keys")

    model_label = st.selectbox("Model", list(MODELS))
    model = MODELS[model_label]
    persona = st.selectbox("Persona", list(PERSONAS))
    max_msgs = st.slider("Messages remembered", 4, 60, 20, 2,
                         help="Older messages are dropped to avoid rate limits.")

    st.divider()
    st.subheader("Document Q&A")
    up = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"])
    doc_chars = st.slider("Max document characters sent", 2000, 40000, 12000, 1000,
                          help="Longer = more context but more tokens (rate limits).")
    if up is not None and up.name != st.session_state["doc_name"]:
        try:
            st.session_state["doc_text"] = extract_text(up)
            st.session_state["doc_name"] = up.name
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read file: {e}")
    if st.session_state["doc_name"]:
        total = len(st.session_state["doc_text"])
        st.caption(f"Loaded: {st.session_state['doc_name']} ({total:,} chars"
                   + (f", only first {doc_chars:,} used)" if total > doc_chars else ")"))
        if not st.session_state["doc_text"].strip():
            st.warning("No text found (scanned PDF?).")
        if st.button("Remove document"):
            st.session_state["doc_text"] = ""
            st.session_state["doc_name"] = ""
            st.rerun()

    st.divider()
    st.subheader("Chat")
    st.download_button(
        "Download chat (JSON)",
        data=json.dumps(st.session_state["messages"], indent=2, ensure_ascii=False),
        file_name="chat.json",
        mime="application/json",
        disabled=not st.session_state["messages"],
    )
    loaded = st.file_uploader("Load a saved chat", type=["json"], key="chat_loader")
    if loaded is not None and st.button("Load chat"):
        try:
            data = json.loads(loaded.getvalue().decode("utf-8"))
            ok = isinstance(data, list) and all(
                isinstance(m, dict) and m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str) for m in data
            )
            if not ok:
                raise ValueError("unexpected format")
            st.session_state["messages"] = data
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load chat: {e}")
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()

# ---------------------------------------------------------------- chat
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("calcs"):
            with st.expander("Calculator steps"):
                for c in m["calcs"]:
                    st.code(c)

prompt = st.chat_input("Ask anything, including math...")
if prompt:
    if not api_key:
        st.error("Add your Groq API key in the sidebar first.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    use_tools = not model.startswith("groq/compound")
    system = PERSONAS[persona] + RULES + (TOOL_RULE if use_tools else "")
    if st.session_state["doc_text"].strip():
        system += (
            f"\n\nThe user uploaded a document named '{st.session_state['doc_name']}'. "
            "Answer from it when relevant and say if the answer is not in it.\n"
            "--- DOCUMENT START ---\n"
            + st.session_state["doc_text"][:doc_chars]
            + "\n--- DOCUMENT END ---"
        )

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"][-max_msgs:]
    ]
    while history and history[0]["role"] != "user":
        history.pop(0)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer, calcs = ask(Groq(api_key=api_key), model, system, history, use_tools)
            st.markdown(answer)
            if calcs:
                with st.expander("Calculator steps"):
                    for c in calcs:
                        st.code(c)
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, **({"calcs": calcs} if calcs else {})}
            )
        except APIStatusError as e:
            st.session_state["messages"].pop()
            st.error(f"API error {e.status_code}: {e.message}")
        except Exception as e:  # noqa: BLE001
            st.session_state["messages"].pop()
            st.error(f"Error: {e}")
