import streamlit as st
from rag import HelpdeskBot, load_kb

st.set_page_config(page_title="Lovely Professional University Helpdesk", page_icon="🎓")
st.title("🎓 Lovely Professional University Helpdesk")

if "bot" not in st.session_state:
    st.session_state.bot = HelpdeskBot()
    st.session_state.chat = []          # for display: (role, text, debug)
bot = st.session_state.bot

with st.sidebar:
    st.subheader("Settings")
    st.caption(f"Mode: **{bot.mode}**")
    show_debug = st.toggle("Show retrieval details", value=True)
    if st.button("🗑️ New conversation"):
        bot.reset(); st.session_state.chat = []; st.rerun()
    st.subheader("Conversation memory")
    st.write(bot.memory.facts or "_No user facts yet_")
    st.caption(f"{len(bot.memory.turns)//2} exchange(s) remembered")
    st.subheader("Try asking")
    examples = ["How do I reset my password?", "When is tuition due?",
                "Can I pay in installments?", "What about the late fee?",
                "What are the library hours?", "Who won the football match?"]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

st.caption("Answers come only from the helpdesk knowledge base. If I don't know, I'll say so.")

for role, text, dbg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)
        if dbg and show_debug:
            with st.expander("Retrieval details"):
                st.write(dbg)

prompt = st.chat_input("Ask a question…") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.chat.append(("user", prompt, None))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            res = bot.ask(prompt)
        st.markdown(res["answer"])
        dbg = {"search_query": res["query"],
               "retrieved": [f"{e['id']} – {e['question']} (score {s:.2f})" for e, s in res["sources"]]
                            or "Nothing above the relevance threshold"}
        if show_debug:
            with st.expander("Retrieval details"):
                st.write(dbg)
    st.session_state.chat.append(("assistant", res["answer"], dbg))
    st.rerun()
