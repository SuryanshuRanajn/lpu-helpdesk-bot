"""Core RAG + memory logic for the LPU University Helpdesk bot (UI-independent).

Modes (auto-selected from environment variables):
  1. Claude   -> set ANTHROPIC_API_KEY
  2. Gemini   -> set GEMINI_API_KEY   (free tier available)
  3. Offline  -> no key needed (extractive answers + simple small talk)
"""
import json, os, re
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).with_name("kb.json")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")   # change if Google renames models
THRESHOLD = 0.15          # minimum cosine similarity to count as "relevant"
TOP_K = 3
MEMORY_TURNS = 6          # how many past user/assistant exchanges to keep in the prompt

FALLBACK = ("I'm sorry, but I don't have that information in my knowledge base. "
            "Please contact the Student Helpdesk at helpdesk@LPU.example or 555-0100 "
            "(Mon-Fri, 9:00 AM-5:00 PM) for help.")

# Words suggesting the question depends on earlier context
FOLLOWUP_HINTS = re.compile(
    r"\b(it|that|this|they|them|those|these|there|its|then|also|instead|else|"
    r"what about|how about|and if|how long|how much|how many|when)\b", re.I)
# Questions about the conversation itself (answerable from memory, not the KB)
MEMORY_Q = re.compile(r"(what('s| is) my name|who am i|what did i (ask|say)|"
                      r"(my )?(last|previous) question|remind me what|what did i tell you)", re.I)

STANDALONE_SCORE = 0.25   # if the question alone retrieves at least this well, don't borrow context

# Offline small talk (used only when no LLM is available)
SMALL_TALK = [
    (re.compile(r"^\s*(hi|hello|hey|hy|good (morning|afternoon|evening))\b", re.I),
     "Hey there! 👋 I'm the LPU helpdesk assistant. What can I help you with?"),
    (re.compile(r"how are you|how's it going|how is your day", re.I),
     "I'm doing great, thanks for asking! How can I help you today?"),
    (re.compile(r"\b(thanks|thank you|thx)\b", re.I),
     "You're welcome! Anything else I can help with?"),
    (re.compile(r"who are you|what can you do", re.I),
     "I'm the LPU helpdesk assistant. I can help with passwords, fees, registration, "
     "the library, student services and more."),
    (re.compile(r"\b(bye|goodbye|see you)\b", re.I),
     "Goodbye! Have a great day 😊"),
]


def load_kb(path=KB_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Retriever:
    """TF-IDF retriever over the FAQ entries (question is up-weighted by repetition)."""
    def __init__(self, kb):
        self.kb = kb
        docs = [f"{e['category']} {e['question']} {e['question']} {e['answer']}" for e in kb]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs)

    def search(self, query, k=TOP_K):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]

        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [(self.kb[i], float(sims[i])) for i in ranked]


@dataclass
class Memory:
    """Conversation memory: message history + facts the user told us + last-used FAQ."""
    turns: list = field(default_factory=list)      # [{"role","content"}]
    facts: dict = field(default_factory=dict)      # e.g. {"name": "Asha"}
    last_faq: dict | None = None

    def add(self, role, content):
        self.turns.append({"role": role, "content": content})

    def window(self):
        return self.turns[-2 * MEMORY_TURNS:]

    def last_user_question(self):
        users = [t["content"] for t in self.turns if t["role"] == "user"]
        return users[-1] if users else None

    def learn(self, text):
        m = re.search(r"\b(?:my name is|i am called|call me)\s+([A-Z][a-zA-Z'-]+)", text, re.I)
        if m: self.facts["name"] = m.group(1).capitalize()
        m = re.search(r"\bi(?:'m| am) an? ([\w\s-]{3,40}?) student\b", text, re.I)
        if m: self.facts["student type"] = m.group(1).strip().lower()


class HelpdeskBot:
    def __init__(self):
        self.kb = load_kb()
        self.retriever = Retriever(self.kb)

        self.memory = Memory()
        self.provider = None          # "claude" | "gemini" | None (offline)
        self.client = None
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                self.provider = "claude"
            except ImportError:
                pass
        if self.provider is None and self.gemini_key:
            try:
                import requests  # noqa: F401
                self.provider = "gemini"
            except ImportError:
                pass

    @property
    def mode(self):
        if self.provider == "claude":
            return f"Claude ({CLAUDE_MODEL})"
        if self.provider == "gemini":
            return f"Gemini ({GEMINI_MODEL})"
        return "Offline (extractive)"

    def reset(self):
        self.memory = Memory()

    # ---- unified LLM call ----
    def _chat(self, system, messages, max_tokens=500):

        if self.provider == "claude":
            r = self.client.messages.create(model=CLAUDE_MODEL, max_tokens=max_tokens,
                                            system=system, messages=messages)
            return r.content[0].text.strip()
        if self.provider == "gemini":
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            contents = [{"role": "user" if m["role"] == "user" else "model",
                         "parts": [{"text": m["content"]}]} for m in messages]
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": contents,
                    "generationConfig": {"maxOutputTokens": max(max_tokens, 1024)}}
            r = requests.post(url, json=body, timeout=30,
                              headers={"x-goog-api-key": self.gemini_key,
                                       "Content-Type": "application/json"})
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        raise RuntimeError("No LLM configured")

    # ---- step 1: make the query standalone using memory ----
    def rewrite_query(self, question):
        prev = self.memory.last_user_question()
        if not prev:
            return question
        if self.provider:
            hist = "\n".join(f"{t['role']}: {t['content']}" for t in self.memory.window())
            try:
                text = self._chat(
                    "Rewrite the user's latest message as a standalone search query using the "
                    "conversation for context. Output only the query, nothing else.",
                    [{"role": "user", "content": f"{hist}\nuser: {question}"}], max_tokens=100)

                if text:
                    return text
            except Exception:
                pass
        # offline heuristic: only borrow context if the question looks like a follow-up
        if not FOLLOWUP_HINTS.search(question):
            return question
        top = self.retriever.search(question, 1)[0][1]
        if top >= STANDALONE_SCORE:
            return question
        topic = self.memory.last_faq["question"] if self.memory.last_faq else prev
        return f"{topic} {question}"

    # ---- step 2: retrieve ----
    def retrieve(self, query):
        hits = self.retriever.search(query)
        return [(e, s) for e, s in hits if s >= THRESHOLD]

    # ---- step 3: generate a grounded, conversational answer ----
    def _llm_answer(self, question, hits):
        ctx = "\n\n".join(f"[{e['id']}] Q: {e['question']}\nA: {e['answer']}" for e, _ in hits) or "(none)"
        facts = "; ".join(f"{k}: {v}" for k, v in self.memory.facts.items()) or "(none)"
        system = (
            "You are Nora, a warm, friendly assistant for the LPU University Student Helpdesk. "
            "Chat naturally like a helpful person: respond to greetings, small talk, thanks and "
            "casual messages in a relaxed, conversational tone.\n"
            "RULES:\n"
            "1. For any university facts (dates, fees, hours, procedures, contacts), use ONLY the "
            "KNOWLEDGE BASE CONTEXT below. Never guess or use outside knowledge.\n"
            "2. If the user asks a university question the context does not cover, say kindly that you "
            f"don't have that information and suggest: {FALLBACK}\n"
            "3. For topics unrelated to the university (news, sports, etc.), politely say that's outside "

            "what you can help with, and offer to help with university questions.\n"
            "4. Use the conversation history to resolve follow-ups (it, that, 'what about...').\n"
            "5. Use USER FACTS (like their name) naturally.\n"
            "6. Keep replies short and friendly. When you use FAQ info, cite it like (FAQ-04).\n\n"
            f"USER FACTS: {facts}\n\nKNOWLEDGE BASE CONTEXT:\n{ctx}")
        msgs = self.memory.window() + [{"role": "user", "content": question}]
        return self._chat(system, msgs, max_tokens=500)

    def _offline_answer(self, question, hits):
        if MEMORY_Q.search(question):
            name = self.memory.facts.get("name")
            if re.search(r"my name|who am i", question, re.I):
                return f"You told me your name is {name}." if name else "You haven't told me your name yet."
            prev = [t["content"] for t in self.memory.turns if t["role"] == "user"]
            if prev and re.search(r"what did i|last question|previous", question, re.I):
                return f'Your previous question was: "{prev[-1]}"'
        if not hits:
            for pattern, reply in SMALL_TALK:
                if pattern.search(question):
                    return reply
            if self.memory.facts and re.search(r"my name is|i am|i'm|call me", question, re.I):
                who = self.memory.facts.get("name", "")
                return f"Nice to meet you{', ' + who if who else ''}! I'll remember that. How can I help you today?"
            return FALLBACK
        top, _ = hits[0]
        return f"{top['answer']}  \n\n*(Source: {top['id']})*"

    def ask(self, question):
        """Returns dict(answer, query, sources[(entry, score)])."""
        self.memory.learn(question)
        query = self.rewrite_query(question)
        hits = self.retrieve(query)


        if self.provider:
            try:
                answer = self._llm_answer(question, hits)
                if not answer:
                    raise RuntimeError("empty response")
            except Exception as ex:
                answer = self._offline_answer(question, hits) + f"\n\n(LLM unavailable: {ex})"
        else:
            answer = self._offline_answer(question, hits)

        if hits:
            self.memory.last_faq = hits[0][0]
        self.memory.add("user", question)
        self.memory.add("assistant", answer)
        return {"answer": answer, "query": query, "sources": hits}