import streamlit as st
import pandas as pd
import google.generativeai as genai
from difflib import SequenceMatcher

# === Set up Gemini API key ===
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# === Load and preprocess CSV data ===
@st.cache_data
def load_data():
    df = pd.read_csv("cfo_data.csv", encoding="utf-8", on_bad_lines="skip")
    df.fillna("", inplace=True)
    
    # Ensure expected columns are present
    expected_cols = ["Chapter", "Section", "Rule", "Example", "Learning Objective", "Related Concept"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    return df

data_df = load_data()

# === Fuzzy Matching ===
def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_relevant_rows(question, top_n=3):
    expected_cols = ["Rule", "Example", "Learning Objective", "Related Concept"]

    def row_score(row):
        return sum([similarity(question, str(row[col])) for col in expected_cols])

    df_copy = data_df.copy()  # avoid mutating global cached df
    df_copy["score"] = df_copy.apply(row_score, axis=1)
    top_rows = df_copy.sort_values("score", ascending=False).head(top_n)
    return top_rows.drop(columns=["score"])

# === Prompt Builder ===
def build_prompt(question, context_df):
    context_text = ""
    for _, row in context_df.iterrows():
        context_text += f"""
:small_blue_diamond: **Chapter:** {row.get('Chapter', '')}
:small_blue_diamond: **Section:** {row.get('Section', '')}
:small_blue_diamond: **Rule:** {row.get('Rule', '')}
:small_blue_diamond: **Example:** {row.get('Example', '')}
:small_blue_diamond: **Learning Objective:** {row.get('Learning Objective', '')}
"""
    return f"""
You are a CFO Assistant AI. Use only the context below from a corporate finance handbook to answer the user’s question.
If the context isn't relevant, answer using your finance expertise but clearly explain your reasoning.

### User Question:
{question}

### CFO Handbook Context:
{context_text}

### Your Answer:
"""

# === UI: Header with Gradient Text ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
.title {
  font-family: 'Poppins', sans-serif;
  font-weight: 900;
  font-size: 3em;
  background: linear-gradient(90deg, #ff6a00, #ee0979);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  margin-bottom: 0;
}
.caption {
  text-align: center;
  font-size: 1.1em;
  color: gray;
  margin-bottom: 30px;
}
</style>
<div class="title">CFO Assistant</div>
<div class="caption">Your AI partner for capital budgeting, cash management, and investment questions.</div>
""", unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.header("📖 About CFO Assistant")
st.sidebar.markdown("""
CFO Assistant is an AI-powered tool that helps you:

- Understand **capital budgeting** decisions  
- Explore **cash flow management** techniques  
- Analyze **debt and investment strategies**  

💡 **Powered by Gemini 2.5 Flash** and your corporate finance handbook.
""")
st.sidebar.markdown("---")
st.sidebar.info("Developed to assist finance teams in decision-making.")

# === Chat History Setup ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your CFO Assistant. Ask me a finance-related question!"}
    ]

# === Display Previous Messages ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Chat Input Handling ===
if user_input := st.chat_input("Ask a CFO-related question..."):
    # Append user input to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_context = find_relevant_rows(user_input)
            prompt = build_prompt(user_input, relevant_context)

            try:
                response = model.generate_content([{"role": "user", "parts": [prompt]}])
                answer = response.text.strip() if response.text else "I couldn’t generate a response."
            except Exception as e:
                answer = f"💥 Oops! Something went wrong.\n\n`{e}`"

            # Show AI answer
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Show the context used
            if not relevant_context.empty:
                with st.expander("📚 Context used from CFO Handbook"):
                    st.dataframe(relevant_context)
