import streamlit as st
import pandas as pd
import google.generativeai as genai

# === Set up Gemini API key ===
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# === Load CSV data ===
@st.cache_data
def load_data():
    df = pd.read_csv("cfo_data.csv", encoding="utf-8", on_bad_lines="skip")  # added encoding & line skipping
    return df.fillna("")

data_df = load_data()

# === Helper functions ===
def find_relevant_rows(question, top_n=3):
    question_lower = question.lower()

    # ensure expected columns exist to avoid crashes
    expected_cols = ["Rule", "Example", "Learning Objective", "Related Concept"]
    for col in expected_cols:
        if col not in data_df.columns:
            data_df[col] = ""  # fill missing columns with blank strings

    relevance_scores = data_df.apply(
        lambda row: sum([
            question_lower in str(row[col]).lower()
            for col in expected_cols
        ]), axis=1
    )
    top_rows = data_df.loc[relevance_scores.nlargest(top_n).index]
    return top_rows

def build_prompt(question, context_df):
    context_text = ""
    for i, row in context_df.iterrows():
        context_text += f"""
:small_blue_diamond: **Chapter:** {row.get('Chapter', '')}
:small_blue_diamond: **Section:** {row.get('Section', '')}
:small_blue_diamond: **Rule:** {row.get('Rule', '')}
:small_blue_diamond: **Example:** {row.get('Example', '')}
:small_blue_diamond: **Learning Objective:** {row.get('Learning Objective', '')}
"""
    return f"""
You are a CFO Assistant AI. You will answer the user's question using only the context provided from a corporate finance handbook. 
If no relevant context is found, respond with a thoughtful answer using your finance expertise.

### User Question:
{question}

### CFO Handbook Context:
{context_text}

### Your Answer:
"""

# === Page Header with Gradient Title ===
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

# === Sidebar (About Section) ===
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

# === Initialize Chat History ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your CFO Assistant. Ask me a finance-related question!"}
    ]

# === Display Chat Messages ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Chat Input ===
if user_input := st.chat_input("Ask a CFO-related question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Find relevant context
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_context = find_relevant_rows(user_input)
            prompt = build_prompt(user_input, relevant_context)
            response = model.generate_content(prompt)

            # Display answer
            answer = response.text
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Show context if found
            if not relevant_context.empty:
                with st.expander("📚 Context used from CFO Handbook"):
                    st.dataframe(relevant_context)

