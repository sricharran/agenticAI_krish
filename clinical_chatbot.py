import os
import streamlit as st
import pandas as pd
import json
import datetime
import matplotlib.pyplot as plt
import mysql.connector
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import RetrievalQA
from langsmith import traceable
from transformers import pipeline
from pinecone import Pinecone
import pinecone
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ClinicalPsychBot"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Charran@2801",
    database="psychologybot"
)

# ✅ Initialize Components
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
groq_llm = ChatGroq(model_name="llama-3.3-70b-versatile")
pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore.from_existing_index(index_name="psychology-chatbot", embedding=embedding_model)
retriever = vectorstore.as_retriever()

# ✅ MySQL Database
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255),
        age INT
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_analysis (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_query TEXT,
        query_sentiment VARCHAR(50),
        system_response TEXT,
        response_sentiment VARCHAR(50),
        timestamp DATETIME
    )
""")
conn.commit()


# ✅ Helper Functions
def save_user_data(name, age):
    cursor.execute("INSERT INTO users (name, age) VALUES (%s, %s)", (name, age))
    conn.commit()


def get_user_data():
    cursor.execute("SELECT name, age FROM users ORDER BY id DESC LIMIT 1")
    return cursor.fetchone()


def save_sentiment_analysis(user_query, query_sentiment, system_response, response_sentiment):
    cursor.execute("""
        INSERT INTO sentiment_analysis (user_query, query_sentiment, system_response, response_sentiment, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """, (user_query, query_sentiment, system_response, response_sentiment, datetime.datetime.now()))
    conn.commit()


# ✅ Sentiment Analysis Dashboard
def sentiment_dashboard():
    st.subheader("📊 Sentiment Trend Analysis")
    cursor.execute("SELECT timestamp, query_sentiment, response_sentiment FROM sentiment_analysis")
    data = cursor.fetchall()
    if not data:
        st.write("No sentiment data available.")
        return

    df = pd.DataFrame(data, columns=["timestamp", "query_sentiment", "response_sentiment"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sentiment_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    df["query_sentiment"] = df["query_sentiment"].map(sentiment_mapping)
    df["response_sentiment"] = df["response_sentiment"].map(sentiment_mapping)

    st.write("### Sentiment Trends Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.set_index("timestamp")["query_sentiment"].value_counts().sort_index().plot(ax=ax, kind="line", marker="o",
                                                                                  label="User Query Sentiment")
    df.set_index("timestamp")["response_sentiment"].value_counts().sort_index().plot(ax=ax, kind="line", marker="s",
                                                                                     label="System Response Sentiment")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment Count")
    ax.legend()
    st.pyplot(fig)


# ✅ Few-Shot Learning Responses
few_shot_qa = {
    "hi": "Hello! How may I assist you today?",
    "hello": "Hi there! How can I help you?",
    "hey": "Hey! How are you feeling today?",
    "who are you": "I am a clinical psychologist chatbot designed to assist with mental health guidance.",
    "who are you?": "I am a clinical psychologist chatbot designed to assist with mental health guidance.",
    "what do you do": "I provide psychological insights and support to help users manage their emotions better.",
    "what do you do?": "I provide psychological insights and support to help users manage their emotions better.",
    "can you help me?": "Of course! I'm here to assist you with any mental health concerns or emotional struggles.",
    "i feel sad": "I'm really sorry to hear that. Do you want to talk about what’s making you feel this way?",
    "i feel anxious": "Anxiety can be tough. Would you like to try a breathing exercise to calm down?",
    "i am stressed": "I understand. Stress is difficult, but deep breathing or mindfulness can help. Would you like a relaxation tip?",
    "how can i reduce stress": "Some helpful methods include deep breathing, journaling, and meditation. Would you like a guided exercise?",
    "how can i be happier": "Happiness comes from within. Practicing gratitude, self-care, and connecting with loved ones can help.",
    "how do i improve my mental health": "Maintaining a routine, eating well, and exercising can improve mental well-being. Would you like more tips?",
    "what is depression": "Depression is a mood disorder that affects how you feel, think, and behave. It’s treatable with therapy and support.",
    "what is anxiety": "Anxiety is a normal stress response but can become overwhelming. Techniques like mindfulness can help.",
    "how can i sleep better": "Try a bedtime routine, limit screen time, and practice relaxation techniques before sleeping.",
    "how do i stop overthinking": "Overthinking can be exhausting. Try journaling your thoughts or using mindfulness techniques.",
    "how do i manage anger": "Anger management strategies include deep breathing, counting to ten, and finding healthy outlets like exercise.",
    "how do i boost confidence": "Building self-confidence takes time. Start by acknowledging your strengths and practicing self-compassion.",
    "what is mindfulness": "Mindfulness is being present in the moment without judgment. Would you like a simple mindfulness exercise?",
    "how do i practice self-love": "Self-love means accepting yourself, setting boundaries, and treating yourself with kindness.",
    "can you tell me a motivational quote?": "Sure! 'You are stronger than you think, and more capable than you imagine.'",
    "how can i deal with loneliness": "Loneliness can be difficult. Engaging in hobbies, connecting with friends, and seeking support can help.",
    "how can i stop negative thoughts": "Challenge negative thoughts by replacing them with positive affirmations and practicing gratitude.",
    "how can i feel more positive": "Focus on gratitude, engage in activities you enjoy, and surround yourself with supportive people.",
    "what are some relaxation techniques?": "Deep breathing, progressive muscle relaxation, and meditation can help with relaxation.",
    "what is emotional intelligence?": "Emotional intelligence is the ability to recognize, understand, and manage your emotions effectively.",
    "how can i be more resilient?": "Building resilience involves self-care, seeking support, and maintaining a positive mindset.",
    "why do i feel empty?": "Feeling empty can stem from emotional burnout or disconnection. Engaging in meaningful activities may help.",
    "how do i stop self-doubt?": "Self-doubt can be overcome by challenging negative beliefs and celebrating small achievements.",
    "how do i set healthy boundaries?": "Setting boundaries involves communicating your needs clearly and respecting your own limits.",
    "how do i handle rejection?": "Rejection is tough, but it’s not a reflection of your worth. Learning from experiences can help you grow.",
    "what is self-care?": "Self-care means taking time for yourself—physically, emotionally, and mentally—to maintain well-being.",
    "how do i manage work stress?": "Work stress can be managed by prioritizing tasks, taking breaks, and maintaining work-life balance.",
    "how can i stop procrastinating?": "Breaking tasks into smaller steps and setting clear goals can help overcome procrastination.",
    "what should i do if i feel overwhelmed?": "Take deep breaths, break tasks into steps, and reach out for support when needed.",
    "how do i handle criticism?": "Constructive criticism helps growth. Focus on learning rather than taking it personally.",
    "how do i let go of the past?": "Letting go involves acceptance, self-compassion, and focusing on the present and future.",
    "how do i improve my relationships?": "Good communication, active listening, and empathy can improve relationships.",
    "what is gratitude practice?": "Gratitude practice involves acknowledging and appreciating the positive aspects of life daily.",
    "how do i develop a positive mindset?": "Challenge negative thoughts, practice gratitude, and surround yourself with positivity.",
    "can i trust you?": "I am designed to provide psychological guidance, but I encourage seeking professional help for serious concerns."
}



@traceable

# ✅ Retrieve Answer


def retrieve_answer(query, age):
    # Define the system prompt
    system_message = (
        "You are a compassionate and professional clinical psychologist. Your goal is to provide thoughtful, empathetic, "
        "and practical advice to the user based on their queries. If the question is medical or urgent, "
        "encourage the user to seek professional help promptly. Tailor your language to be supportive and accessible, "
        "respecting the user's age and context."
    )

    # Create a ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template("{query}")
    ])

    # Format the prompt with the user's query
    formatted_prompt = chat_prompt.format_prompt(query=query)

    lower_query = query.lower().strip()
    if lower_query in few_shot_qa:
        response = few_shot_qa[lower_query]
    else:
        # Use the formatted prompt in the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=groq_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        response = qa_chain.invoke({"query": formatted_prompt.to_string()})["result"]

    # Add age-specific guidance
    if age < 18:
        response += "\n(I'm here to guide you, but consider talking to a trusted adult or professional.)"
    elif age > 50:
        response += "\n(It's great that you're focusing on mental well-being at this stage of life.)"

    return response



def run_chatbot():
    # Page Configuration
    st.set_page_config(page_title="Clinical Psychologist Chatbot", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .css-1cpxqw2 {padding: 1rem;} /* Adjust padding */
        .stChatMessage {margin-bottom: 1rem;} /* Spacing for chat */
        .bot-response {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.image("D:/projects/psych_chatbot/home_image.jpg",width=200)
        st.title("🧠 Clinical Psychologist ChatBot")
        st.subheader("Navigation")
        page = st.radio("Choose a page:", ["Chatbot", "Sentiment Dashboard"])

    if page == "Sentiment Dashboard":
        sentiment_dashboard()
    else:
        st.subheader("💬 Chat with AI")

        # Session State for User Info
        if "user_info" not in st.session_state:
            st.session_state.user_info = None

        if st.session_state.user_info:
            user_name, user_age = st.session_state.user_info
            st.success(f"✅ Logged in as: **{user_name}** (Age: {user_age})")
            if st.button("✏️ Edit Details"):
                st.session_state.user_info = None
                st.rerun()
        else:
            user_name = st.text_input("👤 Your Name:")
            user_age = st.slider("🎂 Your Age:", min_value=10, max_value=100, value=25)
            if st.button("✅ Save User Info"):
                st.session_state.user_info = (user_name, user_age)
                save_user_data(user_name, user_age)
                st.rerun()

        # Initialize Chat History
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chatbot Conversation
        st.subheader("💬 Conversation")
        user_input = st.chat_input("Ask me anything about mental health...")

        if user_input:
            with st.spinner("Thinking..."):
                response = retrieve_answer(user_input, user_age)

                query_sentiment = sentiment_pipeline(user_input)[0]['label']
                response_sentiment = sentiment_pipeline(response)[0]['label']
                save_sentiment_analysis(user_input, query_sentiment, response, response_sentiment)

                # Append new chat to history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response))

        # Display Chat History
        for role, text in st.session_state.chat_history:
            st.chat_message(role).markdown(text)

        # Clear Chat Button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []  # Clear chat history
            st.rerun()

if __name__ == "__main__":
    run_chatbot()
