import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from googletrans import Translator  # From googletrans-py

# Load environment variables
load_dotenv()

DB_FAISS_PATH = "vector_db/"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

CUSTOM_PROMPT_TEMPLATE = """
You are a knowledgeable assistant specializing in vaccines in India. Below is the context from official vaccine data, followed by the user's question. Use the context to provide a clear, concise, and conversational answer that directly addresses the question’s intent. If the question asks for a count, provide only the number or a brief summary. If it asks for a list or details, provide those specifically without repeating unnecessary information. If the context doesn’t fully answer the question, say so honestly and avoid guessing.

Context: {context}
Question: {question}

Answer naturally, as if explaining to someone curious about vaccines in India.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, hf_token):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )

def main():
    st.title("टीका सहायक 🇮🇳💉")
    st.write("हिंदी में बोलें, मैं हिंदी में जवाब दूंगा!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'recording_count' not in st.session_state:
        st.session_state.recording_count = 0

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    st.write("माइक बटन दबाएं और हिंदी में बोलें:")
    with st.spinner("आवाज को टेक्स्ट में बदल रहे हैं..."):
        hindi_text = speech_to_text(language="hi", use_container_width=True, just_once=True, key=f"STT_{st.session_state.recording_count}")

    if hindi_text:
        translator = Translator()
        english_query = translator.translate(hindi_text, src='hi', dest='en').text

        if not HF_TOKEN:
            st.error("टोकन नहीं मिला।")
            return

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("डेटाबेस लोड नहीं हुआ।")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': english_query})
            english_answer = response["result"]
            hindi_answer = translator.translate(english_answer, src='en', dest='hi').text

            output_text = f"**प्रश्न**: {hindi_text}  \n**जवाब**: {hindi_answer}"
            st.chat_message('user').markdown(hindi_text)
            st.chat_message('assistant').markdown(output_text)
            st.session_state.messages.append({'role': 'user', 'content': hindi_text})
            st.session_state.messages.append({'role': 'assistant', 'content': output_text})

            with st.spinner("जवाब को आवाज में बदल रहे हैं..."):
                tts = gTTS(text=hindi_answer, lang="hi")
                tts.save("output.mp3")
                st.audio("output.mp3")

            # Reset for next recording
            st.session_state.recording_count += 1

        except Exception as e:
            st.error(f"कोई गड़बड़ी: {str(e)}")
    else:
        st.warning("कृपया माइक दबाएं और बोलें।")

if __name__ == "__main__":
    main()  # This calls main() to start the app
