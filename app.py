import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def load_gemini_llm():
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not found.")
    return genai.GenerativeModel("gemini-2.0-flash")  # Verify model availability

def clean_text(text):
    return text.replace("**", "").replace("*", "").strip()

def build_context(history, max_messages=5):
    # [Unchanged context-building logic]
    context = ""
    recent_history = history[-max_messages * 2:] if len(history) > max_messages * 2 else history
    vaccination_keywords = ["vacc", "टीक", "ভ্যাক", "વેક", "ವ್ಯಾಕ", "വാക്സ", "वैक्स", "వాక్స", "ویکس", "ਵੈਕ", "measles", "खसरा", "rubella", "रूबेला", "covid", "vitamin", "विटामिन"]
    follow_up_keywords = ["where", "कहाँ", "online", "ऑनलाइन", "get", "पाएं", "this", "यह", "child", "बच्चा", "age", "उम्र", "portal", "पोर्टाल", "is", "क्या", "does", "दिया", "given", "supplement", "सप्लीमेंट", "necessary", "जरूरी"]
    rejection_phrases = ["केवल टीकाकरण से संबंधित", "Ask me only vaccination-related", "শুধুমাত্র ভ্যাকসিনেশন"]
    is_vaccination_context = False
    i = 0

    while i < len(recent_history) - 1:
        user_msg = recent_history[i]
        ai_msg = recent_history[i + 1] if i + 1 < len(recent_history) else None
        
        if user_msg['role'] == 'user' and ai_msg and ai_msg['role'] == 'assistant':
            ai_content_lower = ai_msg['content'].lower()
            if any(phrase in ai_content_lower for phrase in rejection_phrases):
                i += 2
                continue
        
        user_content_lower = user_msg['content'].lower()
        if any(keyword in user_content_lower for keyword in vaccination_keywords) or \
           (is_vaccination_context and any(keyword in user_content_lower for keyword in follow_up_keywords)):
            context += f"Question: {user_msg['content']}\n"
            if ai_msg and ai_msg['role'] == 'assistant':
                context += f"Answer: {ai_msg['content']}\n"
            is_vaccination_context = True
            i += 2
        else:
            is_vaccination_context = False
            i += 1

    return context.strip()

def main():
    ui_text = {
        "Hindi": {
            "title": "टीका सहायक 🇮🇳💉",
            "subtitle": "भारत की भाषाओं में बोलें और जवाब पाएं!",
            "lang_label": "जवाब की भाषा चुनें:",
            "mic_prompt": "माइक बटन दबाएं और बोलें:",
            "stt_spinner": "आवाज को टेक्स्ट में बदल रहे हैं...",
            "tts_spinner": "जवाब को आवाज में बदल रहे हैं...",
            "warning": "कृपया माइक दबाएं और बोलें।"
        },
        # [Other languages unchanged]
        "English": {
            "title": "Vaccination Assistant 🇮🇳💉",
            "subtitle": "Speak in India's languages and get answers!",
            "lang_label": "Choose response language:",
            "mic_prompt": "Press the mic button and speak:",
            "stt_spinner": "Converting voice to text...",
            "tts_spinner": "Converting answer to voice...",
            "warning": "Please press the mic and speak."
        }
        # Add other languages as needed
    }

    if 'messages' not_in st.session_state:
        st.session_state.messages = []
    if 'recording_count' not_in st.session_state:
        st.session_state.recording_count = 0
    if 'recording_active' not_in st.session_state:
        st.session_state.recording_active = False
    if 'selected_lang' not_in st.session_state:
        st.session_state.selected_lang = "Hindi"
    if 'audio_data' not_in st.session_state:
        st.session_state.audio_data = None  # Store audio bytes

    languages = {
        "Hindi": "hi", "English": "en", "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn",
        "Malayalam": "ml", "Marathi": "mr", "Tamil": "ta", "Telugu": "te", "Urdu": "ur", "Punjabi": "pa"
    }

    st.title(ui_text[st.session_state.selected_lang]["title"])
    st.write(ui_text[st.session_state.selected_lang]["subtitle"])

    lang_keys = list(languages.keys())
    selected_lang = st.selectbox(
        "जवाब की भाषा चुनें:",  # Static Hindi label
        lang_keys,
        index=lang_keys.index("Hindi")
    )
    st.session_state.selected_lang = selected_lang
    lang_code = languages[selected_lang]

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    st.write(ui_text[selected_lang]["mic_prompt"])
    with st.spinner(ui_text[selected_lang]["stt_spinner"]):
        raw_text = speech_to_text(
            language=lang_code,
            use_container_width=True,
            key=f"STT_{st.session_state.recording_count}"
        )

    if raw_text and not st.session_state.recording_active:
        try:
            st.session_state.recording_active = True
            llm = load_gemini_llm()
            context = build_context(st.session_state.messages, max_messages=5)

            rejection_messages = {
                "Hindi": "केवल टीकाकरण से संबंधित प्रश्न पूछें।",
                "English": "Ask me only vaccination-related questions.",
                # [Other languages unchanged]
            }

            prompt = (
                f"You are a vaccination assistant for India. Answer all questions in {selected_lang}, including rejections. "
                "A question is vaccination-related if it contains terms like 'vaccine', 'vaccination', 'टीकाकरण', 'measles', 'खसरा', 'rubella', 'रूबेला', 'covid', 'vitamin', 'विटामिन', "
                "or follow-up terms like 'where', 'कहाँ', 'get', 'पाएं', 'this', 'यह', 'child', 'बच्चा', 'is', 'क्या', 'does', 'दिया', 'given', 'supplement', 'सप्लीमेंट', 'necessary', 'जरूरी' "
                "after a vaccination question. Use the context below only if the question relates to it. "
                f"If the question is not vaccination-related, respond with exactly: '{rejection_messages.get(selected_lang, 'Ask me only vaccination-related questions.')}' and ignore context. "
                f"Previous Vaccination-Related Context (if relevant):\n{context}\n\n"
                f"Current Question: {raw_text}"
            )

            response = llm.generate_content(prompt)
            answer = clean_text(response.text)

            output_text = f"{raw_text}  \n{answer}"
            st.chat_message('user').markdown(raw_text)
            st.chat_message('assistant').markdown(output_text)
            st.session_state.messages.append({'role': 'user', 'content': raw_text})
            st.session_state.messages.append({'role': 'assistant', 'content': answer})

            with st.spinner(ui_text[selected_lang]["tts_spinner"]):
                # Generate audio in memory instead of saving to disk
                tts = gTTS(text=answer, lang=lang_code, tld="co.in")
                audio_buffer = BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                st.session_state.audio_data = audio_buffer.read()  # Store bytes in session state

            # Play audio from memory
            if st.session_state.audio_data:
                st.audio(st.session_state.audio_data, format="audio/mp3")

            st.session_state.recording_count += 1
            st.session_state.recording_active = False
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")  # Localized error message can be added
            st.session_state.recording_active = False
            st.rerun()
    elif not raw_text and st.session_state.recording_active:
        st.session_state.recording_active = False
        st.rerun()
    else:
        st.warning(ui_text[selected_lang]["warning"])

if __name__ == "__main__":
    main()
