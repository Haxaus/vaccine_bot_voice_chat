import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
genai.configure(api_key=GEMINI_API_KEY)

def load_gemini_llm():
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not found.")
    return genai.GenerativeModel("gemini-2.0-flash")

def clean_text(text):
    return text.replace("**", "").replace("*", "").strip()

def build_context(history, max_messages=5):
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
    # Language dictionary for UI text
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
        "English": {
            "title": "Vaccination Assistant 🇮🇳💉",
            "subtitle": "Speak in India's languages and get answers!",
            "lang_label": "Choose response language:",
            "mic_prompt": "Press the mic button and speak:",
            "stt_spinner": "Converting voice to text...",
            "tts_spinner": "Converting answer to voice...",
            "warning": "Please press the mic and speak."
        },
        "Bengali": {
            "title": "টিকা সহায়ক 🇮🇳💉",
            "subtitle": "ভারতের ভাষায় কথা বলুন এবং উত্তর পান!",
            "lang_label": "উত্তরের ভাষা নির্বাচন করুন:",
            "mic_prompt": "মাইক বোতাম টিপুন এবং কথা বলুন:",
            "stt_spinner": "কণ্ঠস্বরকে টেক্সটে রূপান্তর করা হচ্ছে...",
            "tts_spinner": "উত্তরকে কণ্ঠস্বরে রূপান্তর করা হচ্ছে...",
            "warning": "দয়া করে মাইক টিপুন এবং কথা বলুন।"
        },
        "Gujarati": {
            "title": "રસી સહાયક 🇮🇳💉",
            "subtitle": "ભારતની ભાષાઓમાં બોલો અને જવાબ મેળવો!",
            "lang_label": "જવાબની ભાષા પસંદ કરો:",
            "mic_prompt": "માઇક બટન દબાવો અને બોલો:",
            "stt_spinner": "અવાજને ટેક્સ્ટમાં રૂપાંતરિત કરી રહ્યાં છીએ...",
            "tts_spinner": "જવાબને અવાજમાં રૂપાંતરિત કરી રહ્યાં છીએ...",
            "warning": "કૃપા કરીને માઇક દબાવો અને બોલો।"
        },
        "Kannada": {
            "title": "ಲಸಿಕೆ ಸಹಾಯಕ 🇮🇳💉",
            "subtitle": "ಭಾರತದ ಭಾಷೆಗಳಲ್ಲಿ ಮಾತನಾಡಿ ಮತ್ತು ಉತ್ತರ ಪಡೆಯಿರಿ!",
            "lang_label": "ಪ್ರತಿಕ್ರಿಯೆ ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ:",
            "mic_prompt": "ಮೈಕ್ ಬಟನ್ ಒತ್ತಿ ಮಾತನಾಡಿ:",
            "stt_spinner": "ಧ್ವನಿಯನ್ನು ಪಠ್ಯಕ್ಕೆ ಪರಿವರ್ತಿಸುತ್ತಿದ್ದೇವೆ...",
            "tts_spinner": "ಉತ್ತರವನ್ನು ಧ್ವನಿಗೆ ಪರಿವರ್ತಿಸುತ್ತಿದ್ದೇವೆ...",
            "warning": "ದಯವಿಟ್ಟು ಮೈಕ್ ಒತ್ತಿ ಮಾತನಾಡಿ।"
        },
        "Malayalam": {
            "title": "വാക്സിൻ സഹായി 🇮🇳💉",
            "subtitle": "ഇന്ത്യയുടെ ഭാഷകളിൽ സംസാരിക്കുകയും ഉത്തരങ്ങൾ നേടുകയും ചെയ്യുക!",
            "lang_label": "പ്രതികരണ ഭാഷ തിരഞ്ഞെടുക്കുക:",
            "mic_prompt": "മൈക്ക് ബട്ടൺ അമർത്തി സംസാരിക്കുക:",
            "stt_spinner": "ശബ്ദത്തെ വാചകമാക്കി മാറ്റുന്നു...",
            "tts_spinner": "ഉത്തരത്തെ ശബ്ദമാക്കി മാറ്റുന്നു...",
            "warning": "ദയവായി മൈക്ക് അമർത്തി സംസാരിക്കുക।"
        },
        "Marathi": {
            "title": "लसीकरण सहाय्यक 🇮🇳💉",
            "subtitle": "भारताच्या भाषांमध्ये बोला आणि उत्तरे मिळवा!",
            "lang_label": "उत्तराची भाषा निवडा:",
            "mic_prompt": "माइक बटण दाबा आणि बोला:",
            "stt_spinner": "आवाजाला मजकुरात रूपांतरित करत आहोत...",
            "tts_spinner": "उत्तराला आवाजात रूपांतरित करत आहोत...",
            "warning": "कृपया माइक दाबा आणि बोला।"
        },
        "Tamil": {
            "title": "தடுப்பூசி உதவியாளர் 🇮🇳💉",
            "subtitle": "இந்தியாவின் மொழிகளில் பேசி பதில்களைப் பெறுங்கள்!",
            "lang_label": "பதில் மொழியைத் தேர்ந்தெடுக்கவும்:",
            "mic_prompt": "மைக் பொத்தானை அழுத்தி பேசுங்கள்:",
            "stt_spinner": "குரலை உரையாக மாற்றுகிறோம்...",
            "tts_spinner": "பதிலை குரலாக மாற்றுகிறோம்...",
            "warning": "தயவுசெய்து மைக் அழுத்தி பேசுங்கள்।"
        },
        "Telugu": {
            "title": "వాక్సిన్ సహాయకుడు 🇮🇳💉",
            "subtitle": "భారతదేశ భాషలలో మాట్లాడండి మరియు సమాధానాలు పొందండి!",
            "lang_label": "సమాధాన భాషను ఎంచుకోండి:",
            "mic_prompt": "మైక్ బటన్ నొక్కి మాట్లాడండి:",
            "stt_spinner": "వాయిస్‌ను టెక్స్ట్‌గా మారుస్తున్నాము...",
            "tts_spinner": "సమాధానాన్ని వాయిస్‌గా మారుస్తున్నాము...",
            "warning": "దయచేసి మైక్ నొక్కి మాట్లాడండి।"
        },
        "Urdu": {
            "title": "ویکسین اسسٹنٹ 🇮🇳💉",
            "subtitle": "بھارت کی زبانوں میں بولیں اور جوابات حاصل کریں!",
            "lang_label": "جواب کی زبان منتخب کریں:",
            "mic_prompt": "مائک بٹن دبائیں اور بولیں:",
            "stt_spinner": "آواز کو متن میں تبدیل کر رہے ہیں...",
            "tts_spinner": "جواب کو آواز میں تبدیل کر رہے ہیں...",
            "warning": "براہ کرم مائک دبائیں اور بولیں۔"
        },
        "Punjabi": {
            "title": "ਟੀਕਾਕਰਨ ਸਹਾਇਕ 🇮🇳💉",
            "subtitle": "ਭਾਰਤ ਦੀਆਂ ਭਾਸ਼ਾਵਾਂ ਵਿੱਚ ਬੋਲੋ ਅਤੇ ਜਵਾਬ ਪ੍ਰਾਪਤ ਕਰੋ!",
            "lang_label": "ਜਵਾਬ ਦੀ ਭਾਸ਼ਾ ਚੁਣੋ:",
            "mic_prompt": "ਮਾਈਕ ਬਟਨ ਦਬਾਓ ਅਤੇ ਬੋਲੋ:",
            "stt_spinner": "ਆਵਾਜ਼ ਨੂੰ ਟੈਕਸਟ ਵਿੱਚ ਬਦਲ ਰਹੇ ਹਾਂ...",
            "tts_spinner": "ਜਵਾਬ ਨੂੰ ਆਵਾਜ਼ ਵਿੱਚ ਬਦਲ ਰਹੇ ਹਾਂ...",
            "warning": "ਕਿਰਪਾ ਕਰਕੇ ਮਾਈਕ ਦਬਾਓ ਅਤੇ ਬੋਲੋ।"
        }
    }

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'recording_count' not in st.session_state:
        st.session_state.recording_count = 0
    if 'recording_active' not in st.session_state:
        st.session_state.recording_active = False

    # Set Hindi as default language initially
    if 'selected_lang' not in st.session_state:
        st.session_state.selected_lang = "Hindi"

    languages = {
        "Hindi": "hi", "English": "en", "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn",
        "Malayalam": "ml", "Marathi": "mr", "Tamil": "ta", "Telugu": "te", "Urdu": "ur", "Punjabi": "pa"
    }
       

    lang_keys = list(languages.keys())
    selected_lang = st.selectbox(
        "जवाब की भाषा चुनें:",  # Static Hindi label
        lang_keys,
        index=lang_keys.index("Hindi")  # Pre-select Hindi
    )
    st.session_state.selected_lang = selected_lang  # Update session state
    lang_code = languages[selected_lang]

    # Use session state to persist selected language, default to Hindi
    st.title(ui_text[st.session_state.selected_lang]["title"])
    st.write(ui_text[st.session_state.selected_lang]["subtitle"])
    
    

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
                "Bengali": "শুধুমাত্র ভ্যাকসিনেশন সম্পর্কিত প্রশ্ন জিজ্ঞাসা করুন।",
                "Gujarati": "મને ફક્ત રસીકરણ સંબંધિત પ્રશ્નો પૂછો।",
                "Kannada": "ನನಗೆ ಕೇವಲ ಲಸಿಕೆ ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ।",
                "Malayalam": "വാക്സിനേഷനുമായി ബന്ധപ്പെട്ട ചോദ്യങ്ങൾ മാത്രം ചോദിക്കുക।",
                "Marathi": "मला फक्त लसीकरणाशी संबंधित प्रश्न विचारा।",
                "Tamil": "தடுப்பூசி தொடர்பான கேள்விகளை மட்டும் கேளுங்கள்।",
                "Telugu": "నాకు కేవలం వాక్సినేషన్ సంబంధిత ప్రశ్నలు మాత్రమే అడగండి।",
                "Urdu": "مجھ سے صرف ویکسینیشن سے متعلق سوالات پوچھیں۔",
                "Punjabi": "ਮੈਨੂੰ ਸਿਰਫ ਟੀਕਾਕਰਨ ਨਾਲ ਸਬੰਧਤ ਸਵਾਲ ਪੁੱਛੋ।"
            }

            prompt = (
                f"You are a vaccination assistant for India. Answer all questions in {selected_lang}, including rejections. "
                "A question is vaccination-related if it contains terms like 'vaccine', 'vaccination', 'टीकाकरण', 'measles', 'खसरा', 'rubella', 'रूबेला', 'covid', 'vitamin', 'विटामिन', "
                "or follow-up terms like 'where', 'कहाँ', 'get', 'पाएं', 'this', 'यह', 'child', 'बच्चा', 'is', 'क्या', 'does', 'दिया', 'given', 'supplement', 'सप्लीमेंट', 'necessary', 'जरूरी' "
                "after a vaccination question. Use the context below only if the question relates to it. "
                "If the question is not vaccination-related, respond with exactly: '{rejection_messages[selected_lang]}' and ignore context. "
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
                tts = gTTS(text=answer, lang=lang_code, tld="co.in")
                tts.save("output.mp3")
                st.audio("output.mp3")

            st.session_state.recording_count += 1
            st.session_state.recording_active = False
            st.rerun()

        except Exception as e:
            st.error(f"कोई गड़बड़ी: {str(e)}")  # Consider localizing if desired
            st.session_state.recording_active = False
            st.rerun()
    elif not raw_text and st.session_state.recording_active:
        st.session_state.recording_active = False
        st.rerun()
    else:
        st.warning(ui_text[selected_lang]["warning"])

if __name__ == "__main__":
    main()
