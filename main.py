import os
import streamlit as st
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please define it in your .env file.")

# Set up Gemini-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Use Gemini model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Agent run configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# --- Language map ---
language_options = {
    "English": "English",
    "German": "German",
    "Arabic": "Arabic",
    "Hindi": "Hindi",
    "Urdu": "Urdu",
    "French": "French",
    "Spanish": "Spanish",
    "Chinese": "Chinese (Simplified)",
    "Russian": "Russian",
    "Japanese": "Japanese",
    "Korean": "Korean"
}

# Streamlit UI
st.set_page_config(page_title="Smart Translator", page_icon="🌐")
st.title("🌍 Google Translate-style Gemini Translator")
st.markdown("Type in any language and choose the target language for translation.")

# Input text
user_input = st.text_area("✍ Enter text (any language):", height=150)

# Target language selection
target_language = st.selectbox("🌐 Choose the language you want to translate *into*:", list(language_options.keys()))

# Async translation function
async def run_translation(text, lang):
    instructions = f"You are a translator. Translate the input text into {language_options[lang]}. Only return the translated sentence in {language_options[lang]} without extra explanation."
    translator_agent = Agent(name="Smart Translator", instructions=instructions, model=model)
    return await Runner.run(translator_agent, input=text, run_config=config)

# Button and translation logic
if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            try:
                response = asyncio.run(run_translation(user_input, target_language))
                st.success(f"Translated to {target_language}:")
                st.markdown("### 📝 Output")
                st.code(response.final_output, language='text')
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")