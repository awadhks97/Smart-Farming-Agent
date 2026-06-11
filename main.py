```python
# =====================================================
# AgriSense AI - Smart Farming Agent
# FDP Project Demo
# =====================================================

import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# =====================================================
# LOAD PDF KNOWLEDGE BASE
# =====================================================

PDF_FILE = "ICAR_Guidelines.pdf"

loader = PyPDFLoader(PDF_FILE)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents,
    embedding
)

retriever = vectordb.as_retriever()

# =====================================================
# WEATHER AGENT (NO API REQUIRED)
# =====================================================

def get_weather(city):

    weather_data = {

        "Vadodara": {
            "temp": 36,
            "humidity": 65,
            "condition": "Sunny"
        },

        "Ahmedabad": {
            "temp": 38,
            "humidity": 55,
            "condition": "Partly Cloudy"
        },

        "Surat": {
            "temp": 33,
            "humidity": 75,
            "condition": "Cloudy"
        },

        "Rajkot": {
            "temp": 35,
            "humidity": 60,
            "condition": "Sunny"
        }
    }

    if city not in weather_data:
        return {
            "temp": 32,
            "humidity": 60,
            "condition": "Normal"
        }

    return weather_data[city]

# =====================================================
# IRRIGATION AGENT
# =====================================================

def irrigation_advice(temp, humidity):

    if temp > 35:
        return "High temperature detected. Irrigation recommended today."

    elif humidity > 80:
        return "High humidity. Delay irrigation."

    else:
        return "Normal weather. Follow regular irrigation schedule."

# =====================================================
# RAG AGENT
# =====================================================

def search_agri_knowledge(question):

    docs = retriever.get_relevant_documents(question)

    context = "\n".join(
        [doc.page_content for doc in docs[:3]]
    )

    return context

# =====================================================
# SMART FARMING AGENT
# =====================================================

def smart_farming_agent(question, city):

    weather = get_weather(city)

    temp = weather["temp"]
    humidity = weather["humidity"]

    irrigation = irrigation_advice(
        temp,
        humidity
    )

    knowledge = search_agri_knowledge(
        question
    )

    result = f"""
🌱 AGRISENSE AI SMART FARMING ADVISOR

Farmer Query:
{question}

---------------------------------

📍 Location:
{city}

🌦 Weather Information

Temperature: {temp} °C
Humidity: {humidity} %
Condition: {weather['condition']}

---------------------------------

💧 Irrigation Recommendation

{irrigation}

---------------------------------

📚 Agricultural Knowledge

{knowledge}

---------------------------------

✅ Final Recommendation

Based on agricultural guidelines and
current weather conditions, follow
the above recommendations carefully.
"""

    return result

# =====================================================
# GRADIO INTERFACE
# =====================================================

demo = gr.Interface(
    fn=smart_farming_agent,

    inputs=[
        gr.Textbox(
            label="Farmer Question"
        ),

        gr.Textbox(
            label="City"
        )
    ],

    outputs="text",

    title="AgriSense AI - Smart Farming Agent",

    description="Agentic AI + RAG based Smart Farming Advisory System"
)

demo.launch()
```
