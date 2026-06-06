"""
AI Interview Coach - Enhanced Streamlit Application
Features:
- Resume & Preset-based contextual question generation via FLAN-T5
- Audio-to-Text via OpenAI Whisper (with interactive fallbacks)
- Multi-Input Interface (Voice Recording, Audio File Upload, Keyboard/Text Input)
- Semantic Answer Evaluation with SentenceTransformers & FAISS Vector Store
- Speech delivery heuristics (WPM Speech Rate & Multi-word Filler Word Density)
- STAR Framework Analysis (Situation/Task, Action, Result)
- DistilBERT Emotion Classification for raw vocal/verbal confidence evaluation
- Modern Animated Custom CSS Dashboard with Session Progress History tracking
"""

import os
import tempfile
import re
import io
import wave
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import faiss
import PyPDF2
import streamlit as st
import whisper
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Page Configuration
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Elegant CSS for animations, modern cards, and interactive visual polish
st.markdown("""
<style>
    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes pulseSoft {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    /* Modern Styled Components */
    .animate-fade-in {
        animation: fadeInUp 0.5s ease-out forwards;
    }
    
    .card-container {
        background-color: #fcfdfe;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .card-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border-color: #1E88E5;
    }

    .metric-box {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    .coach-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #e6fcf5 100%);
        border-left: 5px solid #10b981;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        animation: fadeInUp 0.7s ease-out forwards;
    }

    .highlight-filler {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
        border: 1px solid #fca5a5;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Model Loading with Heuristic Fallbacks ====================

@st.cache_resource
def load_whisper_model(model_size: str = "tiny"):
    """Load Whisper speech model cleanly with explicit CPU mapping"""
    try:
        return whisper.load_model(model_size, device="cpu")
    except Exception as e:
        st.sidebar.error(f"Error loading Whisper ({model_size}): {e}")
        return None

@st.cache_resource
def load_sentence_transformer():
    """Load core Sentence Transformer for indexing and similarity semantic analysis"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading sentence model: {e}")
        return None

@st.cache_resource
def load_t5_model():
    """Load google/flan-t5-small and its corresponding tokenizer"""
    try:
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading FLAN-T5 model: {e}")
        return None, None

@st.cache_resource
def load_emotion_pipeline():
    """Load text emotion analysis pipeline with native cpu fallbacks"""
    try:
        return pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None,
            device=-1 # CPU explicit
        )
    except Exception:
        # Fallback will use a local rules engine if this fails or times out
        return None

def generate_text(model, tokenizer, prompt, max_length=150, temperature=0.7, do_sample=True):
    """Encapsulated helper for T5 sequence-to-sequence response generation"""
    if model is None or tokenizer is None:
        return ""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=1
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==================== Fallback Heuristic Models ====================

def lexical_emotion_analyzer(text: str) -> List[Dict[str, Any]]:
    """Local backup rule engine if HF model fails to fetch or run"""
    confident_vocab = {"absolutely", "definitely", "achieved", "delivered", "led", "managed", "implemented", "solved", "designed", "growth", "results"}
    anxious_vocab = {"maybe", "guess", "probably", "sorry", "uncertain", "nervous", "sort of", "kind of", "try to"}
    
    words = set(text.lower().split())
    conf_score = len(words.intersection(confident_vocab))
    anx_score = len(words.intersection(anxious_vocab))
    
    if conf_score > anx_score:
        return [{"label": "joy", "score": 0.8}]
    elif anx_score > conf_score:
        return [{"label": "fear", "score": 0.7}]
    else:
        return [{"label": "neutral", "score": 0.9}]

# ==================== Core Operations ====================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract and parse document strings from standard PyPDF2 streams"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    """Split text into distinct contextual segments for FAISS mapping"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def build_faiss_index(chunks: List[str], embedder) -> Tuple[faiss.Index, List[str]]:
    """Index chunks on a flat L2 or cosine normalized vector topology"""
    if not chunks or embedder is None:
        return None, []

    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # L2 Normalized Cosine Space
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, chunks

def generate_questions_from_resume(resume_text: str, model, tokenizer, num_questions: int = 5) -> List[str]:
    """Contextually query FLAN-T5-small model using clean system instruction prompts"""
    truncated_text = resume_text[:1200]
    prompt = f"Analyze the context and write {num_questions} professional behavioral interview questions:\nResume Context: {truncated_text}\nQuestions:"
    
    # Pre-calculated structural backup options
    generic_options = [
        "Explain a time you managed a complicated project task under dynamic constraints.",
        "How do you approach learning complex technologies or systems within a brief timeline?",
        "Describe a design choice you initiated that resulted in positive progress.",
        "How do you resolve architectural or goal-oriented conflicts within your team?",
        "What strategies do you employ to manage project workloads during critical execution deadlines?"
    ]

    if model is None or tokenizer is None:
         return generic_options[:num_questions]

    try:
        response = generate_text(model, tokenizer, prompt, max_length=250, temperature=0.6, do_sample=True)
        questions = [q.strip() for q in response.split('\n') if len(q.strip()) > 10]
        
        # Strip potential numbering formatting
        cleaned_questions = [re.sub(r'^\d+[\.\-\)]\s*', '', q) for q in questions]
        cleaned_questions = [q for q in cleaned_questions if q.endswith('?') or len(q) > 15]

        if len(cleaned_questions) < num_questions:
            needed = num_questions - len(cleaned_questions)
            cleaned_questions.extend(generic_options[:needed])
        return cleaned_questions[:num_questions]
    except Exception:
        return generic_options[:num_questions]

# ==================== Evaluation Heuristics ====================

def detect_filler_words(text: str) -> Tuple[List[str], float]:
    """Parse vocal filler patterns using targeted word-boundary checks"""
    filler_patterns = [
        r'\bum\b', r'\buh\b', r'\blike\b', r'\bactually\b', r'\bbasically\b',
        r'\byou know\b', r'\bso\b', r'\bwell\b', r'\bright\b', r'\bok\b',
        r'\bhmm\b', r'\ber\b', r'\bah\b', r'\bkind of\b', r'\bsort of\b'
    ]
    text_lower = text.lower()
    filler_matches = []
    for pattern in filler_patterns:
        matches = re.findall(pattern, text_lower)
        filler_matches.extend(matches)

    total_words = len(text.split())
    filler_density = len(filler_matches) / total_words if total_words > 0 else 0
    return filler_matches, filler_density

def check_star_structure(text: str) -> Dict[str, Any]:
    """Determine whether response includes the crucial behavioral STAR indicators"""
    text_lower = text.lower()
    
    situation_hits = ["situation", "background", "context", "task", "assigned", "challenge", "initially", "responsible for"]
    action_hits = ["implemented", "built", "designed", "led", "managed", "executed", "developed", "focused on", "resolved"]
    result_hits = ["result", "outcome", "consequently", "achieved", "metrics", "delivered", "increased", "decreased", "impact"]
    
    has_sit = any(word in text_lower for word in situation_hits)
    has_act = any(word in text_lower for word in action_hits)
    has_res = any(word in text_lower for word in result_hits)
    
    score = sum([has_sit, has_act, has_res]) * 33.3
    score = min(100.0, score)
    
    return {
        "score": round(score, 1),
        "structure": {
            "Situation/Task": has_sit,
            "Action": has_act,
            "Result": has_res
        }
    }

def calculate_speech_rate(text: str, audio_duration_sec: float) -> float:
    """Calculate the delivery pace in Words Per Minute"""
    word_count = len(text.split())
    minutes = audio_duration_sec / 60.0
    return word_count / minutes if minutes > 0 else 0.0

def calculate_confidence_score(filler_density: float, speech_rate: float, emotion_result: List[Dict]) -> float:
    """Construct a composite verbal/lexical confidence score (0-100)"""
    filler_score = max(0.0, 100.0 - (filler_density * 100.0 * 2.5))
    
    # Ideal range mapping: 120 - 170 Words Per Minute
    if speech_rate < 90:
        rate_score = max(30.0, 100.0 - (90.0 - speech_rate) * 1.2)
    elif speech_rate > 190:
        rate_score = max(30.0, 100.0 - (speech_rate - 190.0) * 1.2)
    else:
        rate_score = 100.0

    emotion_map = {
        'joy': 1.0, 'neutral': 0.8, 'surprise': 0.7,
        'sadness': 0.4, 'fear': 0.3, 'anger': 0.5
    }
    
    if emotion_result:
        top_emotion = emotion_result[0]['label']
        emotion_factor = emotion_map.get(top_emotion, 0.6)
    else:
        emotion_factor = 0.6

    confidence = (filler_score * 0.4) + (rate_score * 0.3) + (emotion_factor * 100.0 * 0.3)
    return round(min(100.0, max(0.0, confidence)), 1)

def evaluate_answer(question: str, answer: str, resume_chunks: List[str], embedder, faiss_index) -> Dict[str, Any]:
    """Generate objective matching grades comparing answer semantic structures with resume nodes"""
    if embedder is None:
        return {
            'relevance': 70.0, 'completeness': 70.0, 'resume_alignment': 50.0, 'overall': 63.3,
            'strengths': ["Valid baseline validation processing active."], 'weaknesses': []
        }

    q_embedding = embedder.encode([question])[0]
    a_embedding = embedder.encode([answer])[0]

    q_embedding = q_embedding / np.linalg.norm(q_embedding)
    a_embedding = a_embedding / np.linalg.norm(a_embedding)

    # Convert native cosine space to representative linear domain [0.1, 0.7] -> [0, 100]
    raw_similarity = float(np.dot(q_embedding, a_embedding))
    relevance = max(10.0, min(100.0, (raw_similarity - 0.1) * 166.7))

    word_count = len(answer.split())
    completeness = min(100.0, (word_count / 120.0) * 100.0) if word_count > 0 else 0.0

    resume_similarity = 0.0
    if faiss_index is not None and resume_chunks:
        a_embedding_norm = a_embedding.reshape(1, -1)
        faiss.normalize_L2(a_embedding_norm)
        scores, _ = faiss_index.search(a_embedding_norm, 1)
        if scores[0][0] > 0:
            resume_similarity = float(scores[0][0]) * 100.0

    star_data = check_star_structure(answer)
    star_score = star_data["score"]

    overall = (relevance * 0.35) + (completeness * 0.25) + (resume_similarity * 0.2) + (star_score * 0.2)
    overall = min(100.0, max(0.0, overall))

    strengths = []
    weaknesses = []

    if relevance > 70.0:
        strengths.append("Answer aligns well with the conceptual focus of the question.")
    elif relevance < 45.0:
        weaknesses.append("Consider focusing the response directly on the core point of the prompt.")

    if completeness > 75.0:
        strengths.append("Provided a broad and structured answer explanation.")
    elif completeness < 40.0:
        weaknesses.append("The response was quite brief. Elaborate further on actions and details.")

    if resume_similarity > 70.0:
        strengths.append("Strongly leveraged skills profile from the resume.")
    elif resume_similarity < 40.0 and faiss_index is not None:
        weaknesses.append("Try referencing specific career achievements noted in your resume.")

    if star_score >= 66.0:
        strengths.append("Followed a clear structural sequence (STAR format).")
    else:
        weaknesses.append("Integrate STAR steps (describe the Situation, explain your Action, and state the final Result).")

    return {
        'relevance': round(relevance, 1),
        'completeness': round(completeness, 1),
        'resume_alignment': round(resume_similarity, 1),
        'overall': round(overall, 1),
        'star_score': star_score,
        'star_details': star_data["structure"],
        'strengths': strengths,
        'weaknesses': weaknesses
    }

def generate_personalized_feedback(question: str, answer: str, evaluation: Dict,
                                   filler_words: List[str], confidence_score: float,
                                   model, tokenizer) -> str:
    """Consolidate high-impact, actionable constructive coaching guidance using sequence models"""
    strengths_text = ", ".join(evaluation['strengths']) if evaluation['strengths'] else "Initial alignment checks look good."
    weaknesses_text = ", ".join(evaluation['weaknesses']) if evaluation['weaknesses'] else "No immediate structural challenges."
    filler_text = ", ".join(list(set(filler_words))[:4]) if filler_words else "None"

    prompt = f"""As an interview coach, provide detailed feedback (3 concise sentences) on this practice run.
Question: {question}
Answer: {answer}
Strengths: {strengths_text}
Improve: {weaknesses_text}
Fillers: {filler_text}
Confidence: {confidence_score}/100
Feedback:"""

    try:
        response = generate_text(model, tokenizer, prompt, max_length=150, temperature=0.6)
        if len(response.strip()) > 20:
            return response.strip()
        raise ValueError("Response length insufficient")
    except Exception:
        # Structured feedback generation fallback
        improve_guide = "Try to explicitly describe the 'Action' and 'Result' phases of your story." if evaluation['star_score'] < 66 else "Maintain this structured layout."
        return (f"Your delivery was evaluated at a baseline score of {evaluation['overall']}/100. "
                f"You showed healthy domain alignment. To refine this, {improve_guide} "
                f"Be careful with verbal filler terms such as '{filler_text}' to sound more confident.")

def get_audio_duration_from_bytes(audio_bytes: bytes) -> float:
    """Extract playback length from standard RIFF/WAV format byte frames"""
    try:
        with io.BytesIO(audio_bytes) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                if rate > 0:
                    return frames / float(rate)
    except Exception:
        pass
    # Baseline estimation fallback for 16-bit, 16000Hz mono formats
    return max(3.0, len(audio_bytes) / 32000.0)

def transcribe_audio(audio_bytes: bytes, whisper_model) -> Tuple[str, float]:
    """Transcribe audio with fallbacks to avoid application crashes on platform resource errors"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_recording_{np.random.randint(100000)}.wav")
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        duration = get_audio_duration_from_bytes(audio_bytes)

        if whisper_model is not None:
            result = whisper_model.transcribe(temp_path)
            transcript = result["text"].strip()
            return transcript, duration
        else:
            return "Unable to initialize local Whisper model on this cloud instance. Please utilize the Keyboard Input Mode option below to complete your evaluation.", duration
    except Exception as e:
        error_msg = str(e).lower()
        if "ffmpeg" in error_msg:
            st.error("⚠️ **System environment missing system FFmpeg drivers.**\n"
                     "You can still continue practicing by switching the selector below to **Keyboard Input Mode**.")
        else:
            st.error(f"Processing Error: {str(e)}")
        return "", 0.0
    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass

def highlight_filler_words(text: str, filler_words: List[str]) -> str:
    """Format matching vocal filler patterns using clear HTML highlighted tag wrappers"""
    highlighted = text
    for filler in sorted(set(filler_words), key=len, reverse=True):
        pattern = re.compile(rf'\b{re.escape(filler)}\b', re.IGNORECASE)
        highlighted = pattern.sub(f'<span class="highlight-filler">{filler}</span>', highlighted)
    return highlighted

# ==================== Streamlit Dashboard Panel Layout ====================

def main():
    # Animated Title layout
    st.markdown('<div class="animate-fade-in"><h1 style="color:#1E88E5; font-size: 2.5rem; text-align: center; font-weight:700; margin-bottom:0.5rem;">🎯 AI Interview Practice Coach</h1></div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>Interactive behavioral practice powered by Whisper ASR, SentenceTransformers, and the STAR Evaluation Framework</p>", unsafe_allow_html=True)

    # Initialize State Keys
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'resume_chunks' not in st.session_state:
        st.session_state.resume_chunks = []
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = []
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""
    if 'transcription_done' not in st.session_state:
        st.session_state.transcription_done = False
    if 'session_history' not in st.session_state:
        st.session_state.session_history = []

    # Model Resource Initialization
    with st.spinner("Initializing models... (first-time execution takes a moment)"):
        embedder = load_sentence_transformer()
        t5_model, t5_tokenizer = load_t5_model()
        emotion_pipeline = load_emotion_pipeline()

    # Sidebar: Setup & Resource Management
    with st.sidebar:
        st.markdown("<h3 style='color:#1E88E5; margin-bottom:1rem;'>⚙️ Configuration</h3>", unsafe_allow_html=True)
        
        # Whisper Memory Safe Selection
        whisper_sz = st.selectbox("Speech Model Footprint", ["tiny", "base"], index=0, help="Tiny loads and runs faster on limited CPU environments.")
        whisper_model = load_whisper_model(whisper_sz)

        st.divider()
        st.markdown("### 📋 Content Track Selection")
        prep_mode = st.radio("Choose Practice Source", ["Predefined Career Tracks", "Upload Custom Resume"], index=0)

        if prep_mode == "Upload Custom Resume":
            resume_file = st.file_uploader("Upload PDF or TXT resume", type=["pdf", "txt"])
            if resume_file is not None:
                with st.spinner("Processing document chunks..."):
                    if resume_file.type == "application/pdf":
                        resume_text = extract_text_from_pdf(resume_file)
                    else:
                        resume_text = resume_file.read().decode("utf-8")

                    st.session_state.resume_text = resume_text
                    chunks = chunk_text(resume_text)
                    index, chunks_ret = build_faiss_index(chunks, embedder)
                    st.session_state.faiss_index = index
                    st.session_state.resume_chunks = chunks_ret

                    st.session_state.generated_questions = generate_questions_from_resume(
                        resume_text, t5_model, t5_tokenizer, num_questions=5
                    )
                    st.session_state.resume_processed = True
                    st.success(f"Successfully chunked {len(chunks)} document nodes")
        else:
            # Career Presets Track logic
            presets = {
                "Software Engineering Track": [
                    "How do you ensure system scalability and code design elegance during aggressive deploy deadlines?",
                    "Describe a time you had to troubleshoot a complex, memory-intensive technical system issue.",
                    "Explain your criteria for selecting database structures or API specifications for a high-traffic system."
                ],
                "Data Science / Machine Learning": [
                    "How do you evaluate objective trade-offs between simpler models and highly complex neural networks?",
                    "Describe your process for structuring an analysis or model training when data is chaotic or missing.",
                    "How do you explain technical model performance configurations to business stakeholders?"
                ],
                "Product Management": [
                    "How do you prioritize your product features roadmap when negotiating conflicting stakeholder requests?",
                    "Describe a feature or product release you owned that did not meet goals. What insights did you acquire?",
                    "How do you leverage qualitative metrics and target user behaviors to evaluate design choices?"
                ],
                "Behavioral & General HR": [
                    "Explain a time you resolved a major workflow or prioritization conflict within your team.",
                    "Tell me about a difficult task decision you made when your key data inputs were incomplete.",
                    "How do you organize your tasks when multiple deadlines must be met concurrently?"
                ]
            }
            selected_track = st.selectbox("Select Target Role Preset", list(presets.keys()))
            if st.button("Load Preset Tracks", type="secondary"):
                st.session_state.generated_questions = presets[selected_track]
                st.session_state.resume_processed = True
                st.session_state.resume_chunks = []
                st.session_state.faiss_index = None
                st.success(f"Loaded questions for {selected_track}")

        st.divider()
        st.markdown("### 🛠️ Practice Guidelines")
        st.info("💡 **STAR Strategy Rule**:\n- **Situation/Task**: State the operational scenario.\n- **Action**: Outline your physical problem-solving measures.\n- **Result**: Highlight quantifiable achievements and performance stats.")

    # Practice Cockpit View
    if not st.session_state.resume_processed:
        st.info("👈 Choose a Career Track Preset or Upload your Resume in the left panel to begin.")
        return

    st.markdown('<div class="card-container animate-fade-in">', unsafe_allow_html=True)
    st.subheader("📋 Select active Practice Question")
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        selected_question = st.selectbox(
            "Target behavioral prompt:",
            options=st.session_state.generated_questions,
            index=0
        )
        st.session_state.selected_question = selected_question
    with col_q2:
        if st.button("🔄 Refresh Options", use_container_width=True):
            if prep_mode == "Upload Custom Resume" and st.session_state.resume_text:
                with st.spinner("Regenerating..."):
                    st.session_state.generated_questions = generate_questions_from_resume(
                        st.session_state.resume_text, t5_model, t5_tokenizer, num_questions=5
                    )
                st.rerun()
            else:
                st.info("Reload career preset tracks from sidebar to update preset lists.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Core Input Setup
    st.subheader("🎙️ Input Practice Response")
    input_type = st.radio("Response Delivery Mode", ["Audio Recording", "Audio File Upload", "Keyboard / Text Input"], index=0, horizontal=True)

    audio_bytes = None
    typed_response = ""

    if input_type == "Audio Recording":
        recorded_audio = st.audio_input("Record your answer below:")
        if recorded_audio:
            audio_bytes = recorded_audio.getvalue()
    elif input_type == "Audio File Upload":
        uploaded_audio = st.file_uploader("Import audio file (WAV or MP3)", type=["wav", "mp3"])
        if uploaded_audio:
            audio_bytes = uploaded_audio.read()
            st.audio(audio_bytes, format="audio/wav")
    else:
        typed_response = st.text_area("Type or paste your response below:", height=150, placeholder="In my last role, we were challenged with...")

    # Evaluation Trigger Process
    trigger_eval = False
    if input_type in ["Audio Recording", "Audio File Upload"] and audio_bytes:
        trigger_eval = st.button("🔍 Analyze Audio Response", type="primary", use_container_width=True)
    elif input_type == "Keyboard / Text Input" and len(typed_response.strip()) > 10:
        trigger_eval = st.button("🔍 Analyze Text Response", type="primary", use_container_width=True)

    if trigger_eval:
        with st.spinner("Decoding and grading response metrics..."):
            if input_type in ["Audio Recording", "Audio File Upload"]:
                transcript, duration = transcribe_audio(audio_bytes, whisper_model)
            else:
                transcript = typed_response
                duration = len(typed_response.split()) / 2.3  # Estimated duration mapping (average pacing)

            if transcript:
                filler_words, filler_density = detect_filler_words(transcript)
                speech_rate = calculate_speech_rate(transcript, duration)
                
                # Active emotion analysis execution
                if emotion_pipeline is not None:
                    try:
                        emotion_data = emotion_pipeline(transcript)[0]
                    except Exception:
                        emotion_data = lexical_emotion_analyzer(transcript)
                else:
                    emotion_data = lexical_emotion_analyzer(transcript)

                confidence_score = calculate_confidence_score(filler_density, speech_rate, emotion_data)

                evaluation = evaluate_answer(
                    st.session_state.selected_question,
                    transcript,
                    st.session_state.resume_chunks,
                    embedder,
                    st.session_state.faiss_index
                )

                feedback_text = generate_personalized_feedback(
                    st.session_state.selected_question,
                    transcript,
                    evaluation,
                    filler_words,
                    confidence_score,
                    t5_model,
                    t5_tokenizer
                )

                # Save history logs
                st.session_state.transcription_done = True
                st.session_state.transcript = transcript
                st.session_state.filler_words = filler_words
                st.session_state.filler_density = filler_density
                st.session_state.speech_rate = speech_rate
                st.session_state.confidence_score = confidence_score
                st.session_state.evaluation = evaluation
                st.session_state.feedback = feedback_text
                st.session_state.duration = duration
                st.session_state.emotion = emotion_data[0]['label'] if emotion_data else "neutral"

                # Append current practice run data to history array
                st.session_state.session_history.append({
                    "question": st.session_state.selected_question,
                    "score": evaluation["overall"],
                    "confidence": confidence_score,
                    "rate": int(speech_rate) if input_type != "Keyboard / Text Input" else "N/A",
                    "emotion": st.session_state.emotion.title()
                })
            else:
                st.error("No valid text output generated for evaluation. Check inputs and verify audio clarity.")

    # Results Section
    if st.session_state.get('transcription_done', False):
        st.divider()
        st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
        st.subheader("📊 Session Scorecard")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Structure & Content", f"{st.session_state.evaluation['overall']}/100")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Delivery Confidence", f"{st.session_state.confidence_score}/100")
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            if input_type != "Keyboard / Text Input":
                st.metric("Speech Tempo", f"{int(st.session_state.speech_rate)} WPM", 
                          delta="Optimal (120-170)" if 120 <= st.session_state.speech_rate <= 170 else "Adjust Tempo", 
                          delta_color="normal")
            else:
                st.metric("Speech Tempo", "N/A (Typed)")
            st.markdown('</div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Core Emotion Key", st.session_state.emotion.title())
            st.markdown('</div>', unsafe_allow_html=True)

        # Transcript display
        st.subheader("📝 Processed Output Transcript")
        highlighted = highlight_filler_words(st.session_state.transcript, st.session_state.filler_words)
        st.markdown(f'<div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 1.25rem; border-radius: 8px; line-height: 1.6; margin-bottom: 1rem;">{highlighted}</div>', unsafe_allow_html=True)

        if st.session_state.filler_words:
            st.warning(f"⚠️ Filler word patterns detected: {', '.join(set(st.session_state.filler_words))} (Density: {st.session_state.filler_density*100:.1f}%)")
        else:
            st.success("✨ Excellent structural vocabulary! No major verbal fillers detected.")

        # Breakdown Matrix
        st.subheader("🎯 Context & Structural Matrix")
        tab_eval, tab_star = st.tabs(["Alignment Analysis", "STAR Structure Verification"])
        
        with tab_eval:
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.metric("Topic Relevance", f"{st.session_state.evaluation['relevance']}/100")
                st.progress(st.session_state.evaluation['relevance']/100)
            with ec2:
                st.metric("Context Completeness", f"{st.session_state.evaluation['completeness']}/100")
                st.progress(st.session_state.evaluation['completeness']/100)
            with ec3:
                st.metric("Document Vector Match", f"{st.session_state.evaluation['resume_alignment']}/100")
                st.progress(st.session_state.evaluation['resume_alignment']/100)

        with tab_star:
            st.markdown(f"**Overall STAR Quality Rating:** `{st.session_state.evaluation['star_score']}/100`")
            sc1, sc2, sc3 = st.columns(3)
            for col, (phase, hit) in zip([sc1, sc2, sc3], st.session_state.evaluation['star_details'].items()):
                with col:
                    status_icon = "✅ Passed" if hit else "❌ Missing"
                    color_style = "color:#10b981;" if hit else "color:#f43f5e;"
                    st.markdown(f"<div class='metric-box'><h5 style='margin:0;'>{phase}</h5><p style='font-size:1.2rem; font-weight:bold; {color_style}'>{status_icon}</p></div>", unsafe_allow_html=True)

        # Direct Feedback
        st.subheader("💡 Personalized Coaching Notes")
        st.markdown(f'<div class="coach-card">{st.session_state.feedback}</div>', unsafe_allow_html=True)

        # Pros and Cons Column View
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            st.markdown("<strong style='color:#10b981;'>💪 Observed Strengths:</strong>", unsafe_allow_html=True)
            for strength in st.session_state.evaluation['strengths']:
                st.markdown(f"- {strength}")
        with sc_col2:
            st.markdown("<strong style='color:#f43f5e;'>⚠️ Suggested Areas of Focus:</strong>", unsafe_allow_html=True)
            for weakness in st.session_state.evaluation['weaknesses']:
                st.markdown(f"- {weakness}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Historic Performance Tracker
    if len(st.session_state.session_history) > 0:
        st.divider()
        st.subheader("📈 Session Progress Dashboard")
        st.markdown("<p style='color:#64748b;'>Review your continuous progress performance metrics across practiced prompts.</p>", unsafe_allow_html=True)
        
        # Display chronological practice timeline table
        history_data = []
        for idx, run in enumerate(st.session_state.session_history):
            history_data.append({
                "Attempt #": idx + 1,
                "Question Prompt": run["question"][:65] + "...",
                "Content Score": f"{run['score']}/100",
                "Confidence Grade": f"{run['confidence']}/100",
                "Delivery Pace": f"{run['rate']} WPM" if run['rate'] != "N/A" else "N/A",
                "Predominant Emotion": run["emotion"]
            })
        st.table(history_data)

        if st.button("Clear Session Progress History"):
            st.session_state.session_history = []
            st.rerun()

if __name__ == "__main__":
    main()