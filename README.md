AI Interview Practice Coach
The AI Interview Practice Coach is an interactive Streamlit application designed to help candidates prepare for behavioral and technical interviews. It evaluates user responses by analyzing vocal delivery metrics, testing structural alignment against the STAR (Situation, Task, Action, Result) methodology, assessing resume relevance, and providing actionable coaching feedback.
The application is structured to run efficiently on standard CPU resources, utilizing lightweight models alongside modular fallback mechanics to remain functional under hardware or API constraints.
Key Features
Context-Driven Question Formulation:
Generate customized, career-specific questions by uploading a resume (PDF or TXT). Text is split into overlapping chunks and indexed using a FAISS vector store.
Alternatively, select from curated preset practice tracks covering Software Engineering, Data Science, Product Management, and Behavioral/HR roles.
Flexible Answer Input Modes:
Audio Recording: Record audio directly in the browser utilizing Streamlit's native audio input capabilities.
Audio Upload: Upload pre-recorded interview files in standard WAV or MP3 formats.
Keyboard Input: Write or paste text directly to test responses silently or bypass hardware/driver issues.
Speech Delivery Metrics:
Automatically computes delivery pace in Words Per Minute (WPM).
Identifies multi-word verbal fillers (e.g., "um", "you know", "basically") and highlights them directly within the output transcript.
STAR Framework Evaluation:
Scans verbal patterns for structured situational context (Situation/Task), programmatic steps (Action), and concrete metrics (Result).
Sentiment & Confidence Assessment:
Leverages a DistilBERT emotion classifier to analyze verbal confidence levels, switching to a rule-based lexical helper if system memory constraints prevent model initialization.
Session Progress Dashboard:
Temporarily caches results locally in session state, plotting progress chronologically across multiple practice runs.
Machine Learning Architecture
To maintain compatibility with standard CPU systems, the application employs a lightweight, highly optimized open-source model pipeline:
Function	Default Model / Library	Description
Speech-to-Text (ASR)	OpenAI Whisper (tiny or base)	Transcribes audio signals on CPU. Highly resilient to accents and ambient noise.
Document Search	all-MiniLM-L6-v2 + FAISS	Vectorizes resume chunks and matches semantic queries.
Question & Feedback Generation	google/flan-t5-small	Seq2Seq transformer that processes context prompts to draft feedback and custom interview questions.
Emotion Analysis	distilbert-base-uncased-emotion	Maps transcript structures to underlying human emotional tones.
System Requirements
1. System Dependencies
The Whisper transcription engine requires FFmpeg, a cross-platform multimedia framework, to handle audio format decoding.
Linux (Debian/Ubuntu):
code
Bash
sudo apt update && sudo apt install ffmpeg
macOS (via Homebrew):
code
Bash
brew install ffmpeg
Windows:
Download binary packages from the FFmpeg official site.
Extract the folder and append the bin directory path to your system's PATH environment variables.
2. Python Dependencies
Install the package requirements specified in your requirements.txt:
code
Bash
pip install -r requirements.txt
Installation and Execution
Clone or Save the Files:
Ensure ai_studio_code.py, requirements.txt, and packages.txt are in the same directory.
Run the Streamlit Application:
Navigate to your project directory and execute the following command:
code
Bash
streamlit run ai_studio_code.py
Access the Interface:
Open the address provided in your terminal (usually http://localhost:8501) in your preferred web browser.
Project Structure
code
Text
├── packages.txt          # System requirements for deployment platforms (e.g., Streamlit Community Cloud)
├── requirements.txt      # Python dependencies
└── ai_studio_code.py     # Main Streamlit application and evaluation engine
