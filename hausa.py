import streamlit as st
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import numpy as np
import os
import tempfile
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import queue

# Page configuration
st.set_page_config(page_title="Hausa Speech Transcription", page_icon="🎙️")

# Load model and processor
@st.cache_resource
def load_model():
    st.info("Loading the transcription model, please wait...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "therealbee/whisper-small-ha-bible-tts",
        ignore_mismatched_sizes=True
    )
    processor = WhisperProcessor.from_pretrained("therealbee/whisper-small-ha-bible-tts")
    return model, processor

# Transcription function for file path
def transcribe_audio_from_file(audio_path, model, processor):
    # Load and resample audio
    audio, sampling_rate = librosa.load(audio_path, sr=None)
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

    # Prepare inputs
    inputs = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt", 
        language="ha"
    )

    # Generate transcription
    with torch.no_grad():
        outputs = model.generate(inputs.input_features, task="transcribe")

    # Decode transcription
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription

# Transcription function for audio bytes
def transcribe_audio_from_bytes(audio_bytes, model, processor):
    # Create temporary file from bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        temp_audio_path = tmp_file.name
    
    try:
        # Load and resample audio
        audio, sampling_rate = librosa.load(temp_audio_path, sr=None)
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

        # Prepare inputs
        inputs = processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            language="ha"
        )

        # Generate transcription
        with torch.no_grad():
            outputs = model.generate(inputs.input_features, task="transcribe")

        # Decode transcription
        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return transcription
    finally:
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# Audio recording with streamlit-webrtc
class AudioRecorder:
    def __init__(self):
        self.audio_frames = queue.Queue()
        self.is_recording = False
    
    def audio_frame_callback(self, frame):
        if self.is_recording:
            self.audio_frames.put(frame.to_ndarray())
        return frame
    
    def start_recording(self):
        self.is_recording = True
        self.audio_frames = queue.Queue()
    
    def stop_recording(self):
        self.is_recording = False
        
        # Collect all audio frames
        frames = []
        while not self.audio_frames.empty():
            frames.append(self.audio_frames.get())
        
        if frames:
            # Concatenate all frames
            audio_data = np.concatenate(frames, axis=0)
            return audio_data
        return None

# Streamlit app
def main():
    st.title("🎙️ Hausa Speech Transcription")
    st.write("Record audio directly or upload a Hausa language audio file for transcription.")

    # Load the model and processor
    model, processor = load_model()

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["🎤 Record Audio", "📁 Upload File"])

    with tab1:
        st.header("Record Audio Directly")
        st.write("Click 'START' to begin recording, then 'STOP' when finished.")
        
        # Initialize audio recorder
        if 'audio_recorder' not in st.session_state:
            st.session_state.audio_recorder = AudioRecorder()
        
        # WebRTC configuration for better connectivity
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Create webrtc streamer
        webrtc_ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"audio": True, "video": False},
            audio_frame_callback=st.session_state.audio_recorder.audio_frame_callback,
        )
        
        # Recording controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Start Recording", disabled=not webrtc_ctx.state.playing):
                st.session_state.audio_recorder.start_recording()
                st.success("Recording started! Speak now...")
        
        with col2:
            if st.button("⏹️ Stop Recording", disabled=not webrtc_ctx.state.playing):
                audio_data = st.session_state.audio_recorder.stop_recording()
                if audio_data is not None:
                    st.session_state.recorded_audio = audio_data
                    st.success("Recording stopped!")
                else:
                    st.warning("No audio data recorded.")
        
        # Display recorded audio and transcription option
        if hasattr(st.session_state, 'recorded_audio'):
            st.write("### Recorded Audio:")
            
            # Save recorded audio to temporary file for playback
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                # Ensure data is in correct dtype and shape for mono audio
                recorded = st.session_state.recorded_audio
                if recorded.ndim > 1:
                    recorded = recorded.mean(axis=1)  # Convert to mono if stereo

                recorded = recorded.astype(np.float32)  # Ensure float32 format
                sf.write(tmp_file.name, recorded, 16000, format='WAV')

                st.audio(tmp_file.name)
                temp_audio_path = tmp_file.name
            
            # Transcription button for recorded audio
            if st.button("Transcribe Recorded Audio", key="transcribe_recorded"):
                with st.spinner("Transcribing recorded audio..."):
                    try:
                        transcription = transcribe_audio_from_file(temp_audio_path, model, processor)
                        st.success("Transcription complete!")
                        
                        # Display transcription with copy functionality
                        st.subheader("Transcription Result:")
                        st.text_area("Transcribed Text:", value=transcription, height=100, key="recorded_transcription")
                        
                    except Exception as e:
                        st.error(f"An error occurred during transcription: {str(e)}")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)
        
        # Fallback information
        if not webrtc_ctx.state.playing:
            st.info("""
            **Alternative Recording Options:**
            1. Use your device's built-in voice recorder
            2. Record and save as WAV, MP3, or OGG format  
            3. Upload the file using the 'Upload File' tab above
            4. Make sure to allow microphone access in your browser
            """)

    with tab2:
        st.header("Upload Audio File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3', 'ogg', 'm4a', 'flac'],
            help="Upload a Hausa language audio file (WAV, MP3, OGG, M4A, or FLAC format)."
        )

        if uploaded_file is not None:
            # Get the file extension
            file_extension = uploaded_file.name.split('.')[-1]
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_audio_path = tmp_file.name

            # Display the audio player
            st.audio(temp_audio_path)
            
            # Show file info
            st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")

            # Transcription button
            if st.button("Transcribe Uploaded File", key="transcribe_uploaded"):
                with st.spinner("Transcribing audio..."):
                    try:
                        transcription = transcribe_audio_from_file(temp_audio_path, model, processor)
                        st.success("Transcription complete!")
                        
                        # Display transcription with copy functionality
                        st.subheader("Transcription Result:")
                        st.text_area("Transcribed Text:", value=transcription, height=100, key="uploaded_transcription")
                        
                    except FileNotFoundError:
                        st.error("Audio file not found. Please try uploading again.")
                    except ValueError as ve:
                        st.error(f"Value error: {ve}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)

    # Add some helpful information
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    ### How to use:
    1. **Record Audio**: Click on the Record Audio tab and use the microphone button
    2. **Upload File**: Click on Upload File tab and select an audio file
    3. **Transcribe**: Click the transcribe button after recording or uploading
    
    ### Supported formats:
    - WAV (recommended)
    - MP3
    - OGG
    - M4A
    - FLAC
    
    ### Tips for better transcription:
    - Speak clearly in Hausa
    - Minimize background noise
    - Keep recordings under 15
    - Ensure good audio quality
    """)
    
    st.sidebar.header("🔧 Technical Info")
    st.sidebar.info("This app uses the Whisper model fine-tuned for Hausa language transcription.")

# Run the app
if __name__ == "__main__":
    main()