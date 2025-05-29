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
import asyncio
import threading
import time
from collections import deque

# Page configuration
st.set_page_config(page_title="Hausa Speech Transcription", page_icon="🎙️", layout="wide")

# Add custom CSS for better UI
st.markdown("""
<style>
    .recording-indicator {
        color: #ff4444;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 10px 0;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and processor
@st.cache_resource
def load_model():
    with st.spinner("Loading the Hausa transcription model..."):
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                "therealbee/whisper-small-ha-bible-tts",
                ignore_mismatched_sizes=True
            )
            processor = WhisperProcessor.from_pretrained("therealbee/whisper-small-ha-bible-tts")
            return model, processor
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None

# Transcription function
def transcribe_audio_from_file(audio_path, model, processor):
    try:
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
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

# Simplified Audio Recorder
class SimpleAudioRecorder:
    def __init__(self):
        self.audio_buffer = deque(maxlen=1500)  # ~30 seconds at 48kHz
        self.is_recording = False
        self.sample_rate = 48000
        self.lock = threading.Lock()
        self.recording_start_time = None
        
    def audio_frame_callback(self, frame: av.AudioFrame):
        """Process incoming audio frames"""
        try:
            # Convert frame to numpy array
            audio_array = frame.to_ndarray()
            
            # Handle stereo to mono conversion
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
            elif len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            
            with self.lock:
                if self.is_recording:
                    # Add to buffer
                    self.audio_buffer.extend(audio_array.astype(np.float32))
                    
        except Exception as e:
            print(f"Audio callback error: {e}")
            
        return frame
    
    def start_recording(self):
        """Start recording"""
        with self.lock:
            self.audio_buffer.clear()
            self.is_recording = True
            self.recording_start_time = time.time()
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        with self.lock:
            self.is_recording = False
            
            if not self.audio_buffer:
                return None
                
            # Convert buffer to numpy array
            audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
            
            # Check duration
            duration = len(audio_data) / self.sample_rate
            if duration < 0.5:
                return None
                
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
            
            # Resample to 16kHz for Whisper
            if self.sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=self.sample_rate, 
                    target_sr=16000
                )
            
            return audio_data, duration
    
    def get_recording_duration(self):
        """Get current recording duration"""
        if self.recording_start_time and self.is_recording:
            return time.time() - self.recording_start_time
        return 0

def main():
    # Initialize session state
    if 'recorder' not in st.session_state:
        st.session_state.recorder = SimpleAudioRecorder()
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = 'idle'  # idle, recording, recorded
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None
    if 'recording_duration' not in st.session_state:
        st.session_state.recording_duration = 0

    # Header
    st.title("🎙️ Hausa Speech Transcription")
    st.markdown("Convert Hausa speech to text using AI-powered transcription")

    # Load model
    model, processor = load_model()
    if model is None or processor is None:
        st.error("❌ Failed to load the transcription model. Please refresh and try again.")
        return

    # Main tabs
    tab1, tab2 = st.tabs(["🎤 Record Audio", "📁 Upload File"])

    with tab1:
        st.header("🎤 Record Hausa Speech")
        
        # Instructions
        with st.expander("📖 How to Record", expanded=True):
            st.markdown("""
            **Quick Steps:**
            1. **Click START** below to initialize audio connection
            2. **Allow microphone access** when prompted
            3. **Start Recording** and speak clearly in Hausa  
            4. **Stop Recording** when finished
            5. **Transcribe** your audio
            
            **💡 Tips:** Speak clearly, avoid background noise, keep recordings 2-30 seconds
            """)

        # WebRTC Configuration
        rtc_config = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        })

        # WebRTC Streamer
        webrtc_ctx = webrtc_streamer(
            key="hausa-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=4096,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "audio": {
                    "sampleRate": 48000,
                    "channelCount": 1,
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                },
                "video": False
            },
            audio_frame_callback=st.session_state.recorder.audio_frame_callback,
            async_processing=False,  # Try synchronous processing
        )

        # Connection Status
        if not webrtc_ctx.state.playing:
            st.info("📡 Click **START** above to connect your microphone")
            st.stop()
        else:
            st.success("✅ Connected - Ready to record!")

        # Recording Controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            start_disabled = st.session_state.recording_state == 'recording'
            if st.button("🔴 Start Recording", disabled=start_disabled, type="primary"):
                st.session_state.recorder.start_recording()
                st.session_state.recording_state = 'recording'
                st.rerun()

        with col2:
            stop_disabled = st.session_state.recording_state != 'recording'
            if st.button("⏹️ Stop Recording", disabled=stop_disabled):
                result = st.session_state.recorder.stop_recording()
                if result:
                    audio_data, duration = result
                    st.session_state.recording_duration = duration
                    
                    # Save audio to file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        sf.write(tmp_file.name, audio_data, 16000, format='WAV')
                        st.session_state.audio_file_path = tmp_file.name
                    
                    st.session_state.recording_state = 'recorded'
                    st.success(f"✅ Recorded {duration:.1f} seconds of audio!")
                else:
                    st.warning("⚠️ No audio captured. Please try again.")
                    st.session_state.recording_state = 'idle'
                st.rerun()

        with col3:
            # Status display
            if st.session_state.recording_state == 'recording':
                duration = st.session_state.recorder.get_recording_duration()
                st.markdown(f'<div class="recording-indicator">🔴 Recording... {duration:.1f}s</div>', 
                           unsafe_allow_html=True)
                time.sleep(0.5)
                st.rerun()
            elif st.session_state.recording_state == 'recorded':
                st.markdown(f"✅ **Recorded:** {st.session_state.recording_duration:.1f}s")
            else:
                st.markdown("⚪ **Ready to record**")

        # Audio Playback and Transcription
        if (st.session_state.recording_state == 'recorded' and 
            st.session_state.audio_file_path and 
            os.path.exists(st.session_state.audio_file_path)):
            
            st.divider()
            st.subheader("🔊 Your Recording")
            
            # Audio player
            st.audio(st.session_state.audio_file_path, format='audio/wav')
            
            # Action buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📝 Transcribe Audio", type="primary", key="transcribe_btn"):
                    with st.spinner("🔄 Transcribing your Hausa speech..."):
                        try:
                            transcription = transcribe_audio_from_file(
                                st.session_state.audio_file_path, model, processor
                            )
                            
                            st.success("✅ Transcription completed!")
                            st.subheader("📄 Hausa Transcription:")
                            
                            # Display result
                            st.text_area(
                                "Result:",
                                value=transcription,
                                height=120,
                                key="transcription_result"
                            )
                            
                            # Copy button
                            if st.button("📋 Copy Text"):
                                st.code(transcription, language=None)
                                
                        except Exception as e:
                            st.error(f"❌ Transcription failed: {str(e)}")
            
            with col2:
                if st.button("🗑️ Delete & Record Again", type="secondary"):
                    try:
                        if st.session_state.audio_file_path:
                            os.remove(st.session_state.audio_file_path)
                        st.session_state.audio_file_path = None
                        st.session_state.recording_state = 'idle'
                        st.success("Recording deleted. Ready for new recording!")
                        st.rerun()
                    except:
                        pass

    with tab2:
        st.header("📁 Upload Audio File")
        st.markdown("Upload a Hausa audio file for transcription")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg", "m4a", "flac", "aac"],
            help="Supported: WAV, MP3, OGG, M4A, FLAC, AAC (Max: 100MB)"
        )

        if uploaded_file is not None:
            # Save uploaded file
            file_extension = uploaded_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name

            # File info
            file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)
            st.success(f"✅ **{uploaded_file.name}** uploaded ({file_size:.1f} MB)")
            
            # Audio player
            st.audio(temp_path)

            # Transcribe button
            if st.button("📝 Transcribe File", type="primary", key="transcribe_upload"):
                with st.spinner("🔄 Processing uploaded audio..."):
                    try:
                        transcription = transcribe_audio_from_file(temp_path, model, processor)
                        
                        st.success("✅ Transcription completed!")
                        st.subheader("📄 Hausa Transcription:")
                        
                        # Display result
                        st.text_area(
                            "Result:",
                            value=transcription,
                            height=120,
                            key="upload_transcription_result"
                        )
                        
                        # Copy option
                        if st.button("📋 Copy Text", key="copy_upload"):
                            st.code(transcription, language=None)
                            
                    except Exception as e:
                        st.error(f"❌ Transcription failed: {str(e)}")
                    finally:
                        # Cleanup
                        try:
                            os.remove(temp_path)
                        except:
                            pass

    # Sidebar
    with st.sidebar:
        st.header("🎯 Quick Guide")
        
        st.markdown("""
        **🎤 Recording:**
        - Click START to connect
        - Allow microphone access
        - Record 2-30 seconds
        - Speak clearly in Hausa
        
        **📁 Upload:**
        - Choose audio file
        - Supported: WAV, MP3, etc.
        - Max size: 100MB
        """)
        
        st.header("💡 Tips")
        st.markdown("""
        **Better Results:**
        - Quiet environment
        - Clear pronunciation  
        - Normal speaking pace
        - Good microphone quality
        
        **Troubleshooting:**
        - Refresh if connection fails
        - Check microphone permissions
        - Try shorter recordings
        - Ensure Hausa language
        """)
        
        st.divider()
        st.info("**Model:** Whisper fine-tuned for Hausa\n**Source:** therealbee/whisper-small-ha-bible-tts")


if __name__ == "__main__":
    main()