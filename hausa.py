import streamlit as st
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import numpy as np
import os
import tempfile
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_webrtc.webrtc import WebRtcStreamerContext
import av
import queue
import threading
import time

# Page configuration
st.set_page_config(page_title="Hausa Speech Transcription", page_icon="🎙️", layout="wide")

# Load model and processor
@st.cache_resource
def load_model():
    st.info("Loading the transcription model, please wait...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            "therealbee/whisper-small-ha-bible-tts",
            ignore_mismatched_sizes=True
        )
        processor = WhisperProcessor.from_pretrained("therealbee/whisper-small-ha-bible-tts")
        st.success("Model loaded successfully!")
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Transcription function from file path
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

# Improved Audio Recorder class
class AudioRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recorded_audio = []
        self.sample_rate = 48000  # WebRTC default
        self.lock = threading.Lock()
        self.min_recording_duration = 1.0  # Minimum 1 second
        self.max_recording_duration = 30.0  # Maximum 30 seconds
        self.recording_start_time = None

    def audio_frame_callback(self, frame: av.AudioFrame):
        """Callback to process audio frames from WebRTC"""
        try:
            with self.lock:
                if self.is_recording:
                    # Convert audio frame to numpy array
                    audio_array = frame.to_ndarray()
                    
                    # Handle different channel configurations
                    if len(audio_array.shape) > 1:
                        # Convert stereo to mono by averaging channels
                        audio_array = np.mean(audio_array, axis=0)
                    
                    # Store the audio data
                    self.recorded_audio.append(audio_array)
                    
                    # Check if we've exceeded max duration
                    if (self.recording_start_time and 
                        time.time() - self.recording_start_time > self.max_recording_duration):
                        self.stop_recording()
                        
        except Exception as e:
            st.error(f"Error in audio callback: {e}")
            
        return frame

    def start_recording(self):
        """Start recording audio"""
        with self.lock:
            self.recorded_audio = []
            self.is_recording = True
            self.recording_start_time = time.time()
            st.session_state.recording_status = "recording"

    def stop_recording(self):
        """Stop recording and return processed audio"""
        with self.lock:
            self.is_recording = False
            st.session_state.recording_status = "stopped"
            
            if not self.recorded_audio:
                return None
                
            try:
                # Concatenate all audio chunks
                full_audio = np.concatenate(self.recorded_audio, axis=0)
                
                # Check if we have enough audio
                duration = len(full_audio) / self.sample_rate
                if duration < self.min_recording_duration:
                    st.warning(f"Recording too short ({duration:.1f}s). Please record for at least {self.min_recording_duration}s")
                    return None
                
                # Convert to float32 and normalize
                audio_data = full_audio.astype(np.float32)
                
                # Normalize audio
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.8  # Leave some headroom
                
                # Resample to 16kHz for Whisper
                if self.sample_rate != 16000:
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=self.sample_rate, 
                        target_sr=16000
                    )
                
                return audio_data, duration
                
            except Exception as e:
                st.error(f"Error processing recorded audio: {e}")
                return None

    def get_recording_duration(self):
        """Get current recording duration"""
        if self.recording_start_time and self.is_recording:
            return time.time() - self.recording_start_time
        return 0

# Main Streamlit app
def main():
    # Initialize session state
    if 'recording_status' not in st.session_state:
        st.session_state.recording_status = "idle"
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    if 'recorded_audio_data' not in st.session_state:
        st.session_state.recorded_audio_data = None
    if 'recorded_audio_path' not in st.session_state:
        st.session_state.recorded_audio_path = None

    st.title("🎙️ Hausa Speech Transcription")
    st.markdown("**Record audio directly or upload a Hausa audio file to get a transcription.**")

    # Load model
    model, processor = load_model()
    
    if model is None or processor is None:
        st.error("Failed to load the model. Please check your internet connection and try again.")
        return

    # Create tabs with better styling
    tab1, tab2 = st.tabs(["🎤 **Record Audio**", "📁 **Upload File**"])

    with tab1:
        st.header("🎤 Record Hausa Audio")
        
        # WebRTC configuration
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Instructions
        with st.expander("📋 Recording Instructions", expanded=False):
            st.markdown("""
            1. **Click START** to initialize the audio connection
            2. **Allow microphone access** when prompted by your browser
            3. **Click "Start Recording"** and speak clearly in Hausa
            4. **Click "Stop Recording"** when finished (max 30 seconds)
            5. **Review your recording** and click "Transcribe"
            
            **Tips for better results:**
            - Speak clearly and at a normal pace
            - Record in a quiet environment
            - Keep recordings between 3-30 seconds
            - Make sure your microphone is working properly
            """)

        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="hausa-audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
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
            audio_frame_callback=st.session_state.audio_recorder.audio_frame_callback,
            async_processing=True,
        )

        # Connection status
        if webrtc_ctx.state.playing:
            st.success("🔗 **Connected** - Ready to record")
        else:
            st.info("📡 **Click START above** to connect and enable recording")
            st.stop()

        # Recording controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button(
                "🔴 Start Recording", 
                disabled=st.session_state.recording_status == "recording",
                type="primary" if st.session_state.recording_status != "recording" else "secondary"
            ):
                st.session_state.audio_recorder.start_recording()
                st.rerun()
        
        with col2:
            if st.button(
                "⏹️ Stop Recording", 
                disabled=st.session_state.recording_status != "recording",
                type="primary" if st.session_state.recording_status == "recording" else "secondary"
            ):
                result = st.session_state.audio_recorder.stop_recording()
                if result:
                    audio_data, duration = result
                    st.session_state.recorded_audio_data = audio_data
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        sf.write(tmp_file.name, audio_data, 16000, format='WAV', subtype='PCM_16')
                        st.session_state.recorded_audio_path = tmp_file.name
                    
                    st.success(f"✅ Recording saved! Duration: {duration:.1f} seconds")
                st.rerun()
        
        with col3:
            # Recording status and duration
            if st.session_state.recording_status == "recording":
                duration = st.session_state.audio_recorder.get_recording_duration()
                st.markdown(f"🔴 **Recording...** {duration:.1f}s / 30s")
                
                # Auto-refresh during recording
                time.sleep(0.1)
                st.rerun()
            elif st.session_state.recording_status == "stopped":
                st.markdown("⏹️ **Recording stopped**")
            else:
                st.markdown("⏸️ **Ready to record**")

        # Playback and transcription section
        if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
            st.divider()
            st.subheader("🔊 Your Recording")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.audio(st.session_state.recorded_audio_path)
            
            with col2:
                if st.button("🗑️ Delete Recording", type="secondary"):
                    try:
                        os.remove(st.session_state.recorded_audio_path)
                        st.session_state.recorded_audio_path = None
                        st.session_state.recorded_audio_data = None
                        st.success("Recording deleted!")
                        st.rerun()
                    except:
                        pass

            if st.button("📝 **Transcribe Audio**", key="transcribe_recorded", type="primary"):
                with st.spinner("🔄 Transcribing your Hausa speech..."):
                    try:
                        transcription = transcribe_audio_from_file(
                            st.session_state.recorded_audio_path, model, processor
                        )
                        st.success("✅ Transcription complete!")
                        
                        st.subheader("📋 Hausa Transcription:")
                        st.text_area(
                            "Transcribed Text:", 
                            value=transcription, 
                            height=100,
                            key="recorded_result",
                            help="Copy this text to use elsewhere"
                        )
                        
                        # Option to copy
                        if st.button("📋 Copy to Clipboard", key="copy_recorded"):
                            st.code(transcription)
                            
                    except Exception as e:
                        st.error(f"❌ Transcription error: {e}")
                    finally:
                        # Clean up
                        try:
                            os.remove(st.session_state.recorded_audio_path)
                            st.session_state.recorded_audio_path = None
                        except:
                            pass

    with tab2:
        st.header("📁 Upload Hausa Audio File")
        st.markdown("**Select an audio file from your device:**")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            help="Supported formats: WAV, MP3, OGG, M4A, FLAC (Max size: 200MB)"
        )

        if uploaded_file is not None:
            # Create temporary file
            file_extension = uploaded_file.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_audio_path = tmp_file.name

            # Display file info and audio player
            file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)  # MB
            st.success(f"✅ **File uploaded:** {uploaded_file.name} ({file_size:.1f} MB)")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.audio(temp_audio_path)
            
            with col2:
                st.info(f"**File:** {uploaded_file.name}\n**Size:** {file_size:.1f} MB")

            if st.button("📝 **Transcribe File**", key="transcribe_uploaded", type="primary"):
                with st.spinner("🔄 Transcribing uploaded file..."):
                    try:
                        transcription = transcribe_audio_from_file(temp_audio_path, model, processor)
                        st.success("✅ Transcription complete!")
                        
                        st.subheader("📋 Hausa Transcription:")
                        st.text_area(
                            "Transcribed Text:", 
                            value=transcription, 
                            height=100,
                            key="uploaded_result",
                            help="Copy this text to use elsewhere"
                        )
                        
                        # Option to copy
                        if st.button("📋 Copy to Clipboard", key="copy_uploaded"):
                            st.code(transcription)
                            
                    except Exception as e:
                        st.error(f"❌ Transcription error: {e}")
                    finally:
                        # Clean up
                        try:
                            os.remove(temp_audio_path)
                        except:
                            pass

    # Enhanced Sidebar
    with st.sidebar:
        st.header("📋 How to Use")
        
        with st.expander("🎤 Recording Guide", expanded=True):
            st.markdown("""
            **Step by Step:**
            1. Click **START** to connect
            2. Allow microphone access
            3. Click **Start Recording** 🔴
            4. Speak clearly in Hausa
            5. Click **Stop Recording** ⏹️
            6. Review and **Transcribe** 📝
            """)
        
        with st.expander("📁 Upload Guide"):
            st.markdown("""
            **Supported Files:**
            - WAV, MP3, OGG, M4A, FLAC
            - Max size: 200MB
            - Best quality: 16kHz, mono
            
            **Steps:**
            1. Choose your audio file
            2. Preview the audio
            3. Click **Transcribe** 📝
            """)
        
        st.header("💡 Tips for Better Results")
        st.markdown("""
        **Recording Quality:**
        - Speak clearly and naturally
        - Use a quiet environment
        - Keep 3-30 seconds length
        - Ensure good microphone

        **Audio Quality:**
        - Clear pronunciation
        - Minimal background noise
        - Standard Hausa dialect
        - Good audio levels
        """)

        st.header("⚠️ Troubleshooting")
        with st.expander("Common Issues"):
            st.markdown("""
            **Recording Problems:**
            - **No audio**: Allow microphone access
            - **Short recordings**: Check mic settings
            - **Poor quality**: Reduce background noise
            - **Connection issues**: Refresh the page
            
            **Transcription Issues:**
            - **Poor results**: Speak more clearly
            - **Wrong language**: Ensure Hausa speech  
            - **Errors**: Try shorter recordings
            """)

        st.divider()
        st.header("ℹ️ About")
        st.info("**Powered by:** Whisper model fine-tuned for Hausa language transcription")
        st.markdown("**Model:** `therealbee/whisper-small-ha-bible-tts`")

# Run the app
if __name__ == "__main__":
    main()