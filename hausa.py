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
import queue
import threading

# Page configuration
st.set_page_config(page_title="Hausa Speech Transcription", page_icon="🎙️")

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

# Audio Recorder class
class AudioRecorder:
    def __init__(self):
        self.audio_frames = []
        self.is_recording = False
        self.lock = threading.Lock()

    def audio_frame_callback(self, frame):
        with self.lock:
            if self.is_recording:
                # Convert frame to numpy array and append
                audio_array = frame.to_ndarray()
                self.audio_frames.append(audio_array)
        return frame

    def start_recording(self):
        with self.lock:
            self.audio_frames = []
            self.is_recording = True

    def stop_recording(self):
        with self.lock:
            self.is_recording = False
            if self.audio_frames:
                try:
                    # Concatenate all frames
                    audio_data = np.concatenate(self.audio_frames, axis=0)
                    
                    # Convert to mono if stereo
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # Ensure we have actual audio data
                    if len(audio_data) < 100:  # Less than 100 samples is likely empty
                        return None
                    
                    # Convert to float32 and normalize
                    audio_data = audio_data.astype(np.float32)
                    
                    # Remove DC offset
                    audio_data = audio_data - np.mean(audio_data)
                    
                    return audio_data
                except Exception as e:
                    st.error(f"Error processing audio frames: {e}")
                    return None
            return None

# Main Streamlit app
def main():
    st.title("🎙️ Hausa Speech Transcription")
    st.write("Record audio directly or upload a Hausa audio file to get a transcription.")

    # Load model
    model, processor = load_model()
    
    if model is None or processor is None:
        st.error("Failed to load the model. Please check your internet connection and try again.")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["🎤 Record Audio", "📁 Upload File"])

    with tab1:
        st.header("Record Hausa Audio")
        st.write("Click 'Start Recording' to begin, then 'Stop Recording' when finished.")

        # Initialize audio recorder in session state
        if 'audio_recorder' not in st.session_state:
            st.session_state.audio_recorder = AudioRecorder()
        if 'recorded_audio_path' not in st.session_state:
            st.session_state.recorded_audio_path = None

        # WebRTC configuration
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="hausa-audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration=rtc_config,
            media_stream_constraints={"audio": True, "video": False},  # Audio only
            audio_frame_callback=st.session_state.audio_recorder.audio_frame_callback,
        )

        # Recording controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Start Recording", disabled=not webrtc_ctx.state.playing):
                st.session_state.audio_recorder.start_recording()
                st.success("🔴 Recording started! Speak in Hausa now...")
        
        with col2:
            if st.button("⏹️ Stop Recording", disabled=not webrtc_ctx.state.playing):
                audio_data = st.session_state.audio_recorder.stop_recording()
                if audio_data is not None and len(audio_data) > 0:
                    try:
                        # Ensure audio data is in correct format
                        audio_data = np.array(audio_data, dtype=np.float32)
                        
                        # Normalize audio if needed
                        if np.max(np.abs(audio_data)) > 1.0:
                            audio_data = audio_data / np.max(np.abs(audio_data))
                        
                        # Save audio to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            sf.write(tmp_file.name, audio_data, 16000, format='WAV', subtype='PCM_16')
                            st.session_state.recorded_audio_path = tmp_file.name
                            st.success("✅ Recording saved!")
                    except Exception as e:
                        st.error(f"❌ Error saving audio: {str(e)}")
                        st.warning("⚠️ Try using the file upload option instead.")
                else:
                    st.warning("⚠️ No audio data captured. Please try again.")

        # Show connection status
        if webrtc_ctx.state.playing:
            st.success("🔗 Connected - Ready to record")
        else:
            st.info("📡 Click 'START' to connect and enable recording")

        # Playback and transcription
        if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
            st.subheader("🔊 Recorded Audio Playback")
            st.audio(st.session_state.recorded_audio_path)

            if st.button("📝 Transcribe Recorded Audio", key="transcribe_recorded"):
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
                            key="recorded_result"
                        )
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
        st.header("Upload Hausa Audio File")
        st.write("Select an audio file from your device:")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            help="Supported formats: WAV, MP3, OGG, M4A, FLAC"
        )

        if uploaded_file is not None:
            # Create temporary file
            file_extension = uploaded_file.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_audio_path = tmp_file.name

            # Display file info and audio player
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.audio(temp_audio_path)

            if st.button("📝 Transcribe Uploaded File", key="transcribe_uploaded"):
                with st.spinner("🔄 Transcribing uploaded file..."):
                    try:
                        transcription = transcribe_audio_from_file(temp_audio_path, model, processor)
                        st.success("✅ Transcription complete!")
                        st.subheader("📋 Hausa Transcription:")
                        st.text_area(
                            "Transcribed Text:", 
                            value=transcription, 
                            height=100,
                            key="uploaded_result"
                        )
                    except Exception as e:
                        st.error(f"❌ Transcription error: {e}")
                    finally:
                        # Clean up
                        try:
                            os.remove(temp_audio_path)
                        except:
                            pass

    # Sidebar with instructions
    st.sidebar.header("📋 How to Use")
    st.sidebar.markdown("""
    ### Recording Audio:
    1. Go to **Record Audio** tab
    2. Click **START** to connect
    3. Click **Start Recording** 🎤
    4. Speak in Hausa clearly
    5. Click **Stop Recording** ⏹️
    6. Click **Transcribe** 📝
    
    ### Uploading Files:
    1. Go to **Upload File** tab
    2. Choose your audio file
    3. Click **Transcribe** 📝
    
    ### Tips:
    - Speak clearly in Hausa
    - Keep recordings under 30 seconds
    - Use good quality audio
    - Allow microphone access in browser
    """)

    st.sidebar.header("ℹ️ Technical Info")
    st.sidebar.info("Powered by Whisper model fine-tuned for Hausa language transcription.")
    
    st.sidebar.header("⚠️ Troubleshooting")
    st.sidebar.markdown("""
    - **No audio captured**: Allow microphone access
    - **Can't record**: Try refreshing the page
    - **Poor transcription**: Speak more clearly
    - **Connection issues**: Check internet connection
    """)

# Run the app
if __name__ == "__main__":
    main()