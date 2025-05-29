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

# Page configuration
st.set_page_config(page_title="Hausa Speech Transcription", page_icon="🎙️")

# Load model and processor
@st.cache_resource
def load_model():
    model = WhisperForConditionalGeneration.from_pretrained(
        "therealbee/whisper-small-ha-bible-tts",
        ignore_mismatched_sizes=True
    )
    processor = WhisperProcessor.from_pretrained("therealbee/whisper-small-ha-bible-tts")
    return model, processor

# Transcription function from file path
def transcribe_audio_from_file(audio_path, model, processor):
    audio, sampling_rate = librosa.load(audio_path, sr=None)
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="ha")
    with torch.no_grad():
        outputs = model.generate(inputs.input_features, task="transcribe")
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription

# Audio Recorder class
class AudioRecorder:
    def __init__(self):
        self.audio_frames = queue.Queue()
        self.is_recording = False

    def audio_frame_callback(self, frame):
        if self.is_recording:
            self.audio_frames.put(frame.to_ndarray())
        return frame

    def start_recording(self):
        self.audio_frames = queue.Queue()
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        frames = []
        while not self.audio_frames.empty():
            frames.append(self.audio_frames.get())
        if frames:
            audio_data = np.concatenate(frames, axis=0).astype(np.float32)
            return audio_data
        return None

# Main Streamlit app
def main():
    st.title("🎙️ Hausa Speech Transcription")
    st.write("Record audio or upload a Hausa audio file to get a transcription.")

    model, processor = load_model()

    tab1, tab2 = st.tabs(["🎤 Record Audio", "📁 Upload File"])

    with tab1:
        st.header("Record Hausa Audio")

        if 'audio_recorder' not in st.session_state:
            st.session_state.audio_recorder = AudioRecorder()
        if 'recorded_audio_path' not in st.session_state:
            st.session_state.recorded_audio_path = None

        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_ctx = webrtc_streamer(
            key="recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration=rtc_config,
            media_stream_constraints={"audio": True, "video": False},
            audio_frame_callback=st.session_state.audio_recorder.audio_frame_callback,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Start Recording", disabled=not webrtc_ctx.state.playing):
                st.session_state.audio_recorder.start_recording()
                st.success("Recording started.")
        with col2:
            if st.button("⏹️ Stop Recording", disabled=not webrtc_ctx.state.playing):
                audio_data = st.session_state.audio_recorder.stop_recording()
                if audio_data is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        if audio_data.ndim > 1:
                            audio_data = audio_data.mean(axis=1)
                        sf.write(tmp_file.name, audio_data, 16000)
                        st.session_state.recorded_audio_path = tmp_file.name
                        st.success("Recording saved!")
                else:
                    st.warning("No audio data captured.")

        if st.session_state.recorded_audio_path:
            st.subheader("Playback")
            st.audio(st.session_state.recorded_audio_path)

            if st.button("Transcribe Recorded Audio"):
                with st.spinner("Transcribing..."):
                    try:
                        transcription = transcribe_audio_from_file(
                            st.session_state.recorded_audio_path, model, processor
                        )
                        st.success("Transcription complete!")
                        st.text_area("Hausa Transcription", transcription, height=100)
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        os.remove(st.session_state.recorded_audio_path)
                        st.session_state.recorded_audio_path = None

    with tab2:
        st.header("Upload Hausa Audio File")
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "ogg", "m4a", "flac"]
        )

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_audio_path = tmp_file.name

            st.audio(temp_audio_path)

            if st.button("Transcribe Uploaded File"):
                with st.spinner("Transcribing uploaded file..."):
                    try:
                        transcription = transcribe_audio_from_file(temp_audio_path, model, processor)
                        st.success("Transcription complete!")
                        st.text_area("Hausa Transcription", transcription, height=100)
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        os.remove(temp_audio_path)

    st.sidebar.header("📋 How to Use")
    st.sidebar.markdown("""
    - Use **Record Audio** tab to speak Hausa directly
    - Or use **Upload File** tab to upload `.wav`, `.mp3`, etc.
    - Click **Transcribe** to see the Hausa text
    """)

    st.sidebar.header("ℹ️ Tech Info")
    st.sidebar.info("Powered by a fine-tuned Whisper model for Hausa transcription.")

# Run
if __name__ == "__main__":
    main()
