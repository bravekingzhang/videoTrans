import whisper
import torch
from pathlib import Path
from typing import Union
import numpy as np
from transformers import pipeline
import os

class AudioTranslator:
    def __init__(self):
        """Initialize the audio translator with required models"""
        # Load Whisper model for speech recognition
        self.whisper_model = whisper.load_model("base")

        # Initialize translation pipeline
        self.translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-ROMANCE",
            device=0 if torch.cuda.is_available() else -1
        )

        # Initialize text-to-speech pipeline with a different model
        self.tts = pipeline(
            "text-to-speech",
            model="microsoft/speecht5_tts",
            device=0 if torch.cuda.is_available() else -1
        )

    def transcribe_audio(self, audio_path: Union[str, Path]) -> str:
        """
        Transcribe audio to text

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text
        """
        result = self.whisper_model.transcribe(str(audio_path))
        return result["text"]

    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text to target language

        Args:
            text: Input text
            target_language: Target language code

        Returns:
            Translated text
        """
        translation = self.translator(
            text,
            src_lang="en",
            tgt_lang=target_language
        )[0]["translation_text"]

        return translation

    def synthesize_speech(
        self,
        text: str,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Convert text to speech

        Args:
            text: Input text
            output_path: Path to save the audio file

        Returns:
            Path to the generated audio file
        """
        output_path = Path(output_path)

        # Generate speech
        speech = self.tts(text, forward_params={"speaker_embeddings": None})[0]["audio"]

        # Save audio file
        import soundfile as sf
        sf.write(str(output_path), speech, samplerate=16000)

        return output_path

    def translate_and_synthesize(
        self,
        audio_path: Union[str, Path],
        target_language: str
    ) -> Path:
        """
        Translate audio and generate speech in target language

        Args:
            audio_path: Path to the input audio file
            target_language: Target language code

        Returns:
            Path to the translated audio file
        """
        # Transcribe original audio
        text = self.transcribe_audio(audio_path)

        # Translate text
        translated_text = self.translate_text(text, target_language)

        # Generate speech in target language
        audio_path = Path(audio_path)
        output_path = audio_path.parent / f"translated_{audio_path.name}"

        return self.synthesize_speech(translated_text, output_path)