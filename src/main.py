import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from video_processing.processor import VideoProcessor
from face_processing.face_swapper import FaceSwapper
from audio_processing.audio_translator import AudioTranslator

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Video Face Swap and Translation",
    description="An API for swapping faces in videos and translating audio",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
video_processor = VideoProcessor()
face_swapper = FaceSwapper()
audio_translator = AudioTranslator()

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Homepage with basic instructions
    """
    return """
    <html>
        <head>
            <title>AI Video Face Swap and Translation</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                code {
                    background-color: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <h1>AI Video Face Swap and Translation API</h1>
            <p>This API provides video face swapping and audio translation services.</p>

            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                <li><code>POST /process</code> - Process a video file</li>
            </ul>

            <h2>Example Usage:</h2>
            <pre>
curl -X POST "http://localhost:3000/process" \\
    -H "accept: application/json" \\
    -H "Content-Type: multipart/form-data" \\
    -F "video=@/path/to/your/video.mp4" \\
    -F "target_language=zh"
            </pre>
        </body>
    </html>
    """

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    target_language: str = "en",
):
    """
    Process a video file by swapping faces and translating audio

    Parameters:
    - video: The input video file
    - target_language: Target language code (e.g., 'en', 'zh', 'es')

    Returns:
    - Processed video file with swapped faces and translated audio
    """
    try:
        # Create temporary directories if they don't exist
        temp_dir = Path(os.getenv("TEMP_DIR", "./temp"))
        output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        temp_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Save uploaded video
        input_path = temp_dir / video.filename
        with open(input_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)

        # Process video
        frames = video_processor.extract_frames(input_path)
        processed_frames = face_swapper.process_frames(frames)

        # Process audio
        audio_path = video_processor.extract_audio(input_path)
        translated_audio = audio_translator.translate_and_synthesize(
            audio_path, target_language
        )

        # Combine processed frames and audio
        output_path = output_dir / f"processed_{video.filename}"
        video_processor.combine_video_audio(
            processed_frames, translated_audio, output_path
        )

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"processed_{video.filename}"
        )

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn

    # Get host and port from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))

    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)