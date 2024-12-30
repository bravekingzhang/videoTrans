# AI Video Face Swap and Translation Tool

This tool allows users to replace faces in videos with AI-generated virtual faces and translate the audio to different languages with synchronized lip movements.

## Features

- Face detection and tracking in videos
- AI-generated virtual face replacement
- Speech recognition and translation
- Lip-sync with translated audio
- Video processing and composition

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd videoTrans
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
For macOS:
```bash
brew install ffmpeg
```

## Project Structure

```
videoTrans/
├── src/
│   ├── face_processing/     # Face detection and replacement
│   ├── audio_processing/    # Audio extraction and translation
│   ├── video_processing/    # Video manipulation
│   └── utils/              # Utility functions
├── models/                  # Pre-trained models
├── tests/                   # Test files
└── config/                  # Configuration files
```

## Usage

1. Start the application:
```bash
python src/main.py
```

2. Upload a video file through the web interface or use the CLI:
```bash
python src/cli.py --input video.mp4 --target_language en
```

## Configuration

Create a `.env` file in the root directory with the following variables:
```
MODEL_PATH=./models
TEMP_DIR=./temp
OUTPUT_DIR=./output
```

## Development

To run tests:
```bash
python -m pytest tests/
```

## Docker

```bash
# 在云服务器上
git clone <your-repo>
cd videoTrans
docker-compose up -d
```

## License

MIT License