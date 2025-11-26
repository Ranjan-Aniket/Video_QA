"""
Modal Cloud Deployment for Gemini QA Pipeline

✅ LATEST SETTINGS INTEGRATED:
   - 2fps frame extraction (uniform temporal sampling)
   - Target: 30-35 questions per video (configured as 32)
   - All official taxonomy updates applied
   - Full dependency stack with GPU support

Deploy with:
    modal deploy modal_pipeline.py

Run single video:
    modal run modal_pipeline.py --video-url "https://youtube.com/watch?v=ABC"

Run batch:
    modal run modal_pipeline.py --batch-file videos.txt
"""

import modal
from pathlib import Path
import os

# Create Modal app
app = modal.App("gemini-qa-pipeline")

# GPU configuration (A10G is $0.30-0.40/hr, good balance of price/performance)
GPU_CONFIG = "A10G"  # Options: "T4" (cheapest), "A10G" (recommended), "A100" (fastest)

# Persistent volume for outputs (shared across all runs)
volume = modal.Volume.from_name(
    "gemini-qa-outputs",
    create_if_missing=True
)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install(
        "ffmpeg",
        "git",
        "wget",
        "curl"
    )
    .pip_install(
        # LLM APIs
        "anthropic>=0.15.0",
        "openai>=1.10.0",
        "google-generativeai>=0.3.0",
        # Core utilities
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "requests>=2.31.0",
        "httpx>=0.26.0",
        "tqdm>=4.66.0",
        "tenacity>=8.2.0",
        "backoff>=2.2.0",
        # Video processing
        "ffmpeg-python>=0.2.0",
        "yt-dlp",
        "opencv-python-headless>=4.9.0",
        "scenedetect[opencv]>=0.6.0",
        # Image processing
        "pillow>=10.0.0",
        "imagehash",
        "pytesseract>=0.3.10",
        "easyocr>=1.7.0",
        "ultralytics>=8.1.0",
        # Audio processing
        "openai-whisper>=20231117",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "noisereduce>=3.0.0",
        "pyannote.audio>=3.0.0",
        # ML/AI
        "numpy>=1.26.0,<2.0.0",
        "scipy>=1.11.0",
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "sentence-transformers>=2.3.0",
        "timm>=0.9.0",
        "open_clip_torch>=2.23.0",
        "git+https://github.com/openai/CLIP.git",
        # NLP
        "spacy>=3.7.0",
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.21.0",
        # Data processing
        "pandas>=2.2.0",
        "openpyxl>=3.1.0",
        "pyarrow>=14.0.0",
        # Pydantic & JSON
        "pydantic>=2.5.0,<3.0",
        "pydantic-settings>=2.1.0",
        "jsonschema>=4.21.0",
        # Web framework
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.0.0",
        # Utilities
        "rich>=13.7.0",
        "click>=8.1.0",
        "pyyaml>=6.0.0",
        "instructor>=0.4.0"
    )
    # Install spacy English model
    .run_commands("python -m spacy download en_core_web_sm")
)

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=7200,  # 2 hour timeout per video
    volumes={"/outputs": volume},
    secrets=[
        modal.Secret.from_name("openai-secret"),      # OPENAI_API_KEY
        modal.Secret.from_name("anthropic-secret"),   # ANTHROPIC_API_KEY
    ],
    memory=16384,  # 16GB RAM
)
def process_video(video_url: str, video_id: str = None):
    """
    Process a single video through the full Pass 1-2B pipeline

    Args:
        video_url: YouTube URL or direct video URL
        video_id: Optional custom ID

    Returns:
        Results dict with video_id, status, and output path
    """
    import sys
    sys.path.insert(0, "/root/Gemini_QA")

    from processing.smart_pipeline import SmartPipeline
    from datetime import datetime

    print(f"\n{'='*80}")
    print(f"PROCESSING VIDEO: {video_url}")
    print(f"{'='*80}\n")

    start_time = datetime.now()

    try:
        # Initialize pipeline with Modal volume as output directory
        pipeline = SmartPipeline(
            video_url=video_url,
            video_id=video_id,
            output_base_dir="/outputs",  # Modal persistent volume
            openai_api_key=os.environ["OPENAI_API_KEY"],
            claude_api_key=os.environ["ANTHROPIC_API_KEY"]
        )

        # Run full Pass 1-2B pipeline
        result = pipeline.run()

        # Commit volume to persist changes
        volume.commit()

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            "status": "success",
            "video_id": pipeline.video_id,
            "video_url": video_url,
            "questions_generated": len(result.get("questions", [])),
            "total_cost": result.get("pipeline_summary", {}).get("total_cost", 0),
            "processing_time_seconds": elapsed,
            "output_path": f"/outputs/video_{pipeline.video_id}"
        }

    except Exception as e:
        import traceback
        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            "status": "error",
            "video_url": video_url,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time_seconds": elapsed
        }


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=14400,  # 4 hour timeout for batch
    volumes={"/outputs": volume},
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ]
)
def batch_process_videos(video_urls: list[str], parallel: bool = True):
    """
    Process multiple videos in parallel or sequentially

    Args:
        video_urls: List of video URLs
        parallel: If True, process in parallel (faster but more expensive)

    Returns:
        Batch results summary
    """
    from datetime import datetime

    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING: {len(video_urls)} videos")
    print(f"Mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    print(f"{'='*80}\n")

    start_time = datetime.now()

    if parallel:
        # Process all videos in parallel using Modal's .map()
        results = list(process_video.map(video_urls))
    else:
        # Process sequentially
        results = []
        for i, url in enumerate(video_urls, 1):
            print(f"\n[{i}/{len(video_urls)}] Processing: {url}")
            result = process_video.remote(url)
            results.append(result)

    elapsed = (datetime.now() - start_time).total_seconds()

    # Calculate summary stats
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    total_questions = sum(r.get("questions_generated", 0) for r in successful)
    total_cost = sum(r.get("total_cost", 0) for r in successful)

    summary = {
        "status": "completed",
        "total_videos": len(video_urls),
        "successful": len(successful),
        "failed": len(failed),
        "total_questions": total_questions,
        "total_cost": total_cost,
        "processing_time_seconds": elapsed,
        "processing_time_formatted": f"{elapsed/60:.1f} minutes",
        "results": results
    }

    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE")
    print(f"  Successful: {len(successful)}/{len(video_urls)}")
    print(f"  Questions: {total_questions}")
    print(f"  Total Cost: ${total_cost:.2f}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"{'='*80}\n")

    return summary


@app.function(
    image=image,
    volumes={"/outputs": volume}
)
def list_outputs():
    """List all processed videos in the volume"""
    import os
    from pathlib import Path

    outputs = []
    output_dir = Path("/outputs")

    if not output_dir.exists():
        return {"videos": [], "count": 0}

    for video_dir in sorted(output_dir.iterdir()):
        if video_dir.is_dir() and video_dir.name.startswith("video_"):
            # Find questions file
            qa_files = list(video_dir.glob("*_pass3_qa_pairs.json"))
            if qa_files:
                import json
                with open(qa_files[0]) as f:
                    qa_data = json.load(f)

                outputs.append({
                    "video_id": video_dir.name.replace("video_", ""),
                    "path": str(video_dir),
                    "questions": qa_data.get("total_questions", 0),
                    "files": [f.name for f in video_dir.glob("*.json")]
                })

    return {
        "videos": outputs,
        "count": len(outputs)
    }


@app.function(
    image=image,
    volumes={"/outputs": volume}
)
def download_results(video_id: str):
    """
    Download results for a specific video

    Returns the contents of all JSON files as a dict
    """
    from pathlib import Path
    import json

    # Find video directory
    output_dir = Path("/outputs")
    video_dirs = list(output_dir.glob(f"video_*{video_id}*"))

    if not video_dirs:
        return {"error": f"No results found for video_id: {video_id}"}

    video_dir = video_dirs[0]
    results = {}

    # Read all JSON files
    for json_file in video_dir.glob("*.json"):
        with open(json_file) as f:
            results[json_file.name] = json.load(f)

    return {
        "video_id": video_id,
        "directory": str(video_dir),
        "results": results
    }


@app.local_entrypoint()
def main(
    video_url: str = None,
    batch_file: str = None,
    list: bool = False,
    download: str = None,
    parallel: bool = True
):
    """
    Main entry point for CLI usage

    Examples:
        # Process single video
        modal run modal_pipeline.py --video-url "https://youtube.com/watch?v=ABC"

        # Process batch (parallel)
        modal run modal_pipeline.py --batch-file videos.txt --parallel

        # Process batch (sequential - cheaper)
        modal run modal_pipeline.py --batch-file videos.txt --no-parallel

        # List all processed videos
        modal run modal_pipeline.py --list

        # Download results for a video
        modal run modal_pipeline.py --download "Copy of w-A-4ckmFJo"
    """

    if list:
        # List all outputs
        print("\n📂 Listing all processed videos...\n")
        result = list_outputs.remote()
        print(f"Found {result['count']} processed videos:\n")
        for video in result['videos']:
            print(f"  • {video['video_id']}: {video['questions']} questions")
        print()

    elif download:
        # Download specific video results
        print(f"\n⬇️  Downloading results for: {download}\n")
        result = download_results.remote(download)

        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            # Save to local files
            from pathlib import Path
            import json

            output_dir = Path("./downloaded_outputs") / download
            output_dir.mkdir(parents=True, exist_ok=True)

            for filename, data in result["results"].items():
                output_file = output_dir / filename
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  ✅ Saved: {output_file}")

            print(f"\n✅ Downloaded to: {output_dir}\n")

    elif batch_file:
        # Process batch file
        with open(batch_file) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        print(f"\n🚀 Processing {len(urls)} videos in {'PARALLEL' if parallel else 'SEQUENTIAL'} mode...\n")
        result = batch_process_videos.remote(urls, parallel=parallel)

        print("\n📊 Batch Results:")
        print(f"  Success: {result['successful']}/{result['total_videos']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Questions: {result['total_questions']}")
        print(f"  Cost: ${result['total_cost']:.2f}")
        print(f"  Time: {result['processing_time_formatted']}")

        # Show failed videos
        if result['failed'] > 0:
            print("\n❌ Failed videos:")
            for r in result['results']:
                if r['status'] == 'error':
                    print(f"  • {r['video_url']}: {r['error']}")

    elif video_url:
        # Process single video
        print(f"\n🚀 Processing single video...\n")
        result = process_video.remote(video_url)

        if result["status"] == "success":
            print(f"\n✅ Success!")
            print(f"  Video ID: {result['video_id']}")
            print(f"  Questions: {result['questions_generated']}")
            print(f"  Cost: ${result['total_cost']:.2f}")
            print(f"  Time: {result['processing_time_seconds']/60:.1f} minutes")
            print(f"  Output: {result['output_path']}\n")
        else:
            print(f"\n❌ Error processing video:")
            print(f"  {result['error']}\n")

    else:
        print("\n❌ Error: Must provide --video-url, --batch-file, --list, or --download\n")
        print("Examples:")
        print('  modal run modal_pipeline.py --video-url "https://youtube.com/..."')
        print("  modal run modal_pipeline.py --batch-file videos.txt")
        print("  modal run modal_pipeline.py --list")
        print('  modal run modal_pipeline.py --download "video_id"')
        print()


# Optional: Web API endpoint
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/outputs": volume},
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ]
)
@modal.asgi_app()
def web_api():
    """
    Deploy as web API:
        modal deploy modal_pipeline.py

    Then call:
        curl -X POST https://your-app.modal.run/process \
            -H "Content-Type: application/json" \
            -d '{"video_url": "https://youtube.com/..."}'
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    api = FastAPI(title="Gemini QA Pipeline API")

    class VideoRequest(BaseModel):
        video_url: str
        video_id: str = None

    @api.post("/process")
    async def process_endpoint(request: VideoRequest):
        """Process a single video"""
        result = process_video.remote(request.video_url, request.video_id)
        return result

    @api.get("/list")
    async def list_endpoint():
        """List all processed videos"""
        return list_outputs.remote()

    @api.get("/results/{video_id}")
    async def results_endpoint(video_id: str):
        """Get results for a specific video"""
        return download_results.remote(video_id)

    @api.get("/health")
    async def health():
        return {"status": "healthy"}

    return api
