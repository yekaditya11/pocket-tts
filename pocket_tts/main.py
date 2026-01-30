import io
import logging
import os
import tempfile
import threading
from pathlib import Path
from queue import Queue

import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from typing_extensions import Annotated

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import PREDEFINED_VOICES, size_of_dict

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model: TTSModel | None = None
global_model_state = None

import torch
import time
from contextlib import asynccontextmanager

# Intel Optimization: Use 4 threads for faster single-request latency
torch.set_num_threads(4)
# Enable AMX/BF16 matrix multiplication on supported Intel CPUs (c7i/c8i)
torch.set_float32_matmul_precision("medium")

def warmup_model(voice_path):
    """Run a dummy generation to trigger torch.compile"""
    logger.info("Initializing model compilation (this may take up to 60 seconds)...")
    start_time = time.time()
    try:
        # Get state
        state = tts_model.get_state_for_audio_prompt(voice_path, truncate=True)
        # Dummy text
        dummy_text = "Warmup the engine."
        # Run generation
        # We need to drain the iterator
        chunks = tts_model.generate_audio_stream(state, dummy_text)
        for _ in chunks:
            pass
        end_time = time.time()
        logger.info(f"Model compilation complete in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load default model if not already loaded (though cli_app.serve loads it)
    # The current code loads it in 'serve' but app is created globaly.
    # To be safe and support running via uvicorn directy, we can ensure it's loaded here if None
    global tts_model
    if tts_model is None:
         # Fallback to defaults
         tts_model = TTSModel.load_model(DEFAULT_VARIANT)

    # Enable torch.compile
    try:
        logger.info("Enabling torch.compile for optimizations...")
        tts_model.generate_audio_stream = torch.compile(tts_model.generate_audio_stream, mode="reduce-overhead")
    except Exception as e:
        logger.warning(f"Could not enable torch.compile: {e}")

    # Warmup
    voice_path = Path("sample_audio/homesoul_sampleaudio.wav")
    if voice_path.exists():
        warmup_model(voice_path)
    else:
        logger.warning(f"Voice file not found at {voice_path}, skipping warmup.")

    yield
    # Cleanup
    pass

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0", lifespan=lifespan
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(static_path)


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


def write_to_queue(queue, text_to_generate, model_state):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    audio_chunks = tts_model.generate_audio_stream(
        model_state=model_state, text_to_generate=text_to_generate
    )
    stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, tts_model.config.mimi.sample_rate)


def generate_data_with_state(text_to_generate: str, model_state: dict):
    queue = Queue()

    # Run your function in a thread
    thread = threading.Thread(target=write_to_queue, args=(queue, text_to_generate, model_state))
    thread.start()

    # Yield data as it becomes available
    i = 0
    while True:
        data = queue.get()
        if data is None:
            break
        i += 1
        yield data

import base64
import json

def generate_ndjson_wrapper(text_to_generate: str, model_state: dict):
    """Generates NDJSON lines with base64 encoded audio."""
    idx = 0
    for chunk in generate_data_with_state(text_to_generate, model_state):
        encoded_audio = base64.b64encode(chunk).decode('utf-8')
        yield json.dumps({"audio": encoded_audio, "index": idx}) + "\n"
        idx += 1


DEFAULT_VOICE_PATH = "sample_audio/homesoul_sampleaudio.wav"

@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    response_format: str = Form("wav"),
):
    """
    Generate speech using the hardcoded voice file: sample_audio/homesoul_sampleaudio.wav
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    voice_path = Path(DEFAULT_VOICE_PATH)
    if not voice_path.exists():
         raise HTTPException(status_code=500, detail=f"Voice file not found at {DEFAULT_VOICE_PATH}")

    # Load state from the hardcoded file
    # We use truncate=True to handle long audio files gracefully
    model_state = tts_model.get_state_for_audio_prompt(voice_path, truncate=True)

    if response_format == "json":
        return StreamingResponse(
            generate_ndjson_wrapper(text, model_state),
            media_type="application/x-ndjson"
        )
    else:
        return StreamingResponse(
            generate_data_with_state(text, model_state),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated_speech.wav",
                "Transfer-Encoding": "chunked",
            },
        )


@cli_app.command()
def serve(
    voice: Annotated[
        str, typer.Option(help="Path to voice prompt audio file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    config: Annotated[
        str,
        typer.Option(
            help="Path to locally-saved model config .yaml file or model variant signature"
        ),
    ] = DEFAULT_VARIANT,
):
    """Start the FastAPI server."""

    global tts_model, global_model_state
    tts_model = TTSModel.load_model(config)

    # Pre-load the voice prompt
    global_model_state = tts_model.get_state_for_audio_prompt(voice)
    logger.info(f"The size of the model state is {size_of_dict(global_model_state) // 1e6} MB")

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    config: Annotated[
        str, typer.Option(help="Model signature or path to config .yaml file")
    ] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens per chunk.")
    ] = MAX_TOKEN_PER_CHUNK,
):
    """Generate speech using Kyutai Pocket TTS."""
    if "cuda" in device:
        # Cuda graphs capturing does not play nice with multithreading.
        os.environ["NO_CUDA_GRAPH"] = "1"

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(
            config, temperature, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.to(device)

        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        # Stream audio generation directly to file or stdout
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        # Only print the result message if not writing to stdout
        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


# ----------------------------------------------
# export audio to safetensors CLI implementation
# ----------------------------------------------


@cli_app.command()
def export_voice(
    audio_path: Annotated[
        str, typer.Argument(help="Audio file or directory to convert and export")
    ],
    export_path: Annotated[str, typer.Argument(help="Output file or directory")],
    truncate: Annotated[
        bool, typer.Option("-tr", "--truncate", help="Truncate long audio")
    ] = False,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    config: Annotated[str, typer.Option(help="Model config path or signature")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
):
    """Convert and save audio to .safetensors file"""
    import re

    def url(path):
        return path.startswith(("http:", "https:", "hf:"))

    def normalize_url(url):
        # utils.py expects urls to be xxx:// so normalize them
        return re.sub(r"^(http|https|hf)\:\/*(.+)$", r"\1://\2", url)

    def likely_file(path):
        return not url(path) and not likely_dir(path)

    def likely_dir(path):
        return not url(path) and (path.endswith(("/", "\\")) or path == ".")

    def convert_one(in_path, out_path, join_path):
        """helper convert function"""
        voice = in_path.stem
        if url(str(in_path)):
            in_path = normalize_url(str(in_path))
        if join_path:
            out_path = out_path / f"{voice}.safetensors"
        else:
            # ensure output file has correct extension
            out_path = out_path.with_suffix(".safetensors")
        try:
            tts_model.save_audio_prompt(in_path, out_path, truncate)
        except Exception as e:
            logger.error(f"❌ Unable to export voice '{in_path}': {e}")
            return False
        logger.info(f"✅ Successfully exported voice '{voice}' to '{out_path}'")
        return True

    if "cuda" in device:
        # Cuda graphs capturing does not play nice with multithreading.
        os.environ["NO_CUDA_GRAPH"] = "1"

    log_level = logging.ERROR if quiet else logging.INFO
    success_count = 0

    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(
            config, temperature, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.to(device)

        in_path = Path(audio_path)
        out_path = Path(export_path)
        if likely_dir(export_path):
            # make sure output dir exists
            out_path.mkdir(parents=True, exist_ok=True)

        if likely_dir(audio_path):  # batch convert whole directory
            if not in_path.is_dir():
                logger.error(f"Input dir '{audio_path}' does not exists")
                exit(1)
            if not likely_dir(export_path):
                # batch convert, output path must be directory, not file
                out_path = Path("./")
            for path in Path(in_path).iterdir():
                if path.is_file() and path.suffix.lower() in [
                    ".wav",
                    ".mp3",
                    ".flac",
                    ".ogg",
                    ".aiff",
                ]:
                    if convert_one(path, out_path, True):
                        success_count += 1
        else:  # convert single file
            if likely_file(audio_path) and not in_path.exists():
                logger.error(f"Input file '{in_path}'' does not exists")
                exit(1)
            if convert_one(in_path, out_path, likely_dir(export_path)):
                success_count += 1

        if success_count > 0:
            logger.info(f"🎉 Successfully exported {success_count} voices.")


if __name__ == "__main__":
    cli_app()
