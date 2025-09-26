# api_server.py
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from indextts.infer_v2 import IndexTTS2

app = FastAPI(title="IndexTTS2 API")


def _resolve_devices() -> List[Optional[str]]:
    """Return a list of devices that should host TTS models."""

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count >= 2:
            return [f"cuda:{i}" for i in range(2)]
        return ["cuda:0"]
    return [None]


def _init_tts_instances() -> List[IndexTTS2]:
    devices = _resolve_devices()
    instances: List[IndexTTS2] = []

    for device in devices:
        instances.append(
            IndexTTS2(
                cfg_path="checkpoints/config.yaml",
                model_dir="checkpoints",
                use_fp16=False,
                device=device,
            )
        )

    if len(instances) > 1:
        print(
            ">> Multi-GPU inference enabled on devices:",
            ", ".join(device or "auto" for device in devices),
        )
    else:
        print(
            ">> Single model instance initialised on device:",
            devices[0] or "auto",
        )

    return instances


_tts_instances = _init_tts_instances()
_tts_queue = deque(_tts_instances)
_tts_lock = asyncio.Lock()
_tts_semaphore = asyncio.Semaphore(len(_tts_instances))


@asynccontextmanager
async def _acquire_tts() -> IndexTTS2:
    await _tts_semaphore.acquire()
    instance: IndexTTS2 | None = None
    try:
        async with _tts_lock:
            instance = _tts_queue.popleft()
        yield instance
    finally:
        if instance is not None:
            async with _tts_lock:
                _tts_queue.append(instance)
        _tts_semaphore.release()

def _save_upload(upload: UploadFile, target: Path) -> str:
    with target.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(target)

@app.post("/tts")
async def synthesize(
    text: str = Form(...),
    speaker_audio: UploadFile = File(..., description="说话人参考音频（wav）"),
    emo_audio: Optional[UploadFile] = File(None, description="情感参考音频（可选）"),
    emo_text: Optional[str] = Form(None, description="情感描述文本（可选）"),
    use_emo_text: bool = Form(False),
    emo_vector_json: Optional[str] = Form(None, description="JSON 数组情感向量"),
    emo_alpha: float = Form(1.0),
    use_random: bool = Form(False),
    interval_silence: int = Form(200),
    max_text_tokens_per_segment: int = Form(120),
    temperature: float = Form(0.8),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    repetition_penalty: float = Form(10.0),
    max_mel_tokens: int = Form(1500),
):
    emo_vector = json.loads(emo_vector_json) if emo_vector_json else None

    tmpdir = Path(tempfile.mkdtemp())
    try:
        prompt_path = tmpdir / "prompt.wav"
        _save_upload(speaker_audio, prompt_path)

        emo_path = None
        if emo_audio is not None:
            emo_path = tmpdir / "emo.wav"
            _save_upload(emo_audio, emo_path)

        output_path = tmpdir / "output.wav"

        async with _acquire_tts() as tts_instance:
            await asyncio.to_thread(
                tts_instance.infer,
                spk_audio_prompt=str(prompt_path),
                text=text,
                output_path=str(output_path),
                emo_audio_prompt=str(emo_path) if emo_path else None,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                emo_text=emo_text,
                use_random=use_random,
                interval_silence=interval_silence,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_mel_tokens=max_mel_tokens,
            )

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts.wav",
            background=BackgroundTask(shutil.rmtree, tmpdir, ignore_errors=True),
        )
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
