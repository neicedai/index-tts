# IndexTTS2 API 使用指南

本文档介绍如何启动 `api_server.py` 并通过 `/tts` 接口合成语音，重点讲解情感控制相关参数的使用方式，包括情感参考音频、情感文本以及情感向量（`emo_vector_json`）。

## 启动服务

`api_server.py` 使用 [FastAPI](https://fastapi.tiangolo.com/) 搭建。在项目根目录执行以下命令即可启动本地服务：

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

服务启动后，通过 `POST http://<host>:8000/tts` 即可请求语音合成。建议在生产环境中配合反向代理或进程守护工具（如 `supervisor`、`systemd`）部署。

## `/tts` 接口参数

接口必须以 `multipart/form-data` 形式提交，主要字段如下：

| 字段 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `text` | `str` | ✅ | 要合成的文本内容。|
| `speaker_audio` | `UploadFile` | ✅ | 说话人参考音频（wav）。|
| `emo_audio` | `UploadFile` | ❌ | 情感参考音频（wav）。提供后可借助原始音频引导情感。|
| `emo_text` | `str` | ❌ | 情感描述文本。开启 `use_emo_text` 时生效，用于从文本自动推断情感向量。|
| `use_emo_text` | `bool` | ❌ | 是否启用情感文本模式。开启后会忽略上传的 `emo_audio`，仅使用文本或向量。默认 `False`。|
| `emo_vector_json` | `str` | ❌ | JSON 数组形式的情感向量，用于直接指定各情感权重。详见下文。|
| `emo_alpha` | `float` | ❌ | 情感强度缩放系数，范围 `0.0 ~ 1.0`，默认 `1.0`。|
| `use_random` | `bool` | ❌ | 是否启用随机采样，增加情感多样性但可能降低音色还原度。默认 `False`。|
| `interval_silence` | `int` | ❌ | 句段间静音长度，单位毫秒。默认 `200`。|
| `max_text_tokens_per_segment` | `int` | ❌ | 文本切分阈值。默认 `120`。|
| `temperature`、`top_p`、`top_k`、`repetition_penalty`、`max_mel_tokens` | 数值 | ❌ | 生成参数，高级使用场景可调节。|

接口返回 `audio/wav` 文件，文件名为 `tts.wav`。

## 情感控制方式

`IndexTTS2` 提供三种互斥的情感引导手段：情感参考音频、情感文本、情感向量。`api_server.py` 中的推理逻辑会根据下列优先级启用对应模式：

1. **当 `use_emo_text=True` 或提供了 `emo_vector_json` 时**：会忽略上传的 `emo_audio`，仅基于文本或向量进行情感混合。
2. **仅提供 `emo_audio` 时**：将 `emo_audio` 作为情感参考；若未提供则自动回退为 `speaker_audio`。

下文详细说明向量与文本模式的使用要点。

### 使用情感向量（`emo_vector_json`）

- 该字段需要传入一个 JSON 编码的浮点数组（长度为 8），依次表示：
  `happy`（开心）、`angry`（愤怒）、`sad`（悲伤）、`afraid`（害怕）、`disgusted`（厌恶）、`melancholic`（忧郁）、`surprised`（惊讶）、`calm`（平静）。
- 模型内部会对向量做两步处理：
  1. **偏置缩放**：为了避免极端情感导致不自然的结果，不同维度会乘以预设偏置系数。调整后的权重依然保持向量比例关系。
  2. **强度归一化**：向量各元素之和如果超过 `0.8`，会整体按比例缩放到 `0.8`。因此无需手动限制，只需按期望比例设置即可。
- `emo_alpha` 会在推理前再次整体缩放向量，可理解为“最终情感强度”，例如 `emo_alpha=0.6` 会将全部权重乘以 `0.6`。
- `use_random=True` 时，模型会在多组候选情感原型中随机采样，再按照向量权重混合，可提升多样性。

#### 示例：使用情感向量

```bash
curl -X POST "http://localhost:8000/tts" \
  -F "text=哇塞！这个爆率也太高了！" \
  -F "speaker_audio=@examples/voice_10.wav" \
  -F "emo_vector_json=[0.0,0.0,0.0,0.0,0.0,0.0,0.45,0.0]" \
  -F "emo_alpha=1.0" \
  -F "use_random=false" \
  --output tts.wav
```

以上示例将重点提升“惊讶”情感分量。

### 使用情感文本（`use_emo_text`/`emo_text`）

- 当 `use_emo_text=True` 时，若未提供 `emo_text`，系统会直接使用 `text` 本身推断情感向量；若提供 `emo_text`，则使用该文本进行情感推断。
- 推断得到的情感向量同样会经过偏置与归一化处理，并受 `emo_alpha` 缩放。
- 文本情感模式通常建议将 `emo_alpha` 设置在 `0.4~0.7` 之间，以获得更自然的音色。

#### 示例：使用情感文本

```bash
curl -X POST "http://localhost:8000/tts" \
  -F "text=快躲起来！他要来了！" \
  -F "speaker_audio=@examples/voice_12.wav" \
  -F "use_emo_text=true" \
  -F "emo_text=你吓死我了！你是鬼吗？" \
  -F "emo_alpha=0.6" \
  --output tts.wav
```

### 使用情感参考音频（`emo_audio`）

- 若未提供向量或文本模式，上传的 `emo_audio` 会直接作为情感参考。
- 可以通过 `emo_alpha` 调节参考音频对结果的影响；取值越接近 `1.0`，情感越接近参考音频。
- 若未上传 `emo_audio`，系统会退回使用 `speaker_audio` 中的情感信息。

## 处理结果与清理

接口会在后台为每次请求创建临时目录保存上传与输出文件，并在响应完成后自动清理，无需额外操作。

---

如需进一步定制 API 行为，可参考 `api_server.py` 中对 `IndexTTS2.infer` 的调用参数，并结合 `indextts/infer_v2.py` 的实现细节进行调整。
