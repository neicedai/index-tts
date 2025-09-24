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

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `text` | `str` | ✅ | 无 | 要合成的文本内容。建议在上传前进行基本的标点和繁简处理，以避免模型错误断句。|
| `speaker_audio` | `UploadFile` | ✅ | 无 | 说话人参考音频（wav，16k/24k 皆可）。音频越干净、越接近目标音色，模型越易复刻；无需裁剪静音，服务端会自动处理。|
| `emo_audio` | `UploadFile` | ❌ | 无 | 情感参考音频（wav）。提供后可借助原始音频引导情感；若时长较长，会自动截取前 6 秒参与分析。|
| `emo_text` | `str` | ❌ | 未提供 (`None`) | 情感描述文本。开启 `use_emo_text` 时生效，用于从文本自动推断情感向量。可使用“高兴”“惊讶”等关键词，也可编写情绪丰富的句子。|
| `use_emo_text` | `bool` | ❌ | `False` | 是否启用情感文本模式。开启后会忽略上传的 `emo_audio`，仅使用文本或向量，并触发情感向量推断逻辑。|
| `emo_vector_json` | `str` | ❌ | 未提供 (`None`) | JSON 数组形式的情感向量，用于直接指定各情感权重。需提供长度为 8 的数组；为空时表示不启用此模式。详见下文。|
| `emo_alpha` | `float` | ❌ | `1.0` | 情感强度缩放系数，范围 `0.0 ~ 1.0`。值越小越接近中性语气，越大越突出选择的情感模式。|
| `use_random` | `bool` | ❌ | `False` | 是否启用随机采样，增加情感多样性但可能降低音色还原度。多轮生成时可开启以避免过度一致。|
| `interval_silence` | `int` | ❌ | `200` | 句段间静音长度，单位毫秒。服务会根据文本自动切句，可通过调大该值保留更多停顿。|
| `max_text_tokens_per_segment` | `int` | ❌ | `120` | 文本切分阈值。模型内部基于 BPE token 分段，阈值过大可能导致长句停顿不自然，过小会产生过多片段。|
| `temperature` | `float` | ❌ | `0.8` | 生成温度，越高结果越多样；越低则更稳定。建议范围 `0.5~1.0`。|
| `top_p` | `float` | ❌ | `0.8` | nucleus sampling 截断阈值。与 `temperature` 协同控制多样性；调高可尝试更夸张的情感表现。|
| `top_k` | `int` | ❌ | `30` | 每步采样时保留的候选数量。大部分场景无需调整，适当调小可增强稳定性。|
| `repetition_penalty` | `float` | ❌ | `10.0` | 重复惩罚系数，用于抑制生成的重复片段。默认值较大以保证音频稳定；若需更灵活表达，可尝试调低至 `2.0~5.0`，甚至接近 `1.0`。|
| `max_mel_tokens` | `int` | ❌ | `1500` | 允许生成的最大梅尔帧长度。控制生成时长上限，超出会被截断；适当增大可容纳较长文本。|

接口返回 `audio/wav` 文件，文件名为 `tts.wav`。

### 参数使用示例

下列 `curl` 片段演示了各字段常见的组合方式，可按需复制拼装：

- **最小化请求（必须字段）**：

  ```bash
  curl -X POST "http://localhost:8000/tts" \
    -F "text=欢迎使用 IndexTTS2" \
    -F "speaker_audio=@examples/voice_01.wav" \
    --output tts_basic.wav
  ```

- **引用情感参考音频 (`emo_audio`) 并放大情感权重 (`emo_alpha`)**：

  ```bash
  curl -X POST "http://localhost:8000/tts" \
    -F "text=今天的演出真是太棒了！" \
    -F "speaker_audio=@examples/voice_02.wav" \
    -F "emo_audio=@examples/voice_02_emotion.wav" \
    -F "emo_alpha=0.9" \
    --output tts_emo_audio.wav
  ```

- **使用情感文本模式 (`use_emo_text`/`emo_text`) 并开启随机采样 (`use_random`)**：

  ```bash
  curl -X POST "http://localhost:8000/tts" \
    -F "text=我们一起冲刺最后一公里！" \
    -F "speaker_audio=@examples/voice_03.wav" \
    -F "use_emo_text=true" \
    -F "emo_text=激动又兴奋地为大家加油" \
    -F "emo_alpha=0.7" \
    -F "use_random=true" \
    --output tts_emo_text.wav
  ```

- **指定情感向量 (`emo_vector_json`) 并拉长句段停顿 (`interval_silence`)**：

  ```bash
  curl -X POST "http://localhost:8000/tts" \
    -F "text=请在下一段落中跟随我的节奏" \
    -F "speaker_audio=@examples/voice_04.wav" \
    -F "emo_vector_json=[0.4,0,0,0,0,0,0.3,0.3]" \
    -F "emo_alpha=0.6" \
    -F "interval_silence=400" \
    --output tts_emo_vector.wav
  ```

- **控制文本切分 (`max_text_tokens_per_segment`) 与生成解码参数**：

  ```bash
  curl -X POST "http://localhost:8000/tts" \
    -F "text=以下内容将以播报风格缓慢读出，请保持安静倾听。" \
    -F "speaker_audio=@examples/voice_05.wav" \
    -F "max_text_tokens_per_segment=80" \
    -F "temperature=0.6" \
    -F "top_p=0.7" \
    -F "top_k=20" \
    -F "repetition_penalty=6.0" \
    -F "max_mel_tokens=2000" \
    --output tts_decoder_tuned.wav
  ```

上述示例可灵活组合，例如同时使用情感向量与随机采样，或在情感文本模式下调整 `interval_silence` 与 `max_mel_tokens` 等高级参数。

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

**调参建议**：

- `emo_vector_json` 中的权重可按“比例”来理解，例如设置 `[0.6,0.0,0.0,0.0,0.0,0.0,0.4,0.0]` 即强调开心与惊讶；无需担心总和超过 1，会自动归一化。
- 若想快速测试不同情感组合，可以将 `use_random` 设为 `true`，逐步增加 `emo_alpha` 并观察音色变化。

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
- 若既提供 `emo_text` 又提供 `emo_vector_json`，系统仍会优先使用向量模式；如需调试两者叠加，可在客户端自行按权重混合再传入向量。当前服务端不会自动混合两种模式。

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
- 建议参考音频与说话人音色保持一致（同一说话人、相似录制环境），以避免音色漂移；可以裁剪只保留最具代表性的 1~2 秒情感片段。

## 高级生成参数调优

在默认配置下，`IndexTTS2` 追求较高的音色还原和情感稳定性。如果需要针对特定场景调优，可参考以下思路：

1. **提升情感表现力**：适度调高 `temperature` 与 `top_p`（例如 `0.9` / `0.9`），同时把 `repetition_penalty` 调小到 `1.0`，让模型更敢于探索夸张表达。
2. **提高稳定性**：当文本较长或需要播报风格时，可将 `temperature` 降至 `0.5`、`top_p` 降至 `0.6`，并将 `use_random` 设为 `false`，确保音色一致。
3. **处理长文本**：若出现生成被提前截断，可根据语速调节 `max_mel_tokens`（例如增加到 `2200`）。该值越大计算量越高，建议按需调整。

这些参数会直接作用于底层解码器，配置不当可能导致音质下降。建议按上述顺序逐项尝试，并记录不同组合下的主观体验，以便确定适合自己业务的模板。

## 处理结果与清理

接口会在后台为每次请求创建临时目录保存上传与输出文件，并在响应完成后自动清理，无需额外操作。

---

如需进一步定制 API 行为，可参考 `api_server.py` 中对 `IndexTTS2.infer` 的调用参数，并结合 `indextts/infer_v2.py` 的实现细节进行调整。
