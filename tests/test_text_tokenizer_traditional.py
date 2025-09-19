import sys
import sys
import types
from pathlib import Path


def _ensure_stub_modules():
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.Tensor = type("Tensor", (), {})

        def _arange_stub(*args, **kwargs):
            return None

        torch.arange = _arange_stub
        sys.modules["torch"] = torch

    if "torchaudio" not in sys.modules:
        torchaudio = types.ModuleType("torchaudio")

        def _load_stub(*args, **kwargs):
            raise RuntimeError("torchaudio.load is not available in tests")

        torchaudio.load = _load_stub
        torchaudio.functional = types.SimpleNamespace(resample=lambda *args, **kwargs: None)
        sys.modules["torchaudio"] = torchaudio

    if "tn" not in sys.modules:
        sys.modules["tn"] = types.ModuleType("tn")

    if "tn.chinese" not in sys.modules:
        sys.modules["tn.chinese"] = types.ModuleType("tn.chinese")

    if "tn.english" not in sys.modules:
        sys.modules["tn.english"] = types.ModuleType("tn.english")

    if "tn.chinese.normalizer" not in sys.modules:
        chinese_normalizer = types.ModuleType("tn.chinese.normalizer")

        class _ZhNormalizer:
            def __init__(self, *args, **kwargs):
                pass

            def normalize(self, text):
                return text

        chinese_normalizer.Normalizer = _ZhNormalizer
        sys.modules["tn.chinese.normalizer"] = chinese_normalizer
        sys.modules["tn.chinese"].normalizer = chinese_normalizer

    if "tn.english.normalizer" not in sys.modules:
        english_normalizer = types.ModuleType("tn.english.normalizer")

        class _EnNormalizer:
            def __init__(self, *args, **kwargs):
                pass

            def normalize(self, text):
                return text

        english_normalizer.Normalizer = _EnNormalizer
        sys.modules["tn.english.normalizer"] = english_normalizer
        sys.modules["tn.english"].normalizer = english_normalizer

    if "sentencepiece" not in sys.modules:
        sentencepiece_module = types.ModuleType("sentencepiece")

        class _SentencePieceProcessor:
            def __init__(self, model_file: str):
                with open(model_file, "r", encoding="utf-8") as fh:
                    tokens = [line.strip() for line in fh if line.strip()]
                self._unk_token = "<unk>"
                self._piece_to_id = {self._unk_token: 0}
                self._id_to_piece = {0: self._unk_token}
                for token in tokens:
                    if token not in self._piece_to_id:
                        idx = len(self._piece_to_id)
                        self._piece_to_id[token] = idx
                        self._id_to_piece[idx] = token

            def GetPieceSize(self):
                return len(self._piece_to_id)

            def unk_id(self):
                return self._piece_to_id[self._unk_token]

            def _map_pieces(self, text: str):
                return [ch for ch in text if ch.strip()]

            def Encode(self, text, out_type=int, **kwargs):
                if isinstance(text, list):
                    return [self.Encode(t, out_type=out_type, **kwargs) for t in text]
                mapped = []
                for piece in self._map_pieces(text):
                    mapped.append(piece if piece in self._piece_to_id else self._unk_token)
                if out_type is str or out_type == str:
                    return mapped
                return [self._piece_to_id[p] for p in mapped]

            def Decode(self, ids, out_type=str, **kwargs):
                if isinstance(ids, list):
                    pieces = [self._id_to_piece.get(i, self._unk_token) for i in ids]
                else:
                    pieces = [self._id_to_piece.get(ids, self._unk_token)]
                if out_type is str or out_type == str:
                    return "".join(pieces)
                return pieces

            def IdToPiece(self, ids):
                if isinstance(ids, list):
                    return [self.IdToPiece(i) for i in ids]
                return self._id_to_piece.get(ids, self._unk_token)

            def PieceToId(self, token):
                return self._piece_to_id.get(token, self.unk_id())

        sentencepiece_module.SentencePieceProcessor = _SentencePieceProcessor
        sys.modules["sentencepiece"] = sentencepiece_module


_ensure_stub_modules()

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from indextts.utils.front import TextNormalizer, TextTokenizer


class DummyConverter:
    def __init__(self):
        self.calls = []

    def convert(self, text: str) -> str:
        self.calls.append(text)
        return text.replace("體", "体")


def test_traditional_phrase_tokenizes_without_unknown(tmp_path):
    vocab_path = tmp_path / "dummy_vocab.txt"
    vocab_path.write_text("\n".join(["繁", "体", "中", "文"]), encoding="utf-8")

    normalizer = TextNormalizer()
    tokenizer = TextTokenizer(str(vocab_path), normalizer=normalizer)

    dummy_converter = DummyConverter()
    normalizer.zh_converter = dummy_converter
    normalizer._converter_initialized = True

    pieces = tokenizer.encode("繁體中文", out_type=str)

    assert normalizer.zh_converter is dummy_converter
    assert dummy_converter.calls, "Traditional-to-simplified converter should be used"
    assert "<unk>" not in pieces
    assert "体" in "".join(pieces)
