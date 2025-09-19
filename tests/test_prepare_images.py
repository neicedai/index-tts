import ast
import textwrap
from pathlib import Path
from typing import List, Optional

import tempfile


def _load_prepare_images():
    module_path = Path(__file__).resolve().parents[1] / "ctv" / "ctv.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace = {
        "List": List,
        "Optional": Optional,
        "NamedTuple": __import__("typing").NamedTuple,
        "tempfile": tempfile,
        "Path": Path,
        "re": __import__("re"),
    }
    selected = []
    wanted = {
        "natural_key",
        "list_images_sorted",
        "PrepareImagesResult",
        "prepare_images",
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            selected.append(textwrap.dedent(ast.get_source_segment(source, node)))

    exec("\n\n".join(selected), namespace, namespace)
    return namespace["prepare_images"], namespace["PrepareImagesResult"]


prepare_images, PrepareImagesResult = _load_prepare_images()


def _build_sample_pdf() -> bytes:
    header = b"%PDF-1.4\n"
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300]"
            b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        ),
        (
            b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 72 200 Td (Hello) Tj ET\n"
            b"endstream\nendobj\n"
        ),
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /Name /F1 /BaseFont /Helvetica >>\nendobj\n",
    ]

    content = bytearray()
    content.extend(header)
    offsets = [0]
    for obj in objects:
        offsets.append(len(content))
        content.extend(obj)

    xref_pos = len(content)
    content.extend(b"xref\n")
    content.extend(f"0 {len(offsets)}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(b"trailer\n")
    content.extend(f"<< /Size {len(offsets)} /Root 1 0 R >>\n".encode("ascii"))
    content.extend(b"startxref\n")
    content.extend(f"{xref_pos}\n".encode("ascii"))
    content.extend(b"%%EOF\n")
    return bytes(content)


def test_prepare_images_converts_pdf_in_directory(tmp_path):
    created_files = []

    def fake_convert(pdf_path, out_dir, dpi=300):
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / f"{pdf_path.stem}_0001.png"
        image_path.write_bytes(b"fake-image")
        created_files.append(image_path)
        return [image_path]

    original_convert = prepare_images.__globals__.get("convert_pdf_to_images")
    prepare_images.__globals__["convert_pdf_to_images"] = fake_convert

    pdf_dir = tmp_path
    pdf_path = pdf_dir / "sample.pdf"
    pdf_path.write_bytes(_build_sample_pdf())

    result = prepare_images(pdf_dir)
    try:
        assert result.pdf_source == pdf_path
        assert result.images, "Converted images should not be empty"
        for img_path in result.images:
            assert img_path.exists()
        assert created_files == result.images
    finally:
        if result.temp_ctx is not None:
            result.temp_ctx.cleanup()
        if original_convert is not None:
            prepare_images.__globals__["convert_pdf_to_images"] = original_convert
        else:
            del prepare_images.__globals__["convert_pdf_to_images"]
