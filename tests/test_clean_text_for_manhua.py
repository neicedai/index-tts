from pathlib import Path
import ast
import textwrap

import pytest


def _load_cleaner():
    module_path = Path(__file__).resolve().parents[1] / "ctv" / "ctv.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "clean_text_for_manhua":
            func_source = ast.get_source_segment(source, node)
            break
    else:
        raise AssertionError("clean_text_for_manhua definition not found")

    namespace = {}
    exec(textwrap.dedent(func_source), {"re": __import__("re")}, namespace)
    return namespace["clean_text_for_manhua"]


clean_text_for_manhua = _load_cleaner()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("（*）Hello", "Hello"),
        ("(* Intro", "Intro"),
        ("（＊）Another", "Another"),
        ("（＊Trailing", "Trailing"),
    ],
)
def test_clean_text_for_manhua_removes_parenthetical_asterisks(raw, expected):
    assert clean_text_for_manhua(raw) == expected


def test_clean_text_for_manhua_mixed_content():
    raw = "（＊）First line\n(1)Second line\nNormal line"
    expected = "First line\nSecond line\nNormal line"
    assert clean_text_for_manhua(raw) == expected
