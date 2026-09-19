"""Tests for scripts/check_code_text_consistency.py.

Covers: the real docs/equation_code_mapping.md passes end-to-end; a broken
mapping (nonexistent file, nonexistent class, nonexistent method/attribute)
is correctly caught; Pending/N/A rows are reported but never enforced.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_code_text_consistency.py"

_spec = importlib.util.spec_from_file_location(
    "check_code_text_consistency", SCRIPT_PATH
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)  # type: ignore[union-attr]


def _write_mapping(tmp_path: Path, rows: str) -> Path:
    content = (
        "# Equation-Code Mapping\n\n"
        "| Thesis Eq. | Description | Code File | Function | Status |\n"
        "|------------|-------------|-----------|----------|--------|\n"
        f"{rows}\n"
    )
    path = tmp_path / "mapping.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestParseMappingFile:
    def test_parses_real_mapping_file(self):
        rows = _module.parse_mapping_file(
            REPO_ROOT / "docs" / "equation_code_mapping.md"
        )
        assert len(rows) >= 22  # 22 Implemented + 3 skipped, at time of writing
        implemented = [r for r in rows if r.status.strip().lower() == "implemented"]
        assert len(implemented) >= 20

    def test_skips_header_row(self, tmp_path):
        path = _write_mapping(
            tmp_path,
            "| (9.1) | Dummy | `cran_env/power_model.py` "
            "| `PowerModel.compute_total_power()` | Implemented |",
        )
        rows = _module.parse_mapping_file(path)
        assert len(rows) == 1
        assert rows[0].eq_num == "9.1"


class TestCheckRow:
    def test_real_method_found(self):
        row = _module.MappingRow(
            "x",
            "d",
            "cran_env/power_model.py",
            "PowerModel.compute_total_power()",
            "Implemented",
        )
        result = _module.check_row(row)
        assert result.ok

    def test_real_module_function_found(self):
        row = _module.MappingRow(
            "x", "d", "training/train_hybrid.py", "train_hybrid_agent()", "Implemented"
        )
        result = _module.check_row(row)
        assert result.ok

    def test_self_attribute_found(self):
        row = _module.MappingRow(
            "x", "d", "cran_env/cran_env.py", "CRANEnv.action_space", "Implemented"
        )
        result = _module.check_row(row)
        assert result.ok

    def test_missing_file_fails(self):
        row = _module.MappingRow(
            "x", "d", "cran_env/does_not_exist.py", "Foo.bar()", "Implemented"
        )
        result = _module.check_row(row)
        assert not result.ok
        assert "does not exist" in result.detail

    def test_missing_class_fails(self):
        row = _module.MappingRow(
            "x",
            "d",
            "cran_env/power_model.py",
            "NoSuchClass.compute_total_power()",
            "Implemented",
        )
        result = _module.check_row(row)
        assert not result.ok
        assert "class" in result.detail.lower()

    def test_missing_method_fails(self):
        row = _module.MappingRow(
            "x",
            "d",
            "cran_env/power_model.py",
            "PowerModel.no_such_method()",
            "Implemented",
        )
        result = _module.check_row(row)
        assert not result.ok
        assert "no method or self-attribute" in result.detail

    def test_missing_module_function_fails(self):
        row = _module.MappingRow(
            "x", "d", "training/train_hybrid.py", "no_such_function()", "Implemented"
        )
        result = _module.check_row(row)
        assert not result.ok


class TestRunEndToEnd:
    def test_real_mapping_file_passes(self, capsys):
        exit_code = _module.run(REPO_ROOT / "docs" / "equation_code_mapping.md")
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "0 failure(s)" in captured.out

    def test_pending_row_with_nonexistent_file_does_not_fail_the_run(
        self, tmp_path, capsys
    ):
        path = _write_mapping(
            tmp_path,
            "| (9.1) | Not built yet | `cran_env/not_built_yet.py` | `Thing.do()` | Pending |",
        )
        exit_code = _module.run(path)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "not enforced" in captured.out

    def test_broken_implemented_row_fails_the_run(self, tmp_path, capsys):
        path = _write_mapping(
            tmp_path,
            "| (9.1) | Dummy | `cran_env/power_model.py` "
            "| `PowerModel.no_such_method()` | Implemented |",
        )
        exit_code = _module.run(path)
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "1 failure(s)" in captured.out

    def test_missing_mapping_file_fails_cleanly(self, tmp_path):
        exit_code = _module.run(tmp_path / "does_not_exist.md")
        assert exit_code == 1

    def test_mapping_with_no_rows_fails_cleanly(self, tmp_path):
        path = tmp_path / "empty.md"
        path.write_text("# Nothing here\n", encoding="utf-8")
        exit_code = _module.run(path)
        assert exit_code == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
