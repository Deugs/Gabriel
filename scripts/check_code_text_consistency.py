#!/usr/bin/env python3
"""Verify docs/equation_code_mapping.md's rows against the actual codebase.

For every row marked "Implemented", checks that:
  1. The referenced code file exists.
  2. The referenced function/method/attribute is actually defined there
     (via static AST inspection -- no imports, no execution).

Rows marked "Pending" or "N/A (...)" are reported but not enforced: a
"Pending" row's file is expected to not exist yet (docs/rules.md Rule 1
tracks these honestly, it doesn't promise them done). "N/A" rows (e.g. the
superseded Hybrid SAC-DDQN's Eq. 3.25) are historical and intentionally
excluded from the current implementation.

Exit code is 0 if every "Implemented" row's mapping holds, 1 otherwise.
Referenced by docs/hooks.md's pre-commit hook and docs/rules.md Rule 1.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAPPING_FILE = REPO_ROOT / "docs" / "equation_code_mapping.md"

# Matches a table row like:
# | (3.5) | Total power | `power_model.py` | `PowerModel.compute_total_power()` | Implemented |
ROW_PATTERN = re.compile(
    r"^\|\s*\(([\d.]+)\)\s*\|\s*(.+?)\s*\|\s*`(.+?)`\s*\|\s*`(.+?)`\s*\|\s*(.+?)\s*\|\s*$"
)

ENFORCED_STATUS = "implemented"


@dataclass
class MappingRow:
    eq_num: str
    description: str
    code_file: str
    func_spec: str
    status: str


@dataclass
class CheckResult:
    row: MappingRow
    ok: bool
    detail: str


def parse_mapping_file(mapping_path: Path) -> list[MappingRow]:
    rows: list[MappingRow] = []
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        match = ROW_PATTERN.match(line.strip())
        if not match:
            continue
        eq_num, description, code_file, func_spec, status = match.groups()
        if eq_num == "Thesis Eq." or code_file == "Code File":
            continue  # header/format row
        rows.append(MappingRow(eq_num, description, code_file, func_spec, status))
    return rows


class _ModuleIndex:
    """Lazily-parsed AST for one source file, cached across rows."""

    def __init__(self, path: Path):
        self.path = path
        self.tree: Optional[ast.Module] = None
        self.parse_error: Optional[str] = None
        if path.exists():
            try:
                self.tree = ast.parse(
                    path.read_text(encoding="utf-8"), filename=str(path)
                )
            except SyntaxError as exc:  # pragma: no cover - defensive
                self.parse_error = f"failed to parse {path}: {exc}"

    def find_class(self, class_name: str) -> Optional[ast.ClassDef]:
        if self.tree is None:
            return None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def has_module_function(self, func_name: str) -> bool:
        if self.tree is None:
            return False
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
            for node in self.tree.body
        )

    @staticmethod
    def has_method(class_node: ast.ClassDef, method_name: str) -> bool:
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
            for node in class_node.body
        )

    @staticmethod
    def has_self_attribute(class_node: ast.ClassDef, attr_name: str) -> bool:
        """Best-effort check for `self.<attr_name> = ...` anywhere in the class body.

        Covers attributes set outside a plain method call, e.g. gymnasium
        spaces assigned in __init__ (CRANEnv.action_space is never a method).
        """
        target = f"self.{attr_name}"
        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr == attr_name:
                        if isinstance(t.value, ast.Name) and t.value.id == "self":
                            return True
            elif isinstance(node, ast.AnnAssign):
                if (
                    isinstance(node.target, ast.Attribute)
                    and node.target.attr == attr_name
                    and isinstance(node.target.value, ast.Name)
                    and node.target.value.id == "self"
                ):
                    return True
        _ = target  # kept for clarity, not used directly (ast.Attribute compare above)
        return False


def _split_func_spec(func_spec: str) -> tuple[Optional[str], str]:
    """Split 'ClassName.member()' into ('ClassName', 'member'); bare 'fn()' into (None, 'fn')."""
    spec = func_spec.strip()
    if spec.endswith("()"):
        spec = spec[:-2]
    if "." in spec:
        class_name, member = spec.rsplit(".", 1)
        return class_name, member
    return None, spec


def check_row(row: MappingRow) -> CheckResult:
    code_path = REPO_ROOT / row.code_file
    if not code_path.exists():
        return CheckResult(row, False, f"code file does not exist: {row.code_file}")

    module = _ModuleIndex(code_path)
    if module.parse_error:
        return CheckResult(row, False, module.parse_error)

    class_name, member_name = _split_func_spec(row.func_spec)

    if class_name is None:
        if module.has_module_function(member_name):
            return CheckResult(row, True, f"module function `{member_name}` found")
        return CheckResult(
            row, False, f"no module-level function `{member_name}` in {row.code_file}"
        )

    class_node = module.find_class(class_name)
    if class_node is None:
        return CheckResult(
            row, False, f"class `{class_name}` not found in {row.code_file}"
        )

    if module.has_method(class_node, member_name):
        return CheckResult(row, True, f"method `{class_name}.{member_name}` found")
    if module.has_self_attribute(class_node, member_name):
        return CheckResult(row, True, f"attribute `{class_name}.{member_name}` found")

    return CheckResult(
        row,
        False,
        f"`{class_name}` has no method or self-attribute named `{member_name}`",
    )


def run(mapping_path: Path) -> int:
    if not mapping_path.exists():
        print(f"ERROR: mapping file not found: {mapping_path}")
        return 1

    rows = parse_mapping_file(mapping_path)
    if not rows:
        print(f"ERROR: no table rows parsed from {mapping_path}")
        return 1

    failures: list[CheckResult] = []
    checked = 0
    skipped = 0

    for row in rows:
        if row.status.strip().lower() != ENFORCED_STATUS:
            skipped += 1
            print(
                f"  ~ Eq. ({row.eq_num}) [{row.status}] {row.description} -- not enforced"
            )
            continue

        checked += 1
        result = check_row(row)
        marker = "OK" if result.ok else "FAIL"
        print(f"  [{marker}] Eq. ({row.eq_num}) {row.description}: {result.detail}")
        if not result.ok:
            failures.append(result)

    print()
    print(
        f"Checked {checked} 'Implemented' row(s), {len(failures)} failure(s); "
        f"{skipped} row(s) skipped (Pending/N/A, not enforced)."
    )

    if failures:
        print()
        print(
            "Fix the code, or correct docs/equation_code_mapping.md if the row itself is stale:"
        )
        for f in failures:
            print(f"  - Eq. ({f.row.eq_num}): {f.detail}")
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=DEFAULT_MAPPING_FILE,
        help="Path to the equation-code mapping markdown file "
        f"(default: {DEFAULT_MAPPING_FILE.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()
    return run(args.mapping_file)


if __name__ == "__main__":
    sys.exit(main())
