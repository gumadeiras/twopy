"""AST scanning and static-use attribution for function inventory reports."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from function_inventory_support.enrichment import (
    _attach_git_metrics,
)
from function_inventory_support.model import (
    ApiSurface,
    FunctionKey,
    FunctionKind,
    FunctionMetric,
)


def build_inventory(
    *,
    root: Path,
    source_dirs: tuple[Path, ...],
    test_dir: Path,
) -> dict[FunctionKey, FunctionMetric]:
    """Build source function metrics and source/test attribution.

    Inputs: repository root, source directories, and test directory.
    Outputs: metrics keyed by module-qualified identity for audit reporting.
    """
    source_files = tuple(_python_files(source_dirs))
    test_files = tuple(_python_files((test_dir,)))
    exported_names = _collect_exported_names(root, source_files)
    metrics = _collect_functions(root, source_files, exported_names)
    _collect_uses(root, source_files + test_files, test_dir, metrics)
    _attach_git_metrics(root, metrics)
    return metrics


def _collect_functions(
    root: Path,
    files: tuple[Path, ...],
    exported_names: dict[str, set[str]],
) -> dict[FunctionKey, FunctionMetric]:
    metrics: dict[FunctionKey, FunctionMetric] = {}
    for path in files:
        source = path.read_text(encoding="utf-8")
        module = _module_name(root, path)
        tree = ast.parse(source, filename=str(path))
        finder = _FunctionFinder(
            root=root,
            path=path,
            module=module,
            source=source,
            exported_names=exported_names,
        )
        finder.visit(tree)
        metrics.update(finder.metrics)
    return metrics


def _collect_uses(
    root: Path,
    files: tuple[Path, ...],
    test_dir: Path,
    metrics: dict[FunctionKey, FunctionMetric],
) -> None:
    resolver = _FunctionResolver(metrics)
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module = _module_name(root, path)
        collector = _UseCollector(
            root=root,
            path=path,
            module=module,
            test_dir=test_dir,
            metrics=metrics,
            resolver=resolver,
        )
        collector.visit(tree)


def _sort_metric(metric: FunctionMetric) -> tuple[str, str]:
    return (metric.path.as_posix(), metric.key.qualname)


def _python_files(directories: Iterable[Path]) -> Iterable[Path]:
    for directory in directories:
        if not directory.exists():
            continue
        yield from sorted(
            path for path in directory.rglob("*.py") if "__pycache__" not in path.parts
        )


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _collect_exported_names(root: Path, files: tuple[Path, ...]) -> dict[str, set[str]]:
    exported: dict[str, set[str]] = {}
    package_exports: set[str] = set()
    for path in files:
        source = path.read_text(encoding="utf-8")
        module = _module_name(root, path)
        names = _module_all_names(ast.parse(source, filename=str(path)))
        exported[module] = names
        if module in {"twopy", "twopy.api", "twopy.custom"}:
            package_exports.update(names)
    exported[""] = package_exports
    return exported


def _module_all_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        values = _string_sequence(node.value)
        names.update(values)
    return names


def _string_sequence(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.List | ast.Tuple):
        values: list[str] = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                values.append(element.value)
        return tuple(values)
    return ()


def _domain_for_module(module: str) -> str:
    if module.startswith("twopy.napari"):
        return "napari_gui"
    if module.startswith("twopy.analysis"):
        return "analysis"
    if module.startswith("twopy.conversion") or module in {
        "twopy.converted",
        "twopy.matlab",
        "twopy.hdf5_utils",
        "twopy.session",
        "twopy.filenames",
        "twopy.inspection",
    }:
        return "conversion_data"
    if module.startswith("twopy.database"):
        return "database"
    if module.startswith("twopy.custom"):
        return "custom_workflows"
    if module.startswith("twopy.parity"):
        return "parity"
    if module in {
        "twopy.photodiode",
        "twopy.photodiode_classification",
        "twopy.synchronization",
        "twopy.stimulus",
        "twopy.frame_ranges",
    }:
        return "timing_sync"
    if module in {
        "twopy.roi",
        "twopy.roi_extraction",
        "twopy.roi_mask_cleanup",
        "twopy.response_roi_extraction",
        "twopy.spatial",
    }:
        return "roi"
    if module in {
        "twopy.api",
        "twopy.config",
        "twopy.pixel_calibration",
        "twopy.pixel_calibration_profiles",
        "twopy.typing_guards",
    }:
        return "metadata_api"
    return "core"


def _api_surface(
    *,
    module: str,
    qualname: str,
    public: bool,
    exported_names: dict[str, set[str]],
) -> ApiSurface:
    if not public:
        return "private"
    parts = qualname.split(".")
    owner = parts[0]
    function_name = parts[-1]
    module_exports = exported_names.get(module, set())
    package_exports = exported_names.get("", set())
    if owner in module_exports or function_name in module_exports:
        return "exported_api"
    if owner in package_exports or function_name in package_exports:
        return "exported_api"
    if module in {"twopy.api", "twopy.custom"}:
        return "exported_api"
    return "public_internal"


def _function_name(qualname: str) -> str:
    return qualname.rsplit(".", maxsplit=1)[-1]


def _line_count(text: str | None) -> int:
    if text is None:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def _node_end_line(node: ast.AST) -> int:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    return int(end if end is not None else start)


def _decorator_name(node: ast.AST) -> str:
    chain = _attribute_chain(node)
    if chain:
        return ".".join(chain)
    if isinstance(node, ast.Call):
        chain = _attribute_chain(node.func)
        if chain:
            return ".".join(chain)
    return ""


def _function_kind(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent_scope: str | None,
) -> FunctionKind:
    if parent_scope != "class":
        return "nested_function" if parent_scope == "function" else "function"
    decorators = {_decorator_name(decorator) for decorator in node.decorator_list}
    names = {decorator.rsplit(".", maxsplit=1)[-1] for decorator in decorators}
    if "staticmethod" in names:
        return "staticmethod"
    if "classmethod" in names:
        return "classmethod"
    if "property" in names:
        return "property"
    return "method"


def _code_line_count(
    lines: list[str],
    start_line: int,
    end_line: int,
    docstring_span: tuple[int, int] | None,
) -> int:
    count = 0
    for line_number in range(start_line, end_line + 1):
        if docstring_span is not None:
            doc_start, doc_end = docstring_span
            if doc_start <= line_number <= doc_end:
                continue
        text = lines[line_number - 1].strip()
        if text and not text.startswith("#"):
            count += 1
    return count


class _ComplexityCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.decision_count = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_If(self, node: ast.If) -> None:
        self.decision_count += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.decision_count += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.decision_count += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.decision_count += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.decision_count += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.decision_count += len(node.handlers)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.decision_count += max(len(node.values) - 1, 0)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.decision_count += len(node.cases)
        self.generic_visit(node)


def _cyclomatic_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    counter = _ComplexityCounter()
    for child in node.body:
        counter.visit(child)
    return counter.decision_count + 1


class _FunctionFinder(ast.NodeVisitor):
    def __init__(
        self,
        *,
        root: Path,
        path: Path,
        module: str,
        source: str,
        exported_names: dict[str, set[str]],
    ) -> None:
        self.root = root
        self.path = path
        self.module = module
        self.lines = source.splitlines()
        self.exported_names = exported_names
        self.scope_names: list[str] = []
        self.scope_kinds: list[str] = []
        self.metrics: dict[FunctionKey, FunctionMetric] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope_names.append(node.name)
        self.scope_kinds.append("class")
        self.generic_visit(node)
        self.scope_kinds.pop()
        self.scope_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent_scope = self.scope_kinds[-1] if self.scope_kinds else None
        qualname = ".".join((*self.scope_names, node.name))
        key = FunctionKey(module=self.module, qualname=qualname)
        docstring_node = _docstring_node(node)
        docstring_span = None
        docstring_span_lines = 0
        if docstring_node is not None:
            doc_start = int(docstring_node.lineno)
            doc_end = _node_end_line(docstring_node)
            docstring_span = (doc_start, doc_end)
            docstring_span_lines = doc_end - doc_start + 1

        end_line = _node_end_line(node)
        public = not any(part.startswith("_") for part in qualname.split("."))
        self.metrics[key] = FunctionMetric(
            key=key,
            path=self.path.relative_to(self.root),
            line=int(node.lineno),
            end_line=end_line,
            kind=_function_kind(node, parent_scope),
            public=public,
            total_lines=end_line - int(node.lineno) + 1,
            code_lines=_code_line_count(
                self.lines,
                int(node.lineno),
                end_line,
                docstring_span,
            ),
            docstring_lines=_line_count(ast.get_docstring(node, clean=False)),
            docstring_span_lines=docstring_span_lines,
            cyclomatic_complexity=_cyclomatic_complexity(node),
            domain=_domain_for_module(self.module),
            api_surface=_api_surface(
                module=self.module,
                qualname=qualname,
                public=public,
                exported_names=self.exported_names,
            ),
        )

        self.scope_names.append(node.name)
        self.scope_kinds.append("function")
        for child in node.body:
            self.visit(child)
        self.scope_kinds.pop()
        self.scope_names.pop()


def _docstring_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Expr | None:
    if not node.body:
        return None
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first
    return None


class _FunctionResolver:
    def __init__(self, metrics: dict[FunctionKey, FunctionMetric]) -> None:
        self.metrics = metrics
        self.by_fully_qualified = {key.display_name(): key for key in metrics}
        self.by_module_qualname = {(key.module, key.qualname): key for key in metrics}
        self.by_name: dict[str, list[FunctionKey]] = defaultdict(list)
        for key in metrics:
            self.by_name[_function_name(key.qualname)].append(key)

    def exact(self, name: str) -> FunctionKey | None:
        return self.by_fully_qualified.get(name)

    def module_member(self, module: str, name: str) -> FunctionKey | None:
        return self.by_module_qualname.get((module, name))

    def same_class_member(
        self,
        module: str,
        class_qualname: str | None,
        name: str,
    ) -> FunctionKey | None:
        if class_qualname is None:
            return None
        return self.by_module_qualname.get((module, f"{class_qualname}.{name}"))

    def ambiguous(self, name: str) -> tuple[FunctionKey, ...]:
        candidates = tuple(self.by_name.get(name, ()))
        return candidates if len(candidates) > 1 else ()

    def unique_name(self, name: str) -> FunctionKey | None:
        candidates = self.by_name.get(name, ())
        if len(candidates) == 1:
            return next(iter(candidates))
        return None


class _UseCollector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        root: Path,
        path: Path,
        module: str,
        test_dir: Path,
        metrics: dict[FunctionKey, FunctionMetric],
        resolver: _FunctionResolver,
    ) -> None:
        self.root = root
        self.path = path
        self.module = module
        self.test_dir = test_dir
        self.metrics = metrics
        self.resolver = resolver
        self.imports: dict[str, str] = {}
        self.scope_names: list[str] = []
        self.scope_kinds: list[str] = []
        self.current_test: str | None = None
        self.is_test_file = _is_relative_to(path, test_dir)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "twopy" or alias.name.startswith("twopy."):
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                self.imports[bound_name] = alias.name if alias.asname else bound_name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        if not self._tracks_imports_from(node.module):
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            self.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def _tracks_imports_from(self, module: str) -> bool:
        if module == "twopy" or module.startswith("twopy."):
            return True
        return self.is_test_file and module.startswith("tests.")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope_names.append(node.name)
        self.scope_kinds.append("class")
        self.generic_visit(node)
        self.scope_kinds.pop()
        self.scope_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_scope(node)

    def _visit_function_scope(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        previous_test = self.current_test
        qualname = ".".join((*self.scope_names, node.name))
        if self.is_test_file and node.name.startswith("test"):
            self.current_test = f"{self.module}.{qualname}"

        self.scope_names.append(node.name)
        self.scope_kinds.append("function")
        for child in node.body:
            self.visit(child)
        self.scope_kinds.pop()
        self.scope_names.pop()
        self.current_test = previous_test

    def visit_Call(self, node: ast.Call) -> None:
        key = self._resolve_node(node.func)
        if key is not None:
            metric = self.metrics[key]
            metric.direct_call_sites.add(self._site(node))
            self._add_test_use(key)
        else:
            self._add_ambiguous(node.func, node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            key = self._resolve_node(node)
            if key is not None:
                self._add_test_use(key)
            else:
                self._add_ambiguous(node, node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            key = self._resolve_node(node)
            if key is not None:
                self._add_test_use(key)
            else:
                self._add_ambiguous(node, node)
        self.generic_visit(node)

    def _add_test_use(self, key: FunctionKey) -> None:
        metric = self.metrics[key]
        if self.current_test is not None:
            metric.direct_test_functions.add(self.current_test)
            metric.direct_test_modules.add(self.module)

    def _add_ambiguous(self, node: ast.AST, site_node: ast.AST) -> None:
        name = _final_symbol_name(node)
        if name is None:
            return
        for key in self.resolver.ambiguous(name):
            self.metrics[key].ambiguous_name_sites.add(self._site(site_node))

    def _resolve_node(self, node: ast.AST) -> FunctionKey | None:
        chain = _attribute_chain(node)
        if not chain:
            if isinstance(node, ast.Attribute):
                return self.resolver.unique_name(node.attr)
            return None

        first = chain[0]
        if first in self.imports:
            candidate = ".".join((self.imports[first], *chain[1:]))
            key = self.resolver.exact(candidate)
            if key is not None:
                return key
            key = self.resolver.unique_name(chain[-1])
            if key is not None:
                return key

        if first == "twopy":
            key = self.resolver.exact(".".join(chain))
            if key is not None:
                return key
            key = self.resolver.unique_name(chain[-1])
            if key is not None:
                return key

        if len(chain) == 1:
            imported = self.imports.get(first)
            if imported is not None:
                key = self.resolver.exact(imported)
                if key is not None:
                    return key
            return self.resolver.module_member(self.module, first)

        if first in {"self", "cls"} and len(chain) == 2:
            return self.resolver.same_class_member(
                self.module,
                self._current_class_qualname(),
                chain[1],
            )

        if len(chain) == 2:
            return self.resolver.module_member(self.module, ".".join(chain))

        if isinstance(node, ast.Attribute):
            return self.resolver.unique_name(chain[-1])

        return None

    def _current_class_qualname(self) -> str | None:
        class_parts: list[str] = []
        for name, kind in zip(self.scope_names, self.scope_kinds, strict=True):
            if kind == "class":
                class_parts.append(name)
        if not class_parts:
            return None
        return ".".join(class_parts)

    def _site(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", 1)
        return f"{self.path.relative_to(self.root).as_posix()}:{line}"


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        if parent:
            return (*parent, node.attr)
    return ()


def _final_symbol_name(node: ast.AST) -> str | None:
    chain = _attribute_chain(node)
    if chain:
        return chain[-1]
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _escape_markdown(text: str) -> str:
    return text.replace("|", "\\|")
