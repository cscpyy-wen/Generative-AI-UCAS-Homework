#!/usr/bin/env python3
"""Check a public export without echoing matched personal data or credentials.

Exact private names/hosts can be supplied in a JSON policy stored OUTSIDE the
project: {"literal_terms": {"personal_name": [...], "student_id": [...]},
"forbidden_sha256": [...]}. The policy is never copied into the export.

Tensor files use torch.load(weights_only=True). Older PyTorch requires the
explicit --trust-local-weights option, which is only appropriate for files you
exported yourself from trusted local models, never downloaded pickle files.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import ipaddress
import json
import os
import pickletools
import re
import sys
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from xml.etree import ElementTree


MAX_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_DEPTH = 5
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}
WEIGHT_EXTENSIONS = {".pt", ".pth"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff"}
FORBIDDEN_EXTENSIONS = {".pkl", ".pickle", ".pyc", ".log", ".pem", ".key"}
PRIVATE_CATEGORIES = {"personal_name", "student_id", "private_hostname",
                      "private_server", "private_email", "credential"}
PATTERNS = {
    "local_absolute_path": re.compile(r"/(?:home|Users|root)/[^\s\"'<>]+"),
    "student_id": re.compile(r"(?<![\w.])20\d{13}(?!\w)"),
    "private_ssh_key": re.compile(r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,255}|github_pat_[A-Za-z0-9_]{30,255})\b"),
    "service_token": re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,255}|hf_[A-Za-z0-9]{20,255}|xox[baprs]-[A-Za-z0-9-]{20,255})\b"),
    "cloud_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,45}\b"),
    "private_server": re.compile(r"\b(?:connect|ssh)(?:\.[a-z0-9-]+)+\.(?:seetacloud|autodl)\.(?:com|net)\b", re.I),
    "ssh_connection": re.compile(r"\bssh\s+-p\s+\d{2,5}\s+[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+"),
}
CREDENTIAL_ASSIGNMENT = re.compile(
    r"[\"']?\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|ssh_password|client_secret)"
    r"\b[\"']?\s*[:=]\s*[\"']([^\"'\n]{8,})[\"']", re.I)
UNQUOTED_CREDENTIAL = re.compile(
    r"(?m)^\s*(?:export\s+)?(?:password|passwd|secret|api[_-]?key|access[_-]?token|ssh_password|client_secret)"
    r"\s*[:=]\s*([^\s\"'#$][^\s#]{7,})\s*$", re.I)
BEARER = re.compile(r"\bBearer\s+([A-Za-z0-9._~+/-]{20,})", re.I)
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
FORBIDDEN_NAME = re.compile(
    r"(?:hw1[-_]zh|hw1_teacher|teacher[_-]?notebook|training_state|resume_state|"
    r"conversation|chat_history|delivery_audit|final_audit|package_audit|"
    r"requirements_review|first_principles_audit)", re.I)


def placeholder(value: str) -> bool:
    value = value.strip()
    return (not value or value.lower() in {"example", "placeholder", "redacted", "changeme",
            "your_api_key", "your_token", "your_password", "not-a-real-secret"}
            or value.startswith(("<", "${", "os.getenv", "os.environ"))
            or any(mark in value for mark in ("[A-Za-z", "[0-9", "(?:", "\\b")))


class Audit:
    def __init__(self, root: Path, policy: dict, trust_weights: bool,
                 exclude_generated: bool = False):
        self.root = root
        self.trust_weights = trust_weights
        self.exclude_generated = exclude_generated
        self.findings: set[tuple[str, str]] = set()
        self.files_checked = 0
        self.weights_checked: list[dict] = []
        self.torch_configured = False
        self.literals: list[tuple[str, str]] = []
        for category, terms in policy.get("literal_terms", {}).items():
            category = category if category in PRIVATE_CATEGORIES else "private_literal"
            for term in terms:
                if isinstance(term, str) and term:
                    self.literals.append((category, term))
        self.forbidden_hashes = set(policy.get("forbidden_sha256", []))

    def safe_path(self, name: str) -> str:
        for _, term in self.literals:
            name = re.sub(re.escape(term), "<redacted>", name, flags=re.I)
        for pattern in PATTERNS.values():
            name = pattern.sub("<redacted>", name)
        name = IPV4.sub("<redacted>", name)
        return name

    def add(self, name: str, category: str) -> None:
        self.findings.add((self.safe_path(name), category))

    def text(self, value: str, name: str) -> None:
        for category, term in self.literals:
            if term.casefold() in value.casefold():
                self.add(name, category)
        for category, pattern in PATTERNS.items():
            if pattern.search(value):
                self.add(name, category)
        for match in CREDENTIAL_ASSIGNMENT.finditer(value):
            if not placeholder(match.group(1)):
                self.add(name, "credential_assignment")
        for match in UNQUOTED_CREDENTIAL.finditer(value):
            if not placeholder(match.group(1)):
                self.add(name, "credential_assignment")
        for match in BEARER.finditer(value):
            if not placeholder(match.group(1)):
                self.add(name, "bearer_credential")
        for match in IPV4.finditer(value):
            try:
                address = ipaddress.ip_address(match.group())
                # Loopback/bind addresses are ordinary local development syntax.
                if not (address.is_loopback or address.is_unspecified):
                    self.add(name, "ip_address")
            except ValueError:
                pass

    def tree(self, value, name: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                self.text(str(key), name)
                if str(key).lower() in {"hostname", "host_name", "computername", "machine_name"}:
                    if isinstance(item, str) and item.strip() not in {"", "localhost", "anonymous"}:
                        self.add(name, "machine_identity_metadata")
                self.tree(item, name)
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(item, str) for item in value):
                self.text("".join(value), name)
            for item in value:
                self.tree(item, name)
        elif isinstance(value, str):
            self.text(value, name)
        elif isinstance(value, bytes):
            self.text(value.decode("utf-8", "replace"), name)

    def zip_members(self, data: bytes, name: str, depth: int) -> None:
        if depth >= MAX_DEPTH:
            self.add(name, "archive_depth_uninspected")
            return
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                total = 0
                for entry in archive.infolist():
                    if entry.is_dir():
                        continue
                    total += entry.file_size
                    child = name + "!" + entry.filename
                    self.text(entry.filename, child)
                    if entry.flag_bits & 1:
                        self.add(child, "encrypted_archive_uninspected")
                    elif entry.file_size > MAX_BYTES or total > MAX_ARCHIVE_BYTES:
                        self.add(child, "archive_size_uninspected")
                    else:
                        self.payload(archive.read(entry), child, depth + 1)
        except Exception:
            self.add(name, "archive_parse_failed")

    def pdf(self, data: bytes, name: str) -> None:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                self.add(name, "encrypted_document_uninspected")
                return
            self.tree(dict(reader.metadata or {}), name)
            for page in reader.pages:
                self.text(page.extract_text() or "", name)
            if reader.xmp_metadata:
                self.text(reader.xmp_metadata.stream.get_data().decode("utf-8", "replace"), name)
        except Exception:
            self.add(name, "pdf_parse_failed_or_dependency_missing")

    def docx(self, data: bytes, name: str) -> None:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                total = 0
                for entry in archive.infolist():
                    if entry.is_dir():
                        continue
                    total += entry.file_size
                    if entry.file_size > MAX_BYTES or total > MAX_ARCHIVE_BYTES:
                        self.add(name, "document_size_uninspected")
                        continue
                    item = archive.read(entry)
                    self.text(entry.filename, name)
                    if entry.filename.endswith((".xml", ".rels")):
                        self.text(item.decode("utf-8", "replace"), name)
                        if entry.filename.endswith(".xml"):
                            document = ElementTree.fromstring(item)
                            # Run boundaries must not hide a name or identifier.
                            self.text("".join(document.itertext()), name)
                    elif entry.filename.startswith("word/embeddings/"):
                        self.add(name, "embedded_document_requires_review")
                        self.payload(item, name + "!" + entry.filename, 1)
                    elif Path(entry.filename).suffix.lower() in IMAGE_EXTENSIONS:
                        self.image(item, name)
        except Exception:
            self.add(name, "docx_parse_failed")

    def array(self, data: bytes, name: str) -> None:
        try:
            import numpy as np
            value = np.load(io.BytesIO(data), allow_pickle=False)
            if value.dtype.kind in {"O", "V"}:
                self.add(name, "opaque_array_uninspected")
            elif value.dtype.kind in {"S", "U"}:
                self.tree(value.tolist(), name)
        except Exception:
            self.add(name, "array_parse_failed_or_object_pickle")

    def image(self, data: bytes, name: str) -> None:
        try:
            from PIL import Image
            image = Image.open(io.BytesIO(data))
            self.tree(image.info, name)
            exif = image.getexif()
            if exif.get(34853):
                self.add(name, "image_location_metadata")
            self.tree(dict(exif), name)
        except Exception:
            self.add(name, "image_parse_failed_or_dependency_missing")

    def weight(self, data: bytes, name: str) -> None:
        # Inspect pickle text without executing it, even when loading is disabled.
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                pickle_entries = [x for x in archive.infolist() if x.filename.endswith("data.pkl")]
                if len(pickle_entries) != 1:
                    self.add(name, "nonstandard_weight_container")
                    return
                for opcode, argument, _ in pickletools.genops(archive.read(pickle_entries[0])):
                    if isinstance(argument, str):
                        self.text(argument, name)
        except Exception:
            self.add(name, "weight_pickle_text_uninspected")
            return
        try:
            import torch
            if not self.torch_configured:
                torch.set_num_threads(min(4, os.cpu_count() or 1))
                self.torch_configured = True
            supports_safe = "weights_only" in inspect.signature(torch.load).parameters
            if supports_safe:
                state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            elif self.trust_weights:
                state = torch.load(io.BytesIO(data), map_location="cpu")
            else:
                self.add(name, "safe_weight_loader_unavailable")
                return
            if not isinstance(state, Mapping) or not state:
                self.add(name, "weight_is_not_tensor_state_dict")
                return
            if getattr(state, "__dict__", {}):
                self.add(name, "weight_object_has_extra_metadata")
            count = 0
            for key, tensor in state.items():
                if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
                    self.add(name, "weight_contains_non_tensor_metadata")
                    continue
                self.text(key, name)
                count += tensor.numel()
            self.weights_checked.append({"file": self.safe_path(name), "keys": len(state),
                                         "tensor_elements": count})
        except Exception:
            self.add(name, "weight_load_failed_or_dependency_missing")

    def payload(self, data: bytes, name: str, depth: int = 0) -> None:
        suffix = Path(name.split("!")[-1]).suffix.lower()
        self.text(name, name)
        if hashlib.sha256(data).hexdigest() in self.forbidden_hashes:
            self.add(name, "private_original_artifact")
        if FORBIDDEN_NAME.search(name) or suffix in FORBIDDEN_EXTENSIONS:
            self.add(name, "forbidden_public_artifact")
        if suffix in WEIGHT_EXTENSIONS:
            self.weight(data, name)
        elif suffix == ".pdf":
            self.pdf(data, name)
        elif suffix == ".docx":
            self.docx(data, name)
        elif suffix == ".npz":
            self.zip_members(data, name, depth)
        elif suffix == ".npy":
            self.array(data, name)
        elif suffix in IMAGE_EXTENSIONS:
            self.image(data, name)
        elif zipfile.is_zipfile(io.BytesIO(data)):
            self.add(name, "general_archive_forbidden")
            self.zip_members(data, name, depth)
        elif suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar"}:
            self.add(name, "general_archive_forbidden")
            try:
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
                    total = 0
                    for entry in archive:
                        total += entry.size
                        child = name + "!" + entry.name
                        if entry.issym() or entry.islnk():
                            self.add(child, "archive_link_forbidden")
                            self.text(entry.linkname, child)
                        elif entry.isfile() and entry.size <= MAX_BYTES and total <= MAX_ARCHIVE_BYTES and depth < MAX_DEPTH:
                            member = archive.extractfile(entry)
                            if member:
                                self.payload(member.read(), child, depth + 1)
                        elif entry.isfile():
                            self.add(child, "archive_member_uninspected")
            except Exception:
                self.add(name, "archive_format_uninspected")
        else:
            try:
                text = data.decode("utf-8")
                self.text(text, name)
                if suffix in {".json", ".ipynb"}:
                    self.tree(json.loads(text), name)
            except UnicodeDecodeError:
                self.add(name, "unknown_binary_uninspected")
            except json.JSONDecodeError:
                self.add(name, "json_parse_failed")

    def run(self) -> None:
        generated_dirs = set()
        if self.exclude_generated:
            for child in self.root.iterdir():
                if (child.is_dir() and not child.is_symlink()
                        and (child.name in {"data", "runs", "samples", "reproduced"}
                             or child.name.startswith("runs-"))):
                    generated_dirs.add(child.name)
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if relative.parts[0] in generated_dirs:
                continue
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            name = relative.as_posix()
            if path.is_symlink():
                self.add(name, "filesystem_link_forbidden")
                self.text(str(path.readlink()), name)
            elif path.is_file():
                self.files_checked += 1
                if path.stat().st_size > MAX_BYTES:
                    self.add(name, "oversized_file_uninspected")
                else:
                    self.payload(path.read_bytes(), name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--private-patterns", type=Path, help="Private JSON policy located outside root")
    parser.add_argument("--trust-local-weights", action="store_true",
                        help="Allow old PyTorch loading of your own trusted local tensor exports")
    parser.add_argument("--expect-checkpoints", type=int, default=0)
    parser.add_argument("--exclude-generated", action="store_true",
                        help="Skip only root data/, runs/, runs-*/, samples/, reproduced/ directories for local checks")
    parser.add_argument("--json", action="store_true", help="Write a redacted JSON result to stdout")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("Project directory does not exist")
    policy = {}
    if args.private_patterns:
        private_path = args.private_patterns.resolve()
        if private_path == root or root in private_path.parents:
            parser.error("The private policy must be stored outside the public project")
        try:
            policy = json.loads(private_path.read_text(encoding="utf-8"))
        except Exception:
            parser.error("Unable to read the private JSON policy")
    audit = Audit(root, policy, args.trust_local_weights, args.exclude_generated)
    audit.run()
    if args.expect_checkpoints and len(audit.weights_checked) != args.expect_checkpoints:
        audit.add("checkpoints/", "unexpected_checkpoint_count")
    result = {"passed": not audit.findings, "files_checked": audit.files_checked,
              "checkpoints_checked": audit.weights_checked,
              "findings": [{"file": name, "category": category}
                           for name, category in sorted(audit.findings)]}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("PASS" if result["passed"] else "FAIL")
        print(f"Files checked: {audit.files_checked}; tensor checkpoints checked: {len(audit.weights_checked)}")
        for item in result["findings"]:
            print(f'{item["file"]}: {item["category"]}')
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
