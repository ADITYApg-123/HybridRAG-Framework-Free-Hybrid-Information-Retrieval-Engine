"""HTML and Markdown parser for technical documentation.

Strips boilerplate navigation/script tags, preserves code blocks as atomic fenced
blocks, and extracts hierarchical metadata (Module -> Symbol -> Section).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger("hybrid_rag.ingestion.parser")


@dataclass
class ParsedDocumentSection:
    """Represents a cleaned, structured section extracted from a documentation file."""

    doc_id: str
    module: str
    symbol: str
    section: str
    title: str
    text: str
    source_file: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HTMLDocParser:
    """Parses raw technical HTML documentation into structured sections with metadata."""

    BOILERPLATE_TAGS = ("nav", "footer", "script", "style", "noscript", "header", "aside")
    BOILERPLATE_CLASSES = (
        "navbar",
        "sidebar",
        "footer",
        "headerlink",
        "breadcrumb",
        "toc",
        "edit-on-github",
    )

    def __init__(self, preserve_code_blocks: bool = True) -> None:
        self.preserve_code_blocks = preserve_code_blocks

    def parse_file(self, file_path: str | Path) -> List[ParsedDocumentSection]:
        """Parse a single HTML or Markdown file into structured sections."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Documentation file not found: {path}")

        raw_content = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".html", ".htm"}:
            return self.parse_html(raw_content, source_file=path.name)
        return self._parse_plain_text(raw_content, source_file=path.name)

    def parse_directory(self, raw_dir: str | Path) -> List[ParsedDocumentSection]:
        """Parse all HTML/Markdown files in a directory sorted deterministically."""
        directory = Path(raw_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Raw documentation directory not found: {directory}")

        parsed_sections: List[ParsedDocumentSection] = []
        files = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in {".html", ".htm", ".md", ".txt"}
        )

        for file_path in files:
            sections = self.parse_file(file_path)
            parsed_sections.extend(sections)
            logger.debug("Parsed %d sections from %s", len(sections), file_path.name)

        logger.info(
            "Completed parsing %d files into %d structured document sections.",
            len(files),
            len(parsed_sections),
        )
        return parsed_sections

    def parse_html(self, html_content: str, source_file: str = "inline.html") -> List[ParsedDocumentSection]:
        """Extract clean text, code blocks, and hierarchical metadata from HTML."""
        soup = BeautifulSoup(html_content, "html.parser")

        module_name = self._extract_meta(soup, "module") or "torch"
        symbol_name = self._extract_meta(soup, "symbol") or Path(source_file).stem
        page_title = (
            soup.title.get_text(strip=True)
            if soup.title
            else f"{module_name}.{symbol_name}"
        )

        for tag_name in self.BOILERPLATE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        for tag in list(soup.find_all(True)):
            if not isinstance(tag, Tag) or tag.attrs is None:
                continue
            classes = tag.get("class") or []
            if any(cls in self.BOILERPLATE_CLASSES for cls in classes):
                tag.decompose()

        if self.preserve_code_blocks:
            for pre in soup.find_all("pre"):
                code_text = pre.get_text().strip()
                if code_text:
                    fenced = f"\n```python\n{code_text}\n```\n"
                    pre.replace_with(fenced)

        section_tags = soup.find_all("section")
        results: List[ParsedDocumentSection] = []

        if section_tags:
            for idx, sec in enumerate(section_tags):
                sec_id = sec.get("id") or f"section_{idx + 1}"
                sec_module = sec.get("data-module") or module_name
                sec_symbol = sec.get("data-symbol") or symbol_name
                heading_tag = sec.find(["h1", "h2", "h3", "h4"])
                sec_title = (
                    heading_tag.get_text(" ", strip=True)
                    if heading_tag
                    else f"{sec_module}.{sec_symbol} - {sec_id}"
                )

                clean_text = self._normalize_whitespace(sec.get_text("\n"))
                if not clean_text:
                    continue

                doc_id = f"{Path(source_file).stem}::{sec_id}"
                results.append(
                    ParsedDocumentSection(
                        doc_id=doc_id,
                        module=str(sec_module),
                        symbol=str(sec_symbol),
                        section=str(sec_id),
                        title=sec_title,
                        text=clean_text,
                        source_file=source_file,
                        metadata={
                            "page_title": page_title,
                            "hierarchy": f"{sec_module} -> {sec_symbol} -> {sec_id}",
                        },
                    )
                )
        else:
            body = soup.find("main") or soup.find("body") or soup
            clean_text = self._normalize_whitespace(body.get_text("\n"))
            if clean_text:
                doc_id = f"{Path(source_file).stem}::main"
                results.append(
                    ParsedDocumentSection(
                        doc_id=doc_id,
                        module=module_name,
                        symbol=symbol_name,
                        section="main",
                        title=page_title,
                        text=clean_text,
                        source_file=source_file,
                        metadata={
                            "page_title": page_title,
                            "hierarchy": f"{module_name} -> {symbol_name}",
                        },
                    )
                )

        return results

    def _parse_plain_text(self, content: str, source_file: str) -> List[ParsedDocumentSection]:
        """Fallback parser for plain text or markdown files."""
        clean_text = self._normalize_whitespace(content)
        stem = Path(source_file).stem
        return [
            ParsedDocumentSection(
                doc_id=f"{stem}::main",
                module="torch",
                symbol=stem,
                section="main",
                title=stem,
                text=clean_text,
                source_file=source_file,
                metadata={"hierarchy": f"torch -> {stem}"},
            )
        ]

    @staticmethod
    def _extract_meta(soup: BeautifulSoup, name: str) -> str | None:
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            return str(meta["content"]).strip()
        return None

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Clean excessive blank lines and trailing spaces while preserving fenced code blocks."""
        lines = [line.rstrip() for line in text.splitlines()]
        cleaned_lines: List[str] = []
        blank_streak = 0
        for line in lines:
            if not line.strip():
                blank_streak += 1
                if blank_streak <= 1:
                    cleaned_lines.append("")
            else:
                blank_streak = 0
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def save_jsonl(sections: List[ParsedDocumentSection], output_path: str | Path) -> None:
        """Save parsed sections to JSONL format."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for section in sections:
                f.write(json.dumps(section.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Saved %d parsed records to %s", len(sections), out)

