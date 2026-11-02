"""Unit tests for the HTML documentation parser (src/ingestion/parser.py)."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.corpus_builder import build_raw_corpus
from src.ingestion.parser import HTMLDocParser


def test_html_parser_strips_boilerplate_and_preserves_code() -> None:
    sample_html = """<!DOCTYPE html>
    <html>
    <head>
        <meta name="module" content="torch.nn" />
        <meta name="symbol" content="Conv2d" />
        <title>torch.nn.Conv2d</title>
        <script>alert("ignore me");</script>
    </head>
    <body>
        <nav class="navbar">Top Navigation Bar</nav>
        <aside class="sidebar">Sidebar Links</aside>
        <main>
            <section id="overview" data-module="torch.nn" data-symbol="Conv2d">
                <h2>Conv2d Overview<a class="headerlink" href="#overview">¶</a></h2>
                <p>Applies a 2D convolution over an input signal.</p>
                <pre><code>conv = nn.Conv2d(3, 16, kernel_size=3)</code></pre>
            </section>
        </main>
        <footer class="footer">Footer Copyright</footer>
    </body>
    </html>
    """
    parser = HTMLDocParser(preserve_code_blocks=True)
    sections = parser.parse_html(sample_html, source_file="torch_nn_conv2d.html")

    assert len(sections) == 1
    sec = sections[0]
    assert sec.doc_id == "torch_nn_conv2d::overview"
    assert sec.module == "torch.nn"
    assert sec.symbol == "Conv2d"
    assert "Top Navigation Bar" not in sec.text
    assert "Sidebar Links" not in sec.text
    assert "Footer Copyright" not in sec.text
    assert "alert(" not in sec.text
    assert "¶" not in sec.text
    assert "```python" in sec.text
    assert "conv = nn.Conv2d(3, 16, kernel_size=3)" in sec.text


def test_corpus_builder_and_directory_parsing(tmp_path: Path) -> None:
    written = build_raw_corpus(tmp_path)
    assert len(written) == 16

    parser = HTMLDocParser()
    sections = parser.parse_directory(tmp_path)
    assert len(sections) == 29
    doc_ids = {s.doc_id for s in sections}
    assert "torch_nn_conv2d::signature_and_parameters" in doc_ids
    assert "torch_amp_mixed_precision::gradscaler_underflow_prevention" in doc_ids
