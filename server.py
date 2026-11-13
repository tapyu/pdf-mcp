"""MCP server exposing PDF reading and annotation tools (comments, highlights, etc.)."""

import os
import tempfile
from pathlib import Path
from typing import Literal

import pymupdf
from mcp.server.mcpserver import MCPServer

mcp_server = MCPServer(
    name="pdf-mcp",
    instructions=(
        "Tools to read and annotate PDF files. Page numbers are 1-based. "
        "Annotation tools modify the PDF in place unless `output_path` is given."
    ),
)

MarkupType = Literal["highlight", "underline", "strikeout", "squiggly"]

COLOR_BY_NAME: dict[str, tuple[float, float, float]] = {
    "yellow": (1.0, 1.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.6, 1.0),
    "red": (1.0, 0.0, 0.0),
    "orange": (1.0, 0.6, 0.0),
    "pink": (1.0, 0.4, 0.8),
    "purple": (0.6, 0.2, 1.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def open_document(pdf_path: str) -> pymupdf.Document:
    resolved_path = Path(pdf_path).expanduser()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"PDF not found: {resolved_path}")
    return pymupdf.open(resolved_path)


def parse_color(color: str) -> tuple[float, float, float]:
    """Accept a color name (e.g. 'yellow') or a hex string (e.g. '#ffcc00')."""
    normalized_color = color.strip().lower()
    if normalized_color in COLOR_BY_NAME:
        return COLOR_BY_NAME[normalized_color]
    hex_digits = normalized_color.lstrip("#")
    if len(hex_digits) != 6:
        raise ValueError(
            f"Invalid color '{color}'. Use one of {sorted(COLOR_BY_NAME)} or a hex like '#ffcc00'."
        )
    return tuple(int(hex_digits[index : index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]


def resolve_page_indices(document: pymupdf.Document, page_number: int | None) -> list[int]:
    """Convert a 1-based page number (or None for all pages) into 0-based indices."""
    if page_number is None:
        return list(range(document.page_count))
    if not 1 <= page_number <= document.page_count:
        raise ValueError(f"page_number must be between 1 and {document.page_count}, got {page_number}.")
    return [page_number - 1]


def save_document(document: pymupdf.Document, pdf_path: str, output_path: str | None) -> str:
    """Save to `output_path` if given, otherwise overwrite the source PDF."""
    if output_path:
        destination_path = str(Path(output_path).expanduser())
        document.save(destination_path, garbage=3, deflate=True)
        document.close()
        return destination_path

    source_path = str(Path(pdf_path).expanduser())
    if document.can_save_incrementally():
        document.save(source_path, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
        document.close()
        return source_path

    # Fall back to writing a temp file and atomically replacing the original.
    file_descriptor, temporary_path = tempfile.mkstemp(suffix=".pdf", dir=os.path.dirname(source_path))
    os.close(file_descriptor)
    document.save(temporary_path, garbage=3, deflate=True)
    document.close()
    os.replace(temporary_path, source_path)
    return source_path


def describe_annotation(annotation: pymupdf.Annot, page_number: int) -> dict:
    info = annotation.info
    return {
        "page_number": page_number,
        "annotation_id": annotation.xref,
        "type": annotation.type[1],
        "content": info.get("content", ""),
        "author": info.get("title", ""),
        "modified": info.get("modDate", ""),
        "rect": [round(coordinate, 2) for coordinate in annotation.rect],
    }


def add_text_markup(
    pdf_path: str,
    text: str,
    markup_type: MarkupType,
    page_number: int | None,
    color: str,
    comment: str | None,
    author: str | None,
    is_first_match_only: bool,
    output_path: str | None,
) -> dict:
    document = open_document(pdf_path)
    rgb_color = parse_color(color)
    markup_method_by_type = {
        "highlight": pymupdf.Page.add_highlight_annot,
        "underline": pymupdf.Page.add_underline_annot,
        "strikeout": pymupdf.Page.add_strikeout_annot,
        "squiggly": pymupdf.Page.add_squiggly_annot,
    }
    add_markup = markup_method_by_type[markup_type]

    created_annotations: list[dict] = []
    for page_index in resolve_page_indices(document, page_number):
        page = document[page_index]
        for quad in page.search_for(text, quads=True):
            annotation = add_markup(page, quad)
            annotation.set_colors(stroke=rgb_color)
            if comment or author:
                annotation.set_info(content=comment or "", title=author or "")
            annotation.update()
            created_annotations.append(describe_annotation(annotation, page_index + 1))
            if is_first_match_only:
                break
        if is_first_match_only and created_annotations:
            break

    if not created_annotations:
        document.close()
        return {"status": "not_found", "message": f"Text '{text}' not found.", "annotations": []}

    saved_path = save_document(document, pdf_path, output_path)
    return {"status": "ok", "saved_to": saved_path, "annotations": created_annotations}


# ---------------------------------------------------------------------------
# Reading tools
# ---------------------------------------------------------------------------


@mcp_server.tool()
def get_pdf_info(pdf_path: str) -> dict:
    """Return metadata, page count, and page sizes of a PDF."""
    with open_document(pdf_path) as document:
        return {
            "page_count": document.page_count,
            "metadata": document.metadata,
            "is_encrypted": document.is_encrypted,
            "page_sizes": [
                {"page_number": page.number + 1, "width": page.rect.width, "height": page.rect.height}
                for page in document
            ],
        }


@mcp_server.tool()
def extract_text(pdf_path: str, page_number: int | None = None) -> list[dict]:
    """Extract plain text from one page (1-based) or from all pages if `page_number` is omitted."""
    with open_document(pdf_path) as document:
        return [
            {"page_number": page_index + 1, "text": document[page_index].get_text()}
            for page_index in resolve_page_indices(document, page_number)
        ]


@mcp_server.tool()
def search_text(pdf_path: str, query: str, page_number: int | None = None) -> list[dict]:
    """Find every occurrence of `query` and return its page number and bounding box."""
    with open_document(pdf_path) as document:
        matches: list[dict] = []
        for page_index in resolve_page_indices(document, page_number):
            for rect in document[page_index].search_for(query):
                matches.append(
                    {"page_number": page_index + 1, "rect": [round(coordinate, 2) for coordinate in rect]}
                )
        return matches


@mcp_server.tool()
def list_annotations(pdf_path: str, page_number: int | None = None) -> list[dict]:
    """List annotations (comments, highlights, ...) on one page or on all pages."""
    with open_document(pdf_path) as document:
        return [
            describe_annotation(annotation, page_index + 1)
            for page_index in resolve_page_indices(document, page_number)
            for annotation in document[page_index].annots()
        ]


# ---------------------------------------------------------------------------
# Annotation tools
# ---------------------------------------------------------------------------


@mcp_server.tool()
def highlight_text(
    pdf_path: str,
    text: str,
    page_number: int | None = None,
    color: str = "yellow",
    comment: str | None = None,
    author: str | None = None,
    is_first_match_only: bool = False,
    output_path: str | None = None,
) -> dict:
    """Highlight every occurrence of `text` (optionally on a single page) with an optional attached comment."""
    return add_text_markup(
        pdf_path, text, "highlight", page_number, color, comment, author, is_first_match_only, output_path
    )


@mcp_server.tool()
def underline_text(
    pdf_path: str,
    text: str,
    page_number: int | None = None,
    color: str = "blue",
    comment: str | None = None,
    author: str | None = None,
    is_first_match_only: bool = False,
    output_path: str | None = None,
) -> dict:
    """Underline every occurrence of `text` (optionally on a single page)."""
    return add_text_markup(
        pdf_path, text, "underline", page_number, color, comment, author, is_first_match_only, output_path
    )


@mcp_server.tool()
def strikeout_text(
    pdf_path: str,
    text: str,
    page_number: int | None = None,
    color: str = "red",
    comment: str | None = None,
    author: str | None = None,
    is_first_match_only: bool = False,
    output_path: str | None = None,
) -> dict:
    """Strike through every occurrence of `text` (optionally on a single page)."""
    return add_text_markup(
        pdf_path, text, "strikeout", page_number, color, comment, author, is_first_match_only, output_path
    )


@mcp_server.tool()
def add_comment(
    pdf_path: str,
    comment: str,
    page_number: int,
    anchor_text: str | None = None,
    x: float = 20.0,
    y: float = 20.0,
    author: str | None = None,
    icon: Literal["Note", "Comment", "Help", "Insert", "Key", "NewParagraph", "Paragraph"] = "Comment",
    output_path: str | None = None,
) -> dict:
    """Add a sticky-note comment on a page.

    If `anchor_text` is given, the note is placed next to its first occurrence on the page;
    otherwise it is placed at the (x, y) point, in PDF points from the top-left corner.
    """
    document = open_document(pdf_path)
    page = document[resolve_page_indices(document, page_number)[0]]

    if anchor_text:
        anchor_rects = page.search_for(anchor_text)
        if not anchor_rects:
            document.close()
            return {"status": "not_found", "message": f"Anchor text '{anchor_text}' not found on page {page_number}."}
        note_point = pymupdf.Point(anchor_rects[0].x1, anchor_rects[0].y0)
    else:
        note_point = pymupdf.Point(x, y)

    annotation = page.add_text_annot(note_point, comment, icon=icon)
    if author:
        annotation.set_info(title=author)
    annotation.update()
    created_annotation = describe_annotation(annotation, page_number)

    saved_path = save_document(document, pdf_path, output_path)
    return {"status": "ok", "saved_to": saved_path, "annotation": created_annotation}


@mcp_server.tool()
def add_free_text(
    pdf_path: str,
    text: str,
    page_number: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    font_size: float = 11.0,
    text_color: str = "red",
    output_path: str | None = None,
) -> dict:
    """Write visible text directly on the page inside the rectangle (x0, y0, x1, y1), in PDF points."""
    document = open_document(pdf_path)
    page = document[resolve_page_indices(document, page_number)[0]]
    annotation = page.add_freetext_annot(
        pymupdf.Rect(x0, y0, x1, y1), text, fontsize=font_size, text_color=parse_color(text_color)
    )
    annotation.update()
    created_annotation = describe_annotation(annotation, page_number)

    saved_path = save_document(document, pdf_path, output_path)
    return {"status": "ok", "saved_to": saved_path, "annotation": created_annotation}


@mcp_server.tool()
def delete_annotation(
    pdf_path: str,
    page_number: int,
    annotation_id: int,
    output_path: str | None = None,
) -> dict:
    """Delete an annotation by its `annotation_id` (as returned by `list_annotations`)."""
    document = open_document(pdf_path)
    page = document[resolve_page_indices(document, page_number)[0]]
    annotation = page.load_annot(annotation_id)
    if annotation is None:
        document.close()
        return {"status": "not_found", "message": f"No annotation {annotation_id} on page {page_number}."}
    page.delete_annot(annotation)

    saved_path = save_document(document, pdf_path, output_path)
    return {"status": "ok", "saved_to": saved_path, "deleted_annotation_id": annotation_id}


def main():
    mcp_server.run()


if __name__ == "__main__":
    main()
