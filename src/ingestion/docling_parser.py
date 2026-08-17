"""
Docling based PDF parser.

Responsibilities:
1. Convert PDF using Docling.
2. Extract text, tables and images.
3. Return normalized multimodal elements.

Output format:

[
    {
        "content": "...",
        "content_type": "text/table/image",
        "metadata": {}
    }
]
"""

import os

from docling.datamodel.base_models import InputFormat

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from src.common.logger import logger

# ============================================================
# Runtime configuration
# ============================================================

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"


# ============================================================
# Metadata
# ============================================================


def _create_metadata(
    source_file: str,
    page_number: int | None,
    section: str | None,
    content_type: str,
):
    return {
        "source_file": source_file,
        "page_number": page_number,
        "section": section,
        "content_type": content_type,
    }


# ============================================================
# Parse Document
# ============================================================


def parse_document(
    file_path: str,
) -> list[dict]:
    """
    Parse PDF using Docling.

    Returns:
        List of normalized document elements.
    """

    # -------------------------------------------------
    # Docling configuration
    # -------------------------------------------------

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        accelerator_options=AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU,
        ),
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    # -------------------------------------------------
    # Convert document
    # -------------------------------------------------

    try:

        result = converter.convert(file_path)

        document = result.document

    except Exception:

        logger.exception(
            "Docling conversion failed | file=%s",
            os.path.basename(file_path),
        )

        raise

    elements = []

    source_file = os.path.basename(file_path)

    current_section = None

    # -------------------------------------------------
    # Counters for observability
    # -------------------------------------------------

    text_count = 0
    table_count = 0
    image_count = 0
    skipped_header_footer_count = 0
    table_warning_count = 0

    # -------------------------------------------------
    # Iterate Docling elements
    # -------------------------------------------------

    try:

        for item in document.iterate_items():

            if isinstance(
                item,
                tuple,
            ):

                node, _ = item

            else:

                node = item

            label = str(
                getattr(
                    node,
                    "label",
                    "",
                )
            ).lower()

            # -------------------------------------------------
            # Skip repeated headers/footer
            # -------------------------------------------------

            if label in (
                "page_header",
                "page_footer",
            ):

                skipped_header_footer_count += 1

                continue

            # -------------------------------------------------
            # Page number
            # -------------------------------------------------

            page_number = None

            prov = getattr(
                node,
                "prov",
                None,
            )

            if prov:

                page_number = prov[0].page_no

            text = getattr(
                node,
                "text",
                "",
            )

            # -------------------------------------------------
            # Section headings
            # -------------------------------------------------

            if "section_header" in label or label == "title":

                if text:

                    current_section = text.strip()

                    elements.append(
                        {
                            "content": (text.strip()),
                            "content_type": "text",
                            "metadata": _create_metadata(
                                source_file,
                                page_number,
                                current_section,
                                "text",
                            ),
                        }
                    )

                    text_count += 1

            # -------------------------------------------------
            # Tables
            # -------------------------------------------------

            elif "table" in label:

                table_content = ""

                try:

                    df = node.export_to_dataframe(doc=document)

                    if df is not None:

                        table_content = df.to_markdown(index=False)

                except Exception:

                    table_warning_count += 1

                    logger.exception(
                        "Table extraction failed | " "file=%s | page=%s",
                        source_file,
                        page_number,
                    )

                if table_content:

                    elements.append(
                        {
                            "content": table_content,
                            "content_type": "table",
                            "metadata": _create_metadata(
                                source_file,
                                page_number,
                                current_section,
                                "table",
                            ),
                        }
                    )

                    table_count += 1

            # -------------------------------------------------
            # Images / Figures
            # -------------------------------------------------

            elif "picture" in label or "figure" in label or "chart" in label:

                elements.append(
                    {
                        "content": (text.strip() if text else "[image]"),
                        "content_type": "image",
                        "metadata": _create_metadata(
                            source_file,
                            page_number,
                            current_section,
                            "image",
                        ),
                    }
                )

                image_count += 1

            # -------------------------------------------------
            # Normal text
            # -------------------------------------------------

            else:

                if text and text.strip():

                    elements.append(
                        {
                            "content": (text.strip()),
                            "content_type": "text",
                            "metadata": _create_metadata(
                                source_file,
                                page_number,
                                current_section,
                                "text",
                            ),
                        }
                    )

                    text_count += 1

    except Exception:

        logger.exception(
            "Docling element extraction failed | file=%s",
            source_file,
        )

        raise

    # -------------------------------------------------
    # Parsing summary
    # -------------------------------------------------

    return elements


# ---------------------------------------------------------
# Local test
# ---------------------------------------------------------


if __name__ == "__main__":

    chunks = parse_document("data/uploads/KB_Smart_Banking.pdf")

    print("\nSample output")

    print("=" * 60)

    for item in chunks[:5]:

        print(
            "\nType:",
            item["content_type"],
        )

        print(
            "Metadata:",
            item["metadata"],
        )

        print(
            "Content:",
            item["content"][:200],
        )
