#!/usr/bin/env python3
"""
🚀 Universal PDF to Markdown Converter using PyMuPDF4LLM (2025 Edition)
========================================================================

The best PDF to markdown converter for scientific papers, with:
- Equations & tables preservation
- Image extraction with proper filenames
- Batch processing support
- Optimal settings for LLM consumption

Usage:
    python pdf_to_markdown.py <pdf_file>           # Single file
    python pdf_to_markdown.py <pdf_file> -o <dir>  # Custom output
    python pdf_to_markdown.py --all                # All PDFs in pdfs/
    python pdf_to_markdown.py --help               # Show help
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pymupdf4llm
import logging
from datetime import datetime
import json
import pymupdf
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PDFToMarkdownConverter:
    """Production-ready PDF to Markdown converter optimized for scientific papers."""
    
    # Optimal settings for scientific papers (2025 best practices)
    DEFAULT_CONFIG = {
        'write_images': True,        # Extract images to files
        'image_format': 'png',        # PNG for quality
        'image_size_limit': 0.05,     # 5% of page = image worth extracting
        'dpi': 150,                   # Good quality/size balance
        'table_strategy': 'lines_strict',  # Best for scientific tables
        'fontsize_limit': 3,          # Ignore tiny text (page numbers, etc)
        'page_chunks': False,         # Keep document flow
        'force_text': True,           # Extract even from images when possible
        'show_progress': True,        # Show conversion progress
        'margins': 0,                 # Use full page
        'detect_bg_color': True,      # Preserve background highlights
        'use_ocr': False,             # Use OCR for scanned PDFs
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize converter with optional custom config."""
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
    
    def convert(
        self, 
        pdf_path: Path, 
        output_dir: Optional[Path] = None,
        overwrite: bool = False
    ) -> Path:
        """
        Convert a single PDF to markdown.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Optional output directory (default: markdown/<pdf_name>/)
            overwrite: Whether to overwrite existing output
            
        Returns:
            Path to the created markdown file
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Setup output paths
        stem = pdf_path.stem
        if output_dir is None:
            output_dir = Path(__file__).parent / 'markdown' / stem
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{stem}.md"
        
        # Check if already exists
        if md_path.exists() and not overwrite:
            logger.info(f"✓ Already converted: {md_path}")
            return md_path
        
        # Convert
        logger.info(f"📄 Converting: {pdf_path.name}")
        start = datetime.now()
        
        try:
            # Open PDF
            doc = pymupdf4llm.pymupdf.open(pdf_path)
            
            # Configure image path for extraction
            image_config = self.config.copy()
            # Remove our custom keys that pymupdf4llm doesn't understand
            use_ocr = image_config.pop('use_ocr', False)
            force_ocr = image_config.pop('force_ocr', False)
            if image_config['write_images']:
                image_config['image_path'] = str(output_dir)
            
            # Try normal extraction first
            md_text = pymupdf4llm.to_markdown(
                doc,
                **image_config
            )

            # If forcing OCR or if text is mostly empty/numbers and OCR is requested
            text_without_spaces = ''.join(md_text.split())
            should_ocr = force_ocr or (use_ocr and (len(md_text.strip()) < 1000 or
                           (text_without_spaces and text_without_spaces.replace('*','').isdigit())))

            if should_ocr:
                logger.info(f"Text extraction yielded little content, attempting OCR...")
                md_text = self._ocr_extract(pdf_path, output_dir)
                if not md_text:
                    logger.warning(f"OCR extraction also failed, using original extraction")
                    md_text = pymupdf4llm.to_markdown(doc, **image_config)
            
            # Save markdown
            md_path.write_text(md_text, encoding='utf-8')
            
            # Count extracted images
            image_count = len(list(output_dir.glob('*.png')))
            
            # Calculate stats
            duration = (datetime.now() - start).total_seconds()
            file_size = md_path.stat().st_size / 1024  # KB
            
            logger.info(
                f"✅ Converted in {duration:.1f}s: "
                f"{len(md_text):,} chars, {image_count} images, {file_size:.1f}KB"
            )
            
            # Save metadata
            meta_path = output_dir / 'metadata.json'
            metadata = {
                'source_pdf': str(pdf_path),
                'converted_at': datetime.now().isoformat(),
                'duration_seconds': duration,
                'markdown_size': len(md_text),
                'images_extracted': image_count,
                'config': self.config,
                'pages': len(doc)
            }
            meta_path.write_text(json.dumps(metadata, indent=2))
            
            doc.close()
            return md_path
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {pdf_path.name}: {e}")
            raise

    def _ocr_extract(self, pdf_path: Path, output_dir: Path) -> str:
        """Extract text using OCR for scanned PDFs."""
        if not OCR_AVAILABLE:
            logger.warning("OCR libraries not available (install pytesseract and pillow)")
            return ""

        try:
            doc = pymupdf.open(pdf_path)
            full_text = []

            for page_num, page in enumerate(doc):
                # Get page as image
                mat = pymupdf.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.pil_tobytes(format="PNG")

                # Convert to PIL Image and run OCR
                image = Image.open(io.BytesIO(img_data))

                try:
                    # Try OCR with Tesseract
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        full_text.append(f"## Page {page_num + 1}\n\n{text}\n")
                        logger.info(f"OCR extracted {len(text)} chars from page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"OCR failed on page {page_num + 1}: {e}")

            doc.close()
            return "\n".join(full_text)

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def convert_all(self, pdf_dir: Path = None, overwrite: bool = False) -> Dict[str, Path]:
        """
        Convert all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDFs (default: pdfs/)
            overwrite: Whether to overwrite existing conversions
            
        Returns:
            Dictionary mapping PDF names to markdown paths
        """
        if pdf_dir is None:
            pdf_dir = Path(__file__).parent / 'pdfs'
        
        pdf_files = sorted(pdf_dir.glob('*.pdf'))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {}
        
        logger.info(f"🚀 Converting {len(pdf_files)} PDFs...")
        results = {}
        
        for pdf_path in pdf_files:
            try:
                md_path = self.convert(pdf_path, overwrite=overwrite)
                results[pdf_path.stem] = md_path
            except Exception as e:
                logger.error(f"Skipping {pdf_path.name}: {e}")
                results[pdf_path.stem] = None
        
        # Summary
        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"✨ Conversion complete: {successful}/{len(pdf_files)} successful")
        
        return results


def main():
    """CLI interface for PDF to Markdown conversion."""
    parser = argparse.ArgumentParser(
        description='Convert PDFs to Markdown using PyMuPDF4LLM (2025 best practices)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_to_markdown.py pdfs/EEGPT.pdf              # Convert single PDF
  python pdf_to_markdown.py pdfs/EEGPT.pdf -o output/   # Custom output dir
  python pdf_to_markdown.py --all                       # Convert all PDFs
  python pdf_to_markdown.py --all --overwrite           # Force reconvert
  python pdf_to_markdown.py --config '{"dpi": 300}'     # Custom settings
        """
    )
    
    parser.add_argument(
        'pdf_file',
        nargs='?',
        help='PDF file to convert'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory (default: markdown/<pdf_name>/)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Convert all PDFs in pdfs/ directory'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing conversions'
    )
    parser.add_argument(
        '--config',
        help='Custom config as JSON string'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip image extraction'
    )
    parser.add_argument(
        '--ocr',
        action='store_true',
        help='Use OCR for scanned PDFs (requires tesseract)'
    )
    parser.add_argument(
        '--force-ocr',
        action='store_true',
        help='Force OCR extraction regardless of text content'
    )
    
    args = parser.parse_args()
    
    # Parse custom config
    config = {}
    if args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON config: {e}")
            sys.exit(1)
    
    if args.no_images:
        config['write_images'] = False

    if args.ocr:
        config['use_ocr'] = True
        logger.info("OCR mode enabled for scanned PDFs")

    if hasattr(args, 'force_ocr') and args.force_ocr:
        config['force_ocr'] = True
        config['use_ocr'] = True
        logger.info("Forcing OCR extraction")
    
    # Create converter
    converter = PDFToMarkdownConverter(config)
    
    # Execute conversion
    try:
        if args.all:
            results = converter.convert_all(overwrite=args.overwrite)
            
            # Print summary
            print("\n📊 Conversion Summary:")
            print("-" * 50)
            for name, path in results.items():
                status = "✅" if path else "❌"
                print(f"{status} {name}")
        
        elif args.pdf_file:
            pdf_path = Path(args.pdf_file)
            output_dir = Path(args.output) if args.output else None
            md_path = converter.convert(
                pdf_path, 
                output_dir=output_dir,
                overwrite=args.overwrite
            )
            print(f"\n✅ Markdown saved to: {md_path}")
        
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()