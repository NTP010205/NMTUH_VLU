"""Rasterize a rendered PDF for per-page document QA; no approval is implied."""
import argparse
import json
from pathlib import Path
import fitz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()
    with fitz.open(args.pdf) as pdf:
        pages = []
        for i, page in enumerate(pdf):
            path = args.pdf.parent / f'page-{i+1}.png'
            page.get_pixmap(dpi=args.dpi, alpha=False).save(path)
            pages.append({'page': i + 1, 'png': path.name, 'text_length': len(page.get_text()),
                          'text': page.get_text()})
    (args.pdf.parent / 'page_inventory.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Rendered {len(pages)} page images. Inspect every image before accepting the DOCX.')


if __name__ == '__main__':
    main()
