"""Build report/PathSnap_Report.pdf (and .html) from report/report.md.

Run from the repo root:  python report/build_report.py
Needs: markdown, playwright (+ `playwright install chromium` once). Images are referenced relative to report/.
"""
import shutil
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC, HTML, PDF = HERE / "report.md", HERE / "PathSnap_Report.html", HERE / "PathSnap_Report.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font: 10.5pt/1.5 "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #1d2433; }
h1 { font-size: 22pt; margin: 0 0 6pt; } h2 { font-size: 15pt; border-bottom: 2px solid #1f6fd1;
padding-bottom: 3pt; margin-top: 22pt; page-break-after: avoid; } h3 { font-size: 12pt; margin-top: 14pt;
page-break-after: avoid; } h4 { font-size: 11pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt; width: 100%; }
th, td { border: 1px solid #cfd6e2; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef3fb; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; page-break-inside: avoid; }
p > em:only-child { display: block; font-size: 9pt; color: #555; text-align: center; margin-bottom: 8pt; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f4f8; padding: 0 3pt; border-radius: 3px; }
.wide table { font-size: 7pt; } .wide th, .wide td { padding: 2pt 3pt; }
.title-page { page-break-after: always; text-align: center; padding-top: 70mm; }
.title-page h2 { border: 0; font-weight: 500; } .title-page table { width: 60%; margin: 20pt auto; font-size: 11pt; }
"""


def main():
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables", "md_in_html", "toc"])
    HTML.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>PathSnap Report</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    print("wrote", HTML)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright not installed - open the HTML in a browser and Print -> Save as PDF")
    exe = shutil.which("google-chrome") or shutil.which("chromium")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        pg = b.new_page()
        pg.goto(HTML.as_uri()); pg.wait_for_load_state("networkidle")
        pg.pdf(path=str(PDF), format="A4", print_background=True, prefer_css_page_size=True,
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:8pt;width:100%;text-align:center;color:#888'>"
                               "PathSnap – Section H – page <span class='pageNumber'></span>/<span class='totalPages'></span></div>")
        b.close()
    print("wrote", PDF)


if __name__ == "__main__":
    main()
