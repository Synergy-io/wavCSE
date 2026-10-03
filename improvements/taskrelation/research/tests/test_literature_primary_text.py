"""Primary text is extracted deterministically, page-indexed and fail-closed.

No PDF is committed: the backend is exercised through an injected runner, and
the real ``pdftotext`` is used only in a gated smoke that skips when it is absent.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_primary_text as lt


class RunnerStub:
    """A stand-in for the extraction subprocess, driven by fixed outputs."""

    def __init__(self, pages=b"", returncode=0, stderr=b"", version=b"pdftotext version 1.2.3"):
        self.pages = pages
        self.returncode = returncode
        self.stderr = stderr
        self.version = version
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if "-v" in argv:
            return 0, b"", self.version
        return self.returncode, self.pages, self.stderr


class SplitPagesTests(unittest.TestCase):
    def test_pages_are_split_on_the_form_feed_terminator(self):
        self.assertEqual(lt.split_pages("alpha\fbeta\fgamma\f"), ("alpha", "beta", "gamma"))

    def test_an_empty_page_is_preserved(self):
        self.assertEqual(lt.split_pages("one\f\ftwo\f"), ("one", "", "two"))

    def test_empty_output_is_no_pages(self):
        self.assertEqual(lt.split_pages(""), ())

    def test_missing_trailing_separator_keeps_the_last_page(self):
        self.assertEqual(lt.split_pages("a\fb"), ("a", "b"))


class LocatorTests(unittest.TestCase):
    def test_single_page_locator(self):
        self.assertEqual(lt.page_locator(4), "primary:page:4")
        self.assertEqual(lt.page_locator(4, 4), "primary:page:4")

    def test_range_locator(self):
        self.assertEqual(lt.page_locator(3, 5), "primary:pages:3-5")

    def test_max_chars_bounds(self):
        self.assertEqual(lt.validate_max_chars(None), lt.DEFAULT_MAX_CHARS)
        self.assertEqual(lt.validate_max_chars(10), 10)
        for value in (0, -1, lt.MAX_CHARS_CEILING + 1, "10", True):
            with self.subTest(value=value):
                with self.assertRaises(lt.PrimaryTextError) as caught:
                    lt.validate_max_chars(value)
                self.assertEqual(caught.exception.kind, lt.INVALID_LOCATOR)

    def test_page_is_a_one_based_physical_index(self):
        self.assertEqual(lt.validate_page(1, 3), 1)
        self.assertEqual(lt.validate_page(3, 3), 3)
        for value in (0, -1, 4, "2", None, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(lt.PrimaryTextError) as caught:
                    lt.validate_page(value, 3)
                self.assertEqual(caught.exception.kind, lt.INVALID_LOCATOR)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="primary-text-")
        self.pdf = Path(self.tempdir.name) / "source.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 placeholder")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_pages_and_provenance_are_returned(self):
        runner = RunnerStub(pages=b"page one\fpage two\fpage three\f", stderr=b"a warning\n")

        document = lt.extract_document(self.pdf, binary="pdftotext", runner=runner)

        self.assertEqual(document.page_count, 3)
        self.assertEqual(document.pages, ("page one", "page two", "page three"))
        self.assertEqual(document.extractor, "pdftotext")
        self.assertEqual(document.extractor_version, "1.2.3")
        self.assertEqual(document.warnings, ("a warning",))

    def test_missing_backend_is_reported_not_guessed(self):
        with self.assertRaises(lt.PrimaryTextError) as caught:
            lt.extract_document(self.pdf, binary="wavcse-no-such-extractor")

        self.assertEqual(caught.exception.kind, lt.EXTRACTOR_UNAVAILABLE)

    def test_nonzero_exit_is_a_structured_failure(self):
        runner = RunnerStub(pages=b"junk", returncode=2)

        with self.assertRaises(lt.PrimaryTextError) as caught:
            lt.extract_document(self.pdf, binary="pdftotext", runner=runner)

        self.assertEqual(caught.exception.kind, lt.EXTRACTION_FAILED)
        self.assertEqual(caught.exception.detail["returncode"], 2)

    def test_extract_text_joins_pages(self):
        runner = RunnerStub(pages=b"one\ftwo\f")

        self.assertEqual(
            lt.extract_text(self.pdf, binary="pdftotext", runner=runner), "one\n\ntwo"
        )


@unittest.skipUnless(shutil.which("pdftotext"), "poppler pdftotext is not installed")
class RealBackendSmokeTests(unittest.TestCase):
    def test_real_backend_extracts_a_text_layer(self):
        # A minimal hand-written single-page PDF with a text object; proves the
        # real backend is wired without committing a paper PDF to Git.
        content = b"BT /F1 12 Tf 72 720 Td (wavcse smoke) Tj ET"
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n"
            + content + b"\nendstream",
        ]
        chunks = [b"%PDF-1.4\n"]
        offsets = []
        for index, body in enumerate(objects, start=1):
            offsets.append(sum(len(chunk) for chunk in chunks))
            chunks.append("{} 0 obj\n".format(index).encode("ascii") + body + b"\nendobj\n")
        xref_at = sum(len(chunk) for chunk in chunks)
        chunks.append("xref\n0 {}\n".format(len(objects) + 1).encode("ascii"))
        chunks.append(b"0000000000 65535 f \n")
        for offset in offsets:
            chunks.append("{:010d} 00000 n \n".format(offset).encode("ascii"))
        chunks.append(
            ("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n".format(
                len(objects) + 1, xref_at
            )).encode("ascii")
        )
        with tempfile.TemporaryDirectory(prefix="primary-text-real-") as tmp:
            pdf = Path(tmp) / "smoke.pdf"
            pdf.write_bytes(b"".join(chunks))

            document = lt.extract_document(pdf)

        self.assertEqual(document.page_count, 1)
        self.assertIn("wavcse smoke", document.pages[0])


if __name__ == "__main__":
    unittest.main()
