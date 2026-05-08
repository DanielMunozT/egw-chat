import unittest

from egw_corpus.vector_store import chunk_paragraphs, count_tokens, resolve_pagination


class VectorStoreTests(unittest.TestCase):
    def test_resolve_pagination_with_page(self) -> None:
        page = resolve_pagination(page_size=15, page=3)
        self.assertEqual(page["page_size"], 15)
        self.assertEqual(page["page"], 3)
        self.assertEqual(page["offset"], 30)

    def test_resolve_pagination_with_offset(self) -> None:
        page = resolve_pagination(page_size=15, offset=45)
        self.assertEqual(page["page"], 4)
        self.assertEqual(page["offset"], 45)

    def test_chunk_paragraphs_respects_token_limit_and_overlap(self) -> None:
        text = "\n\n".join(
            f"SECTION {i}\n"
            + ("This is a long paragraph about health reform and Christian living. " * 30)
            + f"[GC {i}.1]"
            for i in range(1, 8)
        )
        chunks = chunk_paragraphs(text, chunk_tokens=120, overlap_tokens=40)
        self.assertGreaterEqual(len(chunks), 2)

        for chunk in chunks:
            self.assertLessEqual(count_tokens(chunk["text"]), 120)
            self.assertTrue(chunk["text"].strip())

        overlap = set(chunks[0]["text"].split()) & set(chunks[1]["text"].split())
        self.assertTrue(overlap)


if __name__ == "__main__":
    unittest.main()
