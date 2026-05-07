import unittest

from egw_corpus.vector_store import resolve_pagination


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


if __name__ == "__main__":
    unittest.main()
