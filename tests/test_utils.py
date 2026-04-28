import unittest
from utils.processing import get_summary_text

class TestProcessingUtils(unittest.TestCase):
    def test_get_summary_text_empty(self):
        self.assertEqual(get_summary_text({}), "No weapons detected.")
        self.assertEqual(get_summary_text({}, lang='te'), "ఏ ఆయుధాలు గుర్తించబడలేదు.")

    def test_get_summary_text_with_counts(self):
        counts = {"Gun": 2, "Knife": 1}
        summary = get_summary_text(counts)
        self.assertIn("Gun: 2", summary)
        self.assertIn("Knife: 1", summary)
        self.assertTrue(summary.startswith("Detected: "))

if __name__ == '__main__':
    unittest.main()
