import unittest
from wsb_snake.analysis.sentiment import summarize_setup

class TestOpenAI(unittest.TestCase):
    def test_summarize(self):
        summary = summarize_setup("TSLA")
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
        print("OpenAI Sentiment Test Passed")

if __name__ == '__main__':
    unittest.main()
