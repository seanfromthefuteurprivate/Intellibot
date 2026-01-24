import unittest
from wsb_snake.collectors.reddit_collector import collect_mentions

class TestRedditCollector(unittest.TestCase):
    def test_collect_mentions(self):
        tickers = collect_mentions()
        self.assertIsInstance(tickers, list)
        self.assertTrue(len(tickers) > 0)
        print("Reddit Collector Test Passed")

if __name__ == '__main__':
    unittest.main()
