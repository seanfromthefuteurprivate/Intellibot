import unittest
from wsb_snake.collectors.market_data import get_market_data

class TestAlpaca(unittest.TestCase):
    def test_get_market_data(self):
        tickers = ["AAPL", "TSLA"]
        data = get_market_data(tickers)
        self.assertIn("AAPL", data)
        self.assertIn("TSLA", data)
        self.assertIsInstance(data["AAPL"], dict)
        print("Alpaca Data Test Passed")

if __name__ == '__main__':
    unittest.main()
