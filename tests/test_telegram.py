import unittest
from wsb_snake.notifications.telegram_bot import send_alert

class TestTelegram(unittest.TestCase):
    def test_send_alert(self):
        # This just tests the placeholder, doesn't actually send
        try:
            send_alert("Test message")
            print("Telegram Alert Test Passed")
        except Exception as e:
            self.fail(f"Telegram alert failed: {e}")

if __name__ == '__main__':
    unittest.main()
