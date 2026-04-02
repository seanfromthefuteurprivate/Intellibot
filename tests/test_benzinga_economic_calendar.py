import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from wsb_snake.collectors.benzinga_economic_calendar import BenzingaEconomicCalendarAdapter


SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<result>
    <economics is_array="true">
        <item>
            <country>USA</country>
            <prior>0.500</prior>
            <time>08:30:00</time>
            <event_period></event_period>
            <importance>3</importance>
            <event_category>Inflation</event_category>
            <consensus></consensus>
            <id>ppi-1</id>
            <updated>1773333718</updated>
            <event_name>PPI (MoM)</event_name>
            <date>2026-03-18</date>
            <actual></actual>
            <consensus_t>%</consensus_t>
            <notes></notes>
            <actual_t>%</actual_t>
            <period_year>2026</period_year>
            <prior_t>%</prior_t>
            <description>PPI event</description>
        </item>
        <item>
            <country>USA</country>
            <prior>3.750</prior>
            <time>13:00:00</time>
            <event_period></event_period>
            <importance>3</importance>
            <event_category>Central Banks</event_category>
            <consensus>3.750</consensus>
            <id>fomc-1</id>
            <updated>1773729191</updated>
            <event_name>Fed Interest Rate Decision</event_name>
            <date>2026-03-18</date>
            <actual></actual>
            <consensus_t>%</consensus_t>
            <notes></notes>
            <actual_t>%</actual_t>
            <period_year>2026</period_year>
            <prior_t>%</prior_t>
            <description>Rate decision</description>
        </item>
        <item>
            <country>USA</country>
            <prior></prior>
            <time>14:30:00</time>
            <event_period></event_period>
            <importance>3</importance>
            <event_category>Central Banks</event_category>
            <consensus></consensus>
            <id>fomc-2</id>
            <updated>1771973183</updated>
            <event_name>FOMC Press Conference</event_name>
            <date>2026-03-18</date>
            <actual></actual>
            <consensus_t></consensus_t>
            <notes></notes>
            <actual_t></actual_t>
            <period_year>2026</period_year>
            <prior_t></prior_t>
            <description>Press conference</description>
        </item>
    </economics>
</result>
"""


class TestBenzingaEconomicCalendar(unittest.TestCase):
    def setUp(self):
        self.adapter = BenzingaEconomicCalendarAdapter()

    def test_parse_xml_payload(self):
        now = datetime(2026, 3, 18, 12, 45, tzinfo=ZoneInfo("America/New_York"))
        rows = self.adapter._parse_payload(SAMPLE_XML)
        events = [self.adapter._normalize_event(row, now) for row in rows]
        events = [event for event in events if event]

        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["event_code"], "PPI")
        self.assertEqual(events[1]["event_code"], "FOMC")
        self.assertEqual(events[1]["time_label"], "1:00 PM ET")

    def test_event_gate_blocks_before_release(self):
        now = datetime(2026, 3, 18, 12, 45, tzinfo=ZoneInfo("America/New_York"))
        rows = self.adapter._parse_payload(SAMPLE_XML)
        events = [self.adapter._normalize_event(row, now) for row in rows]
        events = [event for event in events if event]

        gate = self.adapter.get_event_gate(
            block_before_minutes=30,
            block_after_minutes=15,
            now=now,
            events=events,
        )

        self.assertTrue(gate["blocked"])
        self.assertIn("Fed Interest Rate Decision", gate["reason"])

    def test_event_gate_blocks_during_cooldown(self):
        now = datetime(2026, 3, 18, 13, 7, tzinfo=ZoneInfo("America/New_York"))
        rows = self.adapter._parse_payload(SAMPLE_XML)
        events = [self.adapter._normalize_event(row, now) for row in rows]
        events = [event for event in events if event]

        gate = self.adapter.get_event_gate(
            block_before_minutes=30,
            block_after_minutes=15,
            now=now,
            events=events,
        )

        self.assertTrue(gate["blocked"])
        self.assertIn("COOLDOWN", gate["reason"])


if __name__ == "__main__":
    unittest.main()
