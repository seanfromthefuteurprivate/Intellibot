"""
Screenshot Collector - Google Drive Integration

Watches a Google Drive folder for new trade screenshots.
Uses Application Default Credentials (ADC) with service account impersonation.
No JSON key files required.

Authentication: ADC + Service Account Impersonation
Service Account: intellibot-drive@intellibot-486323.iam.gserviceaccount.com
Scope: https://www.googleapis.com/auth/drive.readonly
"""

import os
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import threading
import time

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection

logger = get_logger(__name__)

# Service account for impersonation
SERVICE_ACCOUNT_EMAIL = "intellibot-drive@intellibot-486323.iam.gserviceaccount.com"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@dataclass
class ScreenshotFile:
    """Represents a screenshot file from Google Drive."""
    file_id: str
    name: str
    mime_type: str
    created_time: str
    modified_time: str
    size: int
    content_base64: Optional[str] = None
    content_hash: Optional[str] = None


class ScreenshotCollector:
    """
    Collects trade screenshots from Google Drive using ADC authentication.

    Usage:
        collector = ScreenshotCollector(folder_id="your_folder_id")
        collector.start_watching()  # Background polling

        # Or manual fetch
        new_screenshots = collector.fetch_new_screenshots()
    """

    # Supported image types
    SUPPORTED_MIME_TYPES = [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
    ]

    def __init__(
        self,
        folder_id: Optional[str] = None,
        scan_interval_seconds: int = 300,  # 5 minutes default
        service_account_email: str = SERVICE_ACCOUNT_EMAIL,
    ):
        """
        Initialize the Screenshot Collector.

        Args:
            folder_id: Google Drive folder ID to watch
            scan_interval_seconds: How often to check for new files
            service_account_email: Service account for impersonation
        """
        self.folder_id = folder_id or os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
        self.scan_interval = scan_interval_seconds
        self.service_account_email = service_account_email

        self._drive_service = None
        self._processed_file_ids: set = set()
        self._running = False
        self._watch_thread: Optional[threading.Thread] = None
        self._credentials_available = False  # Track if ADC is configured

        # Initialize database tables
        self._init_tables()

        # Load already processed file IDs from database
        self._load_processed_ids()

        # Check if Google Drive credentials are available (once at startup)
        self._check_credentials()

        logger.info(f"ScreenshotCollector initialized (folder: {self.folder_id}, credentials: {'available' if self._credentials_available else 'NOT CONFIGURED'})")

    def _init_tables(self):
        """Create database tables for screenshot tracking."""
        conn = get_connection()
        cursor = conn.cursor()

        # Screenshots table - tracks all ingested screenshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS screenshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT,
                file_size INTEGER,
                content_hash TEXT,
                drive_created_at TEXT,
                drive_modified_at TEXT,

                -- Processing status
                status TEXT DEFAULT 'pending',
                processed_at TEXT,
                error_message TEXT,

                -- Extracted trade data (JSON)
                extracted_data TEXT,

                -- Learning linkage
                learned_trade_id INTEGER,
                pattern_id TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Learned trades table - structured trade data from screenshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screenshot_id INTEGER,

                -- Trade identification
                ticker TEXT NOT NULL,
                trade_type TEXT,
                strike REAL,
                expiry TEXT,

                -- Execution details
                entry_price REAL,
                exit_price REAL,
                contracts INTEGER,
                shares INTEGER,

                -- P&L
                capital_deployed REAL,
                profit_loss REAL,
                profit_loss_pct REAL,

                -- Context
                platform TEXT,
                trade_date TEXT,
                entry_time TEXT,
                exit_time TEXT,
                holding_period_minutes INTEGER,

                -- Pattern analysis (from GPT-4o)
                detected_pattern TEXT,
                setup_description TEXT,
                market_conditions TEXT,

                -- Learning metadata
                confidence_score REAL,
                replication_count INTEGER DEFAULT 0,
                replication_success INTEGER DEFAULT 0,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (screenshot_id) REFERENCES screenshots(id)
            )
        """)

        # Trade recipes - distilled formulas from successful trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,

                -- Conditions
                ticker_pattern TEXT,
                trade_type TEXT,
                time_window TEXT,
                min_confidence REAL,

                -- Entry rules (JSON)
                entry_conditions TEXT,

                -- Exit rules (JSON)
                exit_conditions TEXT,

                -- Performance
                source_trade_count INTEGER DEFAULT 0,
                total_profit REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_profit_pct REAL DEFAULT 0,

                -- Status
                is_active INTEGER DEFAULT 1,
                last_used TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Screenshot database tables initialized")

    def _load_processed_ids(self):
        """Load already processed file IDs from database."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT file_id FROM screenshots")
        rows = cursor.fetchall()
        conn.close()

        self._processed_file_ids = {row["file_id"] for row in rows}
        logger.info(f"Loaded {len(self._processed_file_ids)} previously processed screenshots")

    def _check_credentials(self):
        """Check if Google Drive credentials are available at startup."""
        try:
            import google.auth
            source_credentials, project = google.auth.default()
            self._credentials_available = True
            logger.info("Google Drive ADC credentials available")
        except Exception as e:
            self._credentials_available = False
            logger.warning(f"Google Drive ADC not configured - screenshot learning disabled. To enable, run: gcloud auth application-default login")

    def is_enabled(self) -> bool:
        """Check if screenshot collector is enabled (credentials available)."""
        return self._credentials_available and bool(self.folder_id)

    def _get_drive_service(self):
        """
        Get Google Drive service using ADC with service account impersonation.
        No JSON key files - uses Application Default Credentials.
        """
        if not self._credentials_available:
            return None

        if self._drive_service is not None:
            return self._drive_service

        try:
            import google.auth
            from google.auth import impersonated_credentials
            from googleapiclient.discovery import build

            # Get default credentials (from ADC)
            source_credentials, project = google.auth.default()

            # Impersonate the service account
            target_credentials = impersonated_credentials.Credentials(
                source_credentials=source_credentials,
                target_principal=self.service_account_email,
                target_scopes=SCOPES,
            )

            # Build Drive service
            self._drive_service = build(
                "drive",
                "v3",
                credentials=target_credentials,
                cache_discovery=False
            )

            logger.info(f"Google Drive service initialized (impersonating {self.service_account_email})")
            return self._drive_service

        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            self._credentials_available = False
            return None

    def list_folder_files(self, max_results: int = 100) -> List[ScreenshotFile]:
        """
        List image files in the watched folder.

        Returns:
            List of ScreenshotFile objects
        """
        if not self.is_enabled():
            return []  # Silently return empty if not configured

        if not self.folder_id:
            return []

        try:
            service = self._get_drive_service()
            if service is None:
                return []

            # Query for image files in the folder
            mime_query = " or ".join([f"mimeType='{mt}'" for mt in self.SUPPORTED_MIME_TYPES])
            query = f"'{self.folder_id}' in parents and ({mime_query}) and trashed=false"

            results = service.files().list(
                q=query,
                pageSize=max_results,
                fields="files(id, name, mimeType, createdTime, modifiedTime, size)",
                orderBy="createdTime desc"
            ).execute()

            files = results.get("files", [])

            screenshot_files = [
                ScreenshotFile(
                    file_id=f["id"],
                    name=f["name"],
                    mime_type=f["mimeType"],
                    created_time=f.get("createdTime", ""),
                    modified_time=f.get("modifiedTime", ""),
                    size=int(f.get("size", 0)),
                )
                for f in files
            ]

            logger.debug(f"Found {len(screenshot_files)} image files in folder")
            return screenshot_files

        except Exception as e:
            logger.error(f"Failed to list folder files: {e}")
            return []

    def download_file(self, file_id: str) -> Optional[bytes]:
        """
        Download a file's content from Google Drive.

        Returns:
            File content as bytes, or None on error
        """
        try:
            service = self._get_drive_service()

            # Download file content
            request = service.files().get_media(fileId=file_id)
            content = request.execute()

            return content

        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            return None

    def fetch_new_screenshots(self) -> List[ScreenshotFile]:
        """
        Fetch new (unprocessed) screenshots from the folder.
        Downloads content and returns ready-to-process files.

        Returns:
            List of new ScreenshotFile objects with content_base64 populated
        """
        all_files = self.list_folder_files()

        new_files = []
        for file in all_files:
            if file.file_id in self._processed_file_ids:
                continue

            # Download content
            content = self.download_file(file.file_id)
            if content is None:
                continue

            # Encode to base64
            file.content_base64 = base64.b64encode(content).decode("utf-8")
            file.content_hash = hashlib.sha256(content).hexdigest()[:16]

            new_files.append(file)
            logger.info(f"New screenshot: {file.name} ({file.size} bytes)")

        return new_files

    def mark_processed(
        self,
        file: ScreenshotFile,
        status: str = "processed",
        extracted_data: Optional[Dict] = None,
        error_message: Optional[str] = None
    ) -> int:
        """
        Mark a screenshot as processed in the database.

        Returns:
            The screenshot ID
        """
        import json

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO screenshots (
                file_id, filename, mime_type, file_size, content_hash,
                drive_created_at, drive_modified_at,
                status, processed_at, error_message, extracted_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file.file_id,
            file.name,
            file.mime_type,
            file.size,
            file.content_hash,
            file.created_time,
            file.modified_time,
            status,
            datetime.utcnow().isoformat(),
            error_message,
            json.dumps(extracted_data) if extracted_data else None,
        ))

        screenshot_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Add to processed set
        self._processed_file_ids.add(file.file_id)

        return screenshot_id

    def start_watching(self):
        """Start background thread to watch for new screenshots."""
        if self._running:
            logger.warning("Screenshot watcher already running")
            return

        self._running = True
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        logger.info(f"Screenshot watcher started (interval: {self.scan_interval}s)")

    def stop_watching(self):
        """Stop the background watcher."""
        self._running = False
        if self._watch_thread:
            self._watch_thread.join(timeout=5)
        logger.info("Screenshot watcher stopped")

    def _watch_loop(self):
        """Background loop to check for new screenshots."""
        from wsb_snake.collectors.trade_extractor import TradeExtractor

        extractor = TradeExtractor()

        while self._running:
            try:
                new_files = self.fetch_new_screenshots()

                for file in new_files:
                    try:
                        # Extract trade data using GPT-4o vision
                        extracted = extractor.extract_trade_data(
                            image_base64=file.content_base64,
                            filename=file.name,
                            mime_type=file.mime_type,
                        )

                        if extracted:
                            # Mark as processed with extracted data
                            screenshot_id = self.mark_processed(
                                file,
                                status="processed",
                                extracted_data=extracted
                            )

                            # Save as learned trade
                            extractor.save_learned_trade(screenshot_id, extracted)

                            logger.info(f"Processed screenshot: {file.name} -> {extracted.get('ticker', 'UNKNOWN')}")
                        else:
                            self.mark_processed(
                                file,
                                status="failed",
                                error_message="Could not extract trade data"
                            )

                    except Exception as e:
                        logger.error(f"Error processing {file.name}: {e}")
                        self.mark_processed(
                            file,
                            status="error",
                            error_message=str(e)
                        )

            except Exception as e:
                logger.error(f"Watch loop error: {e}")

            # Wait for next scan
            time.sleep(self.scan_interval)

    def get_processing_stats(self) -> Dict:
        """Get statistics about screenshot processing."""
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
            FROM screenshots
        """)

        row = cursor.fetchone()
        conn.close()

        return {
            "total": row["total"] or 0,
            "processed": row["processed"] or 0,
            "failed": row["failed"] or 0,
            "errors": row["errors"] or 0,
            "pending": row["pending"] or 0,
        }


# Convenience function
def get_screenshot_collector(folder_id: Optional[str] = None) -> ScreenshotCollector:
    """Get or create the screenshot collector singleton."""
    global _collector_instance
    if "_collector_instance" not in globals() or _collector_instance is None:
        _collector_instance = ScreenshotCollector(folder_id=folder_id)
    return _collector_instance

_collector_instance: Optional[ScreenshotCollector] = None
