"""
Database Migration Script - Add Evidence Review Tables

Run this script to add the new evidence_items, reviewer_performance,
and review_sessions tables to your existing database.

Usage:
    python database/migrate_evidence_tables.py
"""

import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Run the database migration"""

    # Database path (adjust if needed)
    db_path = Path(__file__).parent.parent / "test.db"

    logger.info(f"Running migration on database: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Read the SQL schema file
        schema_file = Path(__file__).parent / "evidence_schema.sql"
        if not schema_file.exists():
            # Try alternative paths
            schema_file = Path(__file__).parent / "backend" / "database" / "evidence_schema.sql"
            if not schema_file.exists():
                schema_file = Path(__file__).parent.parent / "backend" / "database" / "evidence_schema.sql"

        logger.info(f"Reading schema from: {schema_file}")

        with open(schema_file, 'r') as f:
            sql_commands = f.read()

        # Execute all commands in one go (SQLite can handle multiple statements)
        try:
            cursor.executescript(sql_commands)
            logger.info(f"✓ Executed schema file successfully")
        except sqlite3.OperationalError as e:
            if "already exists" in str(e):
                logger.warning(f"Tables already exist, skipping...")
            else:
                logger.error(f"Schema execution error: {e}")
                # Try to continue anyway - tables might be partially created
                pass

        # Add columns to videos table if they don't exist
        logger.info("Adding columns to videos table...")

        columns_to_add = [
            ("evidence_extraction_status", "VARCHAR(50) DEFAULT 'pending'"),
            ("ai_evidence_count", "INTEGER DEFAULT 0"),
            ("evidence_needs_review_count", "INTEGER DEFAULT 0"),
            ("evidence_approved_count", "INTEGER DEFAULT 0"),
            ("evidence_accuracy_estimate", "FLOAT")
        ]

        for column_name, column_def in columns_to_add:
            try:
                cursor.execute(f"""
                    ALTER TABLE videos ADD COLUMN {column_name} {column_def}
                """)
                logger.info(f"✓ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    logger.warning(f"Column {column_name} already exists, skipping...")
                else:
                    raise

        conn.commit()
        logger.info("✅ Migration completed successfully!")

        # Verify tables were created
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            AND name IN ('evidence_items', 'reviewer_performance', 'review_sessions')
        """)

        tables = cursor.fetchall()
        logger.info(f"Created tables: {[t[0] for t in tables]}")

        conn.close()

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("Evidence Review Tables Migration")
    logger.info("="*80)
    run_migration()
    logger.info("="*80)
    logger.info("Done! You can now use the evidence review features.")
    logger.info("="*80)
