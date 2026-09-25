BEGIN;
CREATE TABLE assets(
    asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    source_modified_ns INTEGER NOT NULL,
    source_created_ns INTEGER,
    exif_taken_ns INTEGER,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    media_kind TEXT NOT NULL,
    media_format TEXT NOT NULL,
    width INTEGER CHECK(width > 0),
    height INTEGER CHECK(height > 0),
    is_animated INTEGER NOT NULL CHECK(is_animated IN (0, 1)),
    frame_count INTEGER CHECK(frame_count > 0),
    duration_ms INTEGER CHECK(duration_ms >= 0),
    metadata_version INTEGER NOT NULL DEFAULT 2 CHECK(metadata_version > 0)
);
CREATE INDEX assets_modified_idx
    ON assets(source_modified_ns, asset_id);
CREATE INDEX assets_taken_idx
    ON assets(COALESCE(exif_taken_ns, source_modified_ns), asset_id);
CREATE TABLE decode_failure_state(
    asset_id INTEGER PRIMARY KEY,
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0)
);
CREATE TABLE catalog_meta(
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    revision INTEGER NOT NULL CHECK(revision >= 0)
);
INSERT INTO catalog_meta(singleton, revision) VALUES (1, 0);
-- Library definitions. A library owns no assets: its scope is evaluated against assets.path at
-- query time, so these rows are the only thing a folder edit writes.
CREATE TABLE libraries(
    library_id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- The last scan request number handed to one of this library's folders. Numbers are never
    -- reused, so a scan that started before a folder was removed and re-added cannot complete the
    -- re-added folder's request.
    scan_sequence INTEGER NOT NULL CHECK(scan_sequence >= 0),
    index_ocr INTEGER NOT NULL CHECK(index_ocr IN (0, 1)),
    index_image INTEGER NOT NULL CHECK(index_image IN (0, 1)),
    -- Set when a library was created by importing an older client record, so the import can be
    -- repeated without creating duplicates.
    import_key TEXT UNIQUE
);
-- Included and excluded folders, in display order. For included folders a scan is outstanding
-- while scan_requested > scan_completed; a scan records the request number it started from, so a
-- request made during the scan is not lost when it finishes.
CREATE TABLE library_folders(
    library_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    excluded INTEGER NOT NULL CHECK(excluded IN (0, 1)),
    position INTEGER NOT NULL CHECK(position >= 0),
    scan_requested INTEGER NOT NULL CHECK(scan_requested >= 0),
    scan_completed INTEGER NOT NULL CHECK(scan_completed >= 0),
    scan_error TEXT,
    -- Unix nanoseconds when a scan of this folder last finished completely.
    last_scan_completed_ns INTEGER,
    PRIMARY KEY(library_id, path)
) WITHOUT ROWID;
PRAGMA user_version = 7;
COMMIT;
