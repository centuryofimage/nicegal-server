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
PRAGMA user_version = 6;
COMMIT;
