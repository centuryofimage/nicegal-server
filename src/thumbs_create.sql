BEGIN;
CREATE TABLE thumbnails(
    asset_id INTEGER NOT NULL CHECK(asset_id > 0),
    size_bucket INTEGER NOT NULL CHECK(size_bucket IN (128, 256, 512, 1024)),
    generator_version INTEGER NOT NULL CHECK(generator_version > 0),
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    encoding TEXT NOT NULL CHECK(encoding IN ('image/jpeg', 'image/png', 'image/webp')),
    data BLOB NOT NULL,
    PRIMARY KEY(asset_id, size_bucket, generator_version)
) WITHOUT ROWID;
CREATE INDEX thumbnail_lookup_idx
    ON thumbnails(asset_id, generator_version, size_bucket);
PRAGMA user_version = 3;
COMMIT;
