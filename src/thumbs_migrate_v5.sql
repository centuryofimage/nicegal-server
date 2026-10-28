-- Version 4 databases created before video thumbnails existed lack the table.
CREATE TABLE IF NOT EXISTS video_thumbnails(
    asset_id INTEGER NOT NULL CHECK(asset_id > 0),
    timestamp_ms INTEGER NOT NULL CHECK(timestamp_ms >= 0),
    size_bucket INTEGER NOT NULL CHECK(size_bucket IN (128, 256, 512, 1024)),
    sampling_version INTEGER NOT NULL CHECK(sampling_version > 0),
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    encoding TEXT NOT NULL CHECK(encoding IN ('image/jpeg', 'image/png', 'image/webp')),
    data BLOB NOT NULL,
    PRIMARY KEY(asset_id, timestamp_ms, size_bucket)
) WITHOUT ROWID;
DROP INDEX IF EXISTS thumbnail_lookup_idx;
DROP INDEX IF EXISTS video_thumbnail_assets_idx;
CREATE INDEX thumbnail_current_idx
    ON thumbnails(asset_id, generator_version, source_modified_ns, source_size, size_bucket);
CREATE INDEX video_thumbnail_current_idx
    ON video_thumbnails(asset_id, sampling_version, source_modified_ns, source_size);
