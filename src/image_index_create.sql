BEGIN;
CREATE TABLE image_embedding_state(
    asset_id INTEGER PRIMARY KEY NOT NULL CHECK(asset_id > 0),
    source_path TEXT NOT NULL,
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    sampling_version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE image_embedding_samples(
    embedding_id INTEGER PRIMARY KEY,
    asset_id INTEGER NOT NULL,
    timestamp_ms INTEGER
);
CREATE INDEX image_embedding_samples_asset_idx
    ON image_embedding_samples(asset_id, timestamp_ms);
CREATE INDEX image_embedding_video_samples_idx
    ON image_embedding_samples(asset_id, timestamp_ms) WHERE timestamp_ms IS NOT NULL;
CREATE TABLE image_embedding_sample_migration(
    singleton INTEGER PRIMARY KEY,
    complete INTEGER NOT NULL
);
INSERT INTO image_embedding_sample_migration VALUES (1, 1);
PRAGMA user_version = 4;
COMMIT;
