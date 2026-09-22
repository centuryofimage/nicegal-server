BEGIN;
CREATE TABLE image_embedding_state(
    asset_id INTEGER PRIMARY KEY NOT NULL CHECK(asset_id > 0),
    source_path TEXT NOT NULL,
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    sampling_version INTEGER NOT NULL DEFAULT 0
);
PRAGMA user_version = 3;
COMMIT;
