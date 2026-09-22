BEGIN;
CREATE TABLE ocr_results(
    asset_id INTEGER PRIMARY KEY NOT NULL CHECK(asset_id > 0),
    source_path TEXT NOT NULL,
    source_modified_ns INTEGER NOT NULL,
    -- Denormalized from the asset catalog for the same reason source_path is: search filters on it
    -- and the OCR store is opened alone by the read-only search connection.
    exif_taken_ns INTEGER,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    mark_delete INTEGER NOT NULL DEFAULT 0 CHECK(mark_delete IN (0, 1)),
    content TEXT NOT NULL
);
CREATE INDEX ocr_mark_delete_idx ON ocr_results(mark_delete);
CREATE INDEX ocr_source_path_idx ON ocr_results(source_path);
-- One index per timeline, matching the two expressions a time-filtered search can order on.
CREATE INDEX ocr_modified_idx ON ocr_results(source_modified_ns);
CREATE INDEX ocr_taken_idx ON ocr_results(COALESCE(exif_taken_ns, source_modified_ns));
CREATE VIRTUAL TABLE ocr_results_fts USING fts5(content, content=ocr_results, content_rowid=asset_id, tokenize='trigram case_sensitive 0');
-- Glob search is against these unicode61 tokens, rather than byte substrings in the OCR blob.
-- The row vocabulary finds candidate terms once; the instance vocabulary then follows only their
-- postings to asset IDs, avoiding a scan of every OCR document for every glob query.
CREATE VIRTUAL TABLE ocr_results_words_fts USING fts5(content, content=ocr_results, content_rowid=asset_id, tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE ocr_results_words_vocab USING fts5vocab(ocr_results_words_fts, 'row');
CREATE VIRTUAL TABLE ocr_results_words_vocab_instance USING fts5vocab(ocr_results_words_fts, 'instance');
CREATE TRIGGER ocr_results_insert AFTER INSERT ON ocr_results BEGIN
    INSERT INTO ocr_results_fts(rowid, content) VALUES (new.asset_id, new.content);
    INSERT INTO ocr_results_words_fts(rowid, content) VALUES (new.asset_id, new.content);
END;
CREATE TRIGGER ocr_results_delete AFTER DELETE ON ocr_results BEGIN
    INSERT INTO ocr_results_fts(ocr_results_fts, rowid, content) VALUES ('delete', old.asset_id, old.content);
    INSERT INTO ocr_results_words_fts(ocr_results_words_fts, rowid, content) VALUES ('delete', old.asset_id, old.content);
END;
-- Bookkeeping and unchanged OCR writes must not rebuild either text index.
CREATE TRIGGER ocr_results_update AFTER UPDATE OF asset_id, content ON ocr_results
WHEN old.asset_id IS NOT new.asset_id OR old.content IS NOT new.content BEGIN
    INSERT INTO ocr_results_fts(ocr_results_fts, rowid, content) VALUES ('delete', old.asset_id, old.content);
    INSERT INTO ocr_results_fts(rowid, content) VALUES (new.asset_id, new.content);
    INSERT INTO ocr_results_words_fts(ocr_results_words_fts, rowid, content) VALUES ('delete', old.asset_id, old.content);
    INSERT INTO ocr_results_words_fts(rowid, content) VALUES (new.asset_id, new.content);
END;

-- The text-embedding API uses `space` to name its persisted coordinate system. Image embeddings
-- live in their own model-specific database and are not constrained by this OCR schema.
CREATE TABLE ocr_embedding_model(
    space TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0)
);
-- Which assets have a current vector in a space, and the source fingerprint it was computed from.
-- Kept as an ordinary table so coverage and backlog are answerable in plain SQL; the vectors
-- themselves live in one `ocr_embeddings_<space>` vec0 table per space, keyed by the same asset_id.
CREATE TABLE ocr_embedding_state(
    space TEXT NOT NULL,
    asset_id INTEGER NOT NULL REFERENCES ocr_results(asset_id) ON DELETE CASCADE,
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    PRIMARY KEY(space, asset_id)
);
PRAGMA user_version = 10;
COMMIT;
