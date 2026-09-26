-- Video indexing moves from the app-wide runtime settings to each library. The server copies an
-- older "off" choice into every library at startup.
ALTER TABLE libraries ADD COLUMN index_videos INTEGER NOT NULL DEFAULT 1 CHECK(index_videos IN (0, 1));
