CREATE TABLE library_directory_snapshots(
    library_id INTEGER NOT NULL,
    folder_path TEXT NOT NULL,
    path TEXT NOT NULL,
    modified_ns INTEGER NOT NULL,
    scope_key TEXT NOT NULL,
    PRIMARY KEY(library_id, folder_path, path)
) WITHOUT ROWID;
