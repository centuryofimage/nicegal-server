CREATE VIRTUAL TABLE asset_path_fts USING fts5(path, tokenize='trigram', detail='none');
INSERT INTO asset_path_fts(rowid, path)
    SELECT asset_id, replace(path, char(92), '/') FROM assets;
CREATE TRIGGER asset_path_fts_insert AFTER INSERT ON assets BEGIN
    INSERT INTO asset_path_fts(rowid, path)
        VALUES (new.asset_id, replace(new.path, char(92), '/'));
END;
CREATE TRIGGER asset_path_fts_update AFTER UPDATE OF path ON assets BEGIN
    DELETE FROM asset_path_fts WHERE rowid = old.asset_id;
    INSERT INTO asset_path_fts(rowid, path)
        VALUES (new.asset_id, replace(new.path, char(92), '/'));
END;
CREATE TRIGGER asset_path_fts_delete AFTER DELETE ON assets BEGIN
    DELETE FROM asset_path_fts WHERE rowid = old.asset_id;
END;
