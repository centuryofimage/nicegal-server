DROP TRIGGER ocr_results_update;
CREATE TRIGGER ocr_results_update AFTER UPDATE OF asset_id, content ON ocr_results
WHEN old.asset_id IS NOT new.asset_id OR old.content IS NOT new.content BEGIN
    INSERT INTO ocr_results_fts(ocr_results_fts, rowid, content) VALUES ('delete', old.asset_id, old.content);
    INSERT INTO ocr_results_fts(rowid, content) VALUES (new.asset_id, new.content);
    INSERT INTO ocr_results_words_fts(ocr_results_words_fts, rowid, content) VALUES ('delete', old.asset_id, old.content);
    INSERT INTO ocr_results_words_fts(rowid, content) VALUES (new.asset_id, new.content);
END;
