-- Drops the unused mark-and-sweep cleanup flag. OCR rows are removed with their assets.
DROP INDEX IF EXISTS ocr_mark_delete_idx;
ALTER TABLE ocr_results DROP COLUMN mark_delete;
