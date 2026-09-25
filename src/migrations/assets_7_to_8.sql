-- Why the latest attempt to scan a folder stopped short, as a fixed value clients can present
-- without parsing scan_error. NULL means no failed attempt is recorded. Version 7 kept only the
-- text, so a stored error becomes a generic failure.
ALTER TABLE library_folders ADD COLUMN scan_outcome TEXT
    CHECK(scan_outcome IN ('unavailable', 'incomplete', 'cancelled', 'failed'));
UPDATE library_folders SET scan_outcome = 'failed' WHERE scan_error IS NOT NULL;
