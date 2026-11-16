-- Additive PostgreSQL/SQLite index. No data rewrite or permission changes.
-- Apply before rollout; existing tables are not altered by create_all.
CREATE INDEX IF NOT EXISTS full_scan_shared_lookup
    ON full_scans (fingerprint, status, updated_at);
