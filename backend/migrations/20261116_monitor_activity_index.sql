-- Additive PostgreSQL performance migration. Execute outside a transaction.
-- Required for bounded recent-activity reads on an existing jobs history.
-- Applied and checked concurrently during the API economy release.
-- No listing, quota, receipt, payment or subscription records are changed.
CREATE INDEX CONCURRENTLY IF NOT EXISTS monitor_jobs_first_seen
    ON public.monitor_jobs (first_seen);
