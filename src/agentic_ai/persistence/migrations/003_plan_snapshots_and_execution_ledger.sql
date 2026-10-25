-- Migration: 003_plan_snapshots_and_execution_ledger.sql
-- Description: Immutable snapshots of the plan an approval was given for, and a durable
--              per-step execution ledger. Together they let an approved plan be resumed
--              exactly as stored (no re-planning) without ever starting a step twice.
-- Requires PostgreSQL 11+ (EXECUTE FUNCTION in triggers; identity columns). Safe to run again.

-- 1. Plan snapshots -------------------------------------------------------------------
--
-- One row per distinct presentation of a plan: the stored plan and the inputs the guardrails
-- used (all inside `content`), plus the plan fingerprint approvals are bound to and a checksum
-- over everything execution-relevant. Rows are immutable: an UPDATE is refused by a trigger,
-- so "the plan that was approved" cannot be edited in place. A re-presentation of the same
-- plan with different guardrail inputs is a new row (new checksum), never an overwrite.
--
-- ON DELETE RESTRICT: an incident with stored plans cannot be deleted from under them.
--
-- `snapshot_seq` is what defines "the latest" snapshot of a plan. A timestamp cannot: two
-- presentations stored within one clock tick would tie, and which one counted as latest
-- would be arbitrary. That matters, because resume re-runs the guardrails on the latest
-- snapshot's inputs.
CREATE TABLE IF NOT EXISTS incident_plan_snapshots (
    snapshot_seq BIGINT GENERATED ALWAYS AS IDENTITY,
    snapshot_id VARCHAR(64) PRIMARY KEY,
    incident_id VARCHAR(64) NOT NULL REFERENCES incidents(incident_id) ON DELETE RESTRICT,
    plan_fingerprint CHAR(64) NOT NULL,
    snapshot_version INTEGER NOT NULL,
    checksum CHAR(64) NOT NULL,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_incident_plan_snapshots_version CHECK (snapshot_version >= 1),
    CONSTRAINT uq_incident_plan_snapshots_identity UNIQUE (incident_id, plan_fingerprint, checksum)
);

CREATE INDEX IF NOT EXISTS idx_incident_plan_snapshots_lookup
    ON incident_plan_snapshots(incident_id, plan_fingerprint, snapshot_seq DESC);

CREATE OR REPLACE FUNCTION forbid_plan_snapshot_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'incident_plan_snapshots rows are immutable (snapshot %)', OLD.snapshot_id;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_incident_plan_snapshots_immutable ON incident_plan_snapshots;
CREATE TRIGGER trg_incident_plan_snapshots_immutable
    BEFORE UPDATE ON incident_plan_snapshots
    FOR EACH ROW EXECUTE FUNCTION forbid_plan_snapshot_update();

-- 2. Execution ledger -------------------------------------------------------------------
--
-- One row per (incident, plan, step), ever. A caller claims a step by inserting its row as
-- 'started' BEFORE the action runs, then completes it. The uniqueness rule is what prevents a
-- second caller (a resubmission, a resume, a concurrent request) from running the step again.
--
-- A row still 'started' means the action may or may not have happened: the outcome is unknown
-- and the step is never replayed automatically.
--
-- There is deliberately NO foreign key to incidents (unlike approvals and snapshots). An
-- execution has to be recordable before the incident record exists, because a workflow run
-- executes steps and only afterwards saves its record, and a record of something that really
-- happened must survive even if that save fails. An orphaned execution row is harmless; a
-- missing one could cause a duplicate action.
CREATE TABLE IF NOT EXISTS incident_step_executions (
    execution_id VARCHAR(64) PRIMARY KEY,
    incident_id VARCHAR(64) NOT NULL,
    plan_fingerprint CHAR(64) NOT NULL,
    step_index INTEGER NOT NULL,
    step_fingerprint CHAR(64) NOT NULL,
    action_type VARCHAR(64) NOT NULL,
    target_service VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'started',
    source VARCHAR(16) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMPTZ,
    output TEXT,
    error TEXT,
    CONSTRAINT chk_incident_step_executions_index CHECK (step_index >= 0),
    CONSTRAINT chk_incident_step_executions_status CHECK (
        status IN ('started', 'succeeded', 'failed')
    ),
    CONSTRAINT chk_incident_step_executions_source CHECK (source IN ('workflow', 'resume')),
    CONSTRAINT chk_incident_step_executions_lifecycle CHECK (
        (status = 'started' AND finished_at IS NULL)
        OR (status <> 'started' AND finished_at IS NOT NULL)
    ),
    CONSTRAINT uq_incident_step_executions_step UNIQUE (incident_id, plan_fingerprint, step_index)
);

CREATE INDEX IF NOT EXISTS idx_incident_step_executions_incident
    ON incident_step_executions(incident_id, plan_fingerprint);

-- A finished execution is a historical fact and is never rewritten.
CREATE OR REPLACE FUNCTION forbid_finished_execution_update() RETURNS trigger AS $$
BEGIN
    IF OLD.status <> 'started' THEN
        RAISE EXCEPTION 'incident_step_executions row % is already % and cannot change',
            OLD.execution_id, OLD.status;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_incident_step_executions_finished ON incident_step_executions;
CREATE TRIGGER trg_incident_step_executions_finished
    BEFORE UPDATE ON incident_step_executions
    FOR EACH ROW EXECUTE FUNCTION forbid_finished_execution_update();
