-- Migration: 002_incident_approvals.sql
-- Description: Persisted, auditable approvals for high-risk remediation steps, and a
--              non-terminal 'awaiting_approval' incident status.

-- 1. A run that escalates while a step still waits for a human is not finished, so it needs a
--    non-terminal status. Re-creating the constraint keeps this safe to run again.
ALTER TABLE incidents DROP CONSTRAINT IF EXISTS chk_incidents_status;
ALTER TABLE incidents ADD CONSTRAINT chk_incidents_status CHECK (
    status IN ('detected', 'investigating', 'mitigation', 'awaiting_approval', 'resolved',
               'escalated', 'monitored', 'closed')
);

-- 2. Approvals.
--
-- An approval is tied to the incident by foreign key, and to *one exact step of one exact plan*
-- by (plan_fingerprint, step_index) plus a fingerprint of the step itself. It deliberately has no
-- foreign key to incident_remediations: those rows are replaced wholesale every time an incident
-- is re-saved, so their ids are not stable. A foreign key there would either cascade-delete the
-- audit trail or block the re-save. The action/target/rationale columns are a snapshot of what
-- the approver was shown, kept independent of the replaceable remediation rows.
--
-- ON DELETE RESTRICT: an incident with approval history cannot be deleted out from under it.
CREATE TABLE IF NOT EXISTS incident_approvals (
    approval_id VARCHAR(64) PRIMARY KEY,
    incident_id VARCHAR(64) NOT NULL REFERENCES incidents(incident_id) ON DELETE RESTRICT,
    plan_fingerprint CHAR(64) NOT NULL,
    step_index INTEGER NOT NULL,
    step_fingerprint CHAR(64) NOT NULL,
    action_type VARCHAR(64) NOT NULL,
    target_service VARCHAR(64) NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMPTZ,
    approver VARCHAR(128),
    decision_reason TEXT,
    CONSTRAINT chk_incident_approvals_step_index CHECK (step_index >= 0),
    CONSTRAINT chk_incident_approvals_status CHECK (
        status IN ('pending', 'approved', 'rejected')
    ),
    -- A decision always records who and when; a pending approval records neither.
    CONSTRAINT chk_incident_approvals_decision CHECK (
        (status = 'pending' AND decided_at IS NULL AND approver IS NULL)
        OR (status <> 'pending' AND decided_at IS NOT NULL AND approver IS NOT NULL)
    ),
    -- One approval per step of a given plan, which is what makes registering them idempotent.
    CONSTRAINT uq_incident_approvals_step UNIQUE (incident_id, plan_fingerprint, step_index)
);

-- 3. Indexes.
CREATE INDEX IF NOT EXISTS idx_incident_approvals_pending
    ON incident_approvals(requested_at, approval_id) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_incident_approvals_incident_status
    ON incident_approvals(incident_id, status);
