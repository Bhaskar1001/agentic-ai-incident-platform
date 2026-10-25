-- Migration: 001_initial_incident_schema.sql
-- Description: Create operational incident persistence tables, constraints, and performance indexes.

-- 1. Incidents Table
CREATE TABLE IF NOT EXISTS incidents (
    incident_id VARCHAR(64) PRIMARY KEY,
    correlation_id VARCHAR(64) NOT NULL,
    scenario VARCHAR(64) NOT NULL,
    incident_description TEXT NOT NULL,
    severity VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    final_outcome VARCHAR(32),
    diagnostic_decision VARCHAR(64),
    diagnostic_trigger VARCHAR(64),
    diagnostic_reason TEXT,
    verification_status VARCHAR(64),
    verification_attempts INTEGER NOT NULL DEFAULT 0,
    verification_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_duration_ms DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMPTZ,
    CONSTRAINT chk_incidents_severity CHECK (
        severity IN ('low', 'medium', 'high', 'critical')
    ),
    CONSTRAINT chk_incidents_status CHECK (
        status IN ('detected', 'investigating', 'mitigation', 'resolved', 'escalated', 'monitored', 'closed')
    ),
    CONSTRAINT chk_incidents_final_outcome CHECK (
        final_outcome IS NULL OR final_outcome IN ('resolve', 'monitor', 'escalate')
    )
);

-- 2. Incident Stages Table (Workflow execution spans)
CREATE TABLE IF NOT EXISTS incident_stages (
    stage_id BIGSERIAL PRIMARY KEY,
    incident_id VARCHAR(64) NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
    stage_name VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    error_category VARCHAR(64),
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- 3. Incident Remediations Table (Remediation plan steps and outputs)
CREATE TABLE IF NOT EXISTS incident_remediations (
    remediation_id BIGSERIAL PRIMARY KEY,
    incident_id VARCHAR(64) NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
    step_order INTEGER NOT NULL,
    action_type VARCHAR(64) NOT NULL,
    target_service VARCHAR(64) NOT NULL,
    disposition VARCHAR(32) NOT NULL,
    execution_status VARCHAR(32) NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    output TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 4. Performance & Query Optimization Indexes
CREATE INDEX IF NOT EXISTS idx_incidents_correlation_id ON incidents(correlation_id);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_created_at_desc ON incidents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_status_created_at ON incidents(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_incident_stages_incident_id ON incident_stages(incident_id);
CREATE INDEX IF NOT EXISTS idx_incident_remediations_incident_step ON incident_remediations(incident_id, step_order ASC);

