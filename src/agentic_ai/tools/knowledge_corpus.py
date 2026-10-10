"""The synthetic engineering knowledge base: a static corpus of past incidents.

Deliberately not a trivial set. A retrieval system tested only on obviously
distinct documents proves nothing - the real test is whether it can tell
apart entries that *sound* similar but have different root causes (several
pairs here share surface vocabulary like "timeout" or "latency" on purpose),
and whether it still ranks a genuinely matching entry above a near-miss.

Covers failure categories beyond the three mock-tool scenarios deliberately,
since the point of this corpus is to be retrieved against *novel* incident
descriptions, not just the ones the project already has fixtures for.
"""

from agentic_ai.domain.knowledge import KnowledgeEntry

KNOWLEDGE_BASE: list[KnowledgeEntry] = [
    KnowledgeEntry(
        id="RB-001",
        title="Database connection pool exhaustion under load",
        symptom_description=(
            "Service returns intermittent errors under moderate-to-high "
            "traffic. Logs show 'failed to acquire database connection' or "
            "'connection pool exhausted'. Error rate rises with request "
            "volume rather than staying constant."
        ),
        root_cause=(
            "Connection pool size was sized for average load, not peak. A "
            "slow query or a burst of concurrent requests holds connections "
            "longer than expected, starving the pool."
        ),
        resolution=(
            "Increase the pool size to match observed peak concurrency, add "
            "a connection acquisition timeout so requests fail fast instead "
            "of queuing indefinitely, and audit for any query holding a "
            "connection open longer than necessary."
        ),
        tags=["database", "connection-pool", "capacity"],
    ),
    KnowledgeEntry(
        id="RB-002",
        title="Upstream dependency failure mistaken for local fault",
        symptom_description=(
            "Service reports elevated error rate and high latency, but its "
            "own CPU, memory, and database metrics are normal. Logs show "
            "timeouts or 5xx responses when calling a specific external API."
        ),
        root_cause=(
            "A downstream dependency degraded or went down. The calling "
            "service was healthy throughout; it was only reporting the "
            "symptom of someone else's outage."
        ),
        resolution=(
            "Confirm the dependency's own status page or metrics before "
            "touching the calling service. If no circuit breaker exists, "
            "add one so a failing dependency degrades gracefully instead of "
            "cascading. No action is needed on the calling service itself."
        ),
        tags=["upstream-dependency", "circuit-breaker", "misdiagnosis"],
    ),
    KnowledgeEntry(
        id="RB-003",
        title="Memory leak causing gradual latency increase and eventual crash",
        symptom_description=(
            "Latency climbs steadily over hours or days with no corresponding "
            "traffic increase, culminating in an out-of-memory crash or "
            "forced restart. Memory usage graphs show a steady upward slope "
            "rather than a spike."
        ),
        root_cause=(
            "An unbounded in-memory cache, or objects retained by a "
            "forgotten event listener, grew without limit until the process "
            "exhausted available memory."
        ),
        resolution=(
            "Add a size or TTL bound to any in-process cache. Use a heap "
            "profiler to confirm what is actually retained before assuming "
            "where the leak is. A restart is a temporary mitigation, not a "
            "fix."
        ),
        tags=["memory-leak", "latency", "crash"],
    ),
    KnowledgeEntry(
        id="RB-004",
        title="Deployment-triggered regression in request latency",
        symptom_description=(
            "Latency or error rate increases sharply at a specific, "
            "identifiable timestamp with no change in traffic volume. The "
            "timestamp correlates with a recent deployment."
        ),
        root_cause=(
            "A code change in the most recent deployment introduced a "
            "performance regression or a bug - commonly a new N+1 query "
            "pattern, a synchronous call added to a hot path, or a "
            "misconfigured feature flag."
        ),
        resolution=(
            "Correlate the onset timestamp with the deployment log first, "
            "before investigating infrastructure. Roll back the deployment "
            "to confirm it is the cause, then bisect the change set."
        ),
        tags=["deployment", "regression", "latency"],
    ),
    KnowledgeEntry(
        id="RB-005",
        title="Thread pool starvation from a slow synchronous call",
        symptom_description=(
            "Request latency increases and eventually requests start timing "
            "out entirely, even though CPU usage is low and the database "
            "looks healthy. The service appears to be doing very little work "
            "despite accepting requests slowly."
        ),
        root_cause=(
            "A blocking synchronous call (often a slow external HTTP call "
            "with no timeout) occupies worker threads for far longer than "
            "expected, so the thread pool fills up and new requests queue "
            "behind it rather than being processed."
        ),
        resolution=(
            "Add an explicit timeout to the blocking call so a thread is "
            "never held indefinitely, and move it off the request thread "
            "pool (async or a dedicated executor) if it is inherently slow."
        ),
        tags=["thread-pool", "latency", "blocking-call"],
    ),
    KnowledgeEntry(
        id="RB-006",
        title="Pod crash loop from a missing or invalid configuration value",
        symptom_description=(
            "Pods repeatedly restart (CrashLoopBackOff) immediately after "
            "deployment, never reaching a ready state. Logs show the process "
            "exiting within seconds of startup, often citing a missing "
            "environment variable or failed config validation."
        ),
        root_cause=(
            "A required configuration value or secret was not set, or was "
            "set incorrectly, for the new deployment environment. The "
            "application fails its own startup validation rather than "
            "running in a broken state."
        ),
        resolution=(
            "Check the pod's startup logs for the specific validation error "
            "before assuming an infrastructure problem - the crash is "
            "usually the application correctly refusing to start unsafely. "
            "Correct the configuration and redeploy."
        ),
        tags=["kubernetes", "crash-loop", "configuration"],
    ),
    KnowledgeEntry(
        id="RB-007",
        title="Pod crash loop from insufficient memory limits",
        symptom_description=(
            "Pods restart repeatedly under load but run fine at startup and "
            "under light traffic. Logs show the process being killed "
            "abruptly with no application-level error, and the pod's last "
            "termination reason is OOMKilled."
        ),
        root_cause=(
            "The container's memory limit was set below what the "
            "application actually needs under real traffic, so the "
            "container runtime kills it once it exceeds that ceiling."
        ),
        resolution=(
            "Distinguish this from RB-006 by checking the termination "
            "reason: OOMKilled points at resource limits, not "
            "configuration. Raise the memory limit to match observed usage "
            "under load, and investigate separately if usage itself seems "
            "abnormally high."
        ),
        tags=["kubernetes", "crash-loop", "memory", "resource-limits"],
    ),
    KnowledgeEntry(
        id="RB-008",
        title="Cache stampede after a cold cache or mass invalidation",
        symptom_description=(
            "A brief, severe spike in database load and latency immediately "
            "following a cache flush, a cache node restart, or a deployment "
            "that happens to invalidate many cache keys at once. Recovers on "
            "its own after a short period."
        ),
        root_cause=(
            "With the cache cold or partially invalidated, many concurrent "
            "requests for the same uncached keys all fall through to the "
            "database simultaneously, multiplying load far beyond steady "
            "state."
        ),
        resolution=(
            "Add request coalescing or a lock around cache-fill logic so "
            "only one request repopulates a given key while others wait, "
            "and consider warming the cache before a planned restart or "
            "deployment."
        ),
        tags=["cache", "database", "spike"],
    ),
    KnowledgeEntry(
        id="RB-009",
        title="Elevated error rate from a third-party rate limit, not an outage",
        symptom_description=(
            "Calls to an external API begin failing with 4xx errors rather "
            "than 5xx or timeouts, typically HTTP 429. The external "
            "service's own status page shows no incident."
        ),
        root_cause=(
            "Request volume to the third party exceeded its rate limit, "
            "often because of a traffic spike, a retry storm amplifying the "
            "original request volume, or a newly added caller the limit was "
            "never raised for."
        ),
        resolution=(
            "Implement backoff that respects the provider's documented "
            "retry-after behaviour rather than retrying immediately, which "
            "only worsens the rate limiting. Request a limit increase if "
            "the volume is legitimate and sustained."
        ),
        tags=["rate-limit", "third-party", "upstream-dependency"],
    ),
    KnowledgeEntry(
        id="RB-010",
        title="Clock skew causing authentication failures across services",
        symptom_description=(
            "Requests between internal services begin failing authentication "
            "with no credential or configuration change. Errors mention "
            "token expiry or an invalid signature timestamp, and the "
            "failures correlate with specific hosts rather than being "
            "global."
        ),
        root_cause=(
            "One or more hosts' system clocks drifted out of sync with the "
            "rest of the fleet (often after an NTP sync failure), so "
            "time-sensitive tokens are generated or validated against the "
            "wrong timestamp window."
        ),
        resolution=(
            "Check system time on the affected hosts against a reliable "
            "source. Restart the NTP service or replace the host if "
            "synchronisation does not recover. Consider wider time-skew "
            "tolerance in token validation if this is a recurring risk."
        ),
        tags=["authentication", "clock-skew", "infrastructure"],
    ),
    KnowledgeEntry(
        id="RB-011",
        title="Disk space exhaustion from unrotated logs",
        symptom_description=(
            "A host or pod becomes unresponsive or starts failing writes "
            "with no clear application error. Available disk space is at or "
            "near zero, and log files are unusually large."
        ),
        root_cause=(
            "Log rotation was missing or misconfigured, so verbose logging "
            "(often debug-level logging left on, or a logging loop from a "
            "repeated error) filled the disk over time."
        ),
        resolution=(
            "Free space immediately by rotating or truncating the offending "
            "logs, then fix the rotation policy and confirm log level is "
            "appropriate for the environment."
        ),
        tags=["disk-space", "logging", "infrastructure"],
    ),
    KnowledgeEntry(
        id="RB-012",
        title="Database replica lag causing stale-read errors downstream",
        symptom_description=(
            "Reads immediately following a write occasionally return stale "
            "or missing data, inconsistently and only under load. The "
            "primary database itself reports healthy write performance."
        ),
        root_cause=(
            "Reads were served from a replica that had fallen behind the "
            "primary under write load, so a read-after-write briefly saw "
            "the pre-write state."
        ),
        resolution=(
            "Route read-after-write paths to the primary, or to a replica "
            "only once confirmed caught up, and monitor replica lag "
            "directly rather than inferring it from symptoms."
        ),
        tags=["database", "replication", "consistency"],
    ),
    KnowledgeEntry(
        id="RB-013",
        title="DNS resolution failure causing intermittent connection errors",
        symptom_description=(
            "Outbound calls to a dependency fail intermittently with "
            "connection or name-resolution errors, not timeouts, and the "
            "target service itself reports no issue. Failures are sporadic "
            "rather than total."
        ),
        root_cause=(
            "A DNS resolver was intermittently failing or returning stale "
            "records - often a resolver cache serving decommissioned IPs "
            "after a dependency's infrastructure changed."
        ),
        resolution=(
            "Confirm with a direct DNS query whether resolution is actually "
            "failing before suspecting the dependency itself. Lower DNS TTL "
            "for frequently-changing endpoints and ensure the resolver cache "
            "is actually honouring it."
        ),
        tags=["dns", "networking", "upstream-dependency"],
    ),
    KnowledgeEntry(
        id="RB-014",
        title="Queue backlog growth from a slow or failing consumer",
        symptom_description=(
            "A message queue's depth grows continuously rather than staying "
            "roughly flat, and downstream processing (e.g. emails, "
            "notifications, async jobs) becomes delayed or stops entirely, "
            "while the producer side reports no issue."
        ),
        root_cause=(
            "The consumer slowed down or stopped processing - commonly from "
            "a downstream dependency it calls becoming slow, an unhandled "
            "exception causing it to crash-loop, or a dead-lettering "
            "misconfiguration silently dropping work without alerting."
        ),
        resolution=(
            "Check consumer health and logs, not the queue infrastructure "
            "itself, first - queue depth growth is almost always a consumer "
            "symptom. Scale or fix the consumer, and add alerting on queue "
            "depth trend rather than only on absolute thresholds."
        ),
        tags=["message-queue", "consumer", "backlog"],
    ),
    KnowledgeEntry(
        id="RB-015",
        title="Elevated latency from garbage collection pauses",
        symptom_description=(
            "p99 latency shows periodic spikes to several seconds at "
            "regular-ish intervals, while p50 latency stays normal and CPU "
            "usage shows brief sawtooth spikes. No single slow endpoint "
            "explains it - the spikes affect requests broadly."
        ),
        root_cause=(
            "Garbage collection pauses (particularly full/major collections "
            "under memory pressure) stop request processing fleet-wide for "
            "the pause duration."
        ),
        resolution=(
            "Confirm via GC logs or runtime metrics before assuming an "
            "application bug. Tune heap size and GC settings for the "
            "workload, or reduce allocation pressure (fewer short-lived "
            "large objects) if pauses remain too frequent."
        ),
        tags=["garbage-collection", "latency", "jvm"],
    ),
    KnowledgeEntry(
        id="RB-016",
        title="TLS certificate expiry causing sudden total outage",
        symptom_description=(
            "A service or endpoint that was working normally suddenly "
            "fails 100% of requests at an exact moment, with clients "
            "reporting certificate validation errors rather than connection "
            "refused or timeout."
        ),
        root_cause=(
            "A TLS certificate expired. Because expiry is a hard cutover at "
            "an exact timestamp rather than a gradual degradation, the "
            "failure is total and instantaneous rather than following the "
            "usual gradual-onset pattern of capacity or dependency issues."
        ),
        resolution=(
            "Renew the certificate immediately. Treat the suddenness and "
            "totality of the failure as the diagnostic signal pointing "
            "straight at certificates rather than capacity, since almost no "
            "other failure mode produces an instant 0-to-100% outage."
        ),
        tags=["tls", "certificate", "outage"],
    ),
    KnowledgeEntry(
        id="RB-017",
        title="Retry storm amplifying an initially minor degradation",
        symptom_description=(
            "A brief, modest increase in latency or error rate rapidly "
            "escalates into a full outage within minutes, far out of "
            "proportion to the triggering event. Request volume as measured "
            "at a load balancer is far higher than actual client traffic."
        ),
        root_cause=(
            "Clients (or an upstream service) retried failed or slow "
            "requests aggressively with no backoff, multiplying load on an "
            "already-struggling service and turning a minor issue into a "
            "self-inflicted outage."
        ),
        resolution=(
            "Distinguish real traffic growth from retry amplification by "
            "comparing unique request IDs to total requests received. Add "
            "exponential backoff with jitter to retrying clients, and "
            "consider load-shedding on the server side as a stopgap."
        ),
        tags=["retry-storm", "cascading-failure", "outage"],
    ),
    KnowledgeEntry(
        id="RB-018",
        title="Data skew causing one shard or partition to be overloaded",
        symptom_description=(
            "Overall aggregate metrics look fine or only mildly elevated, "
            "but a subset of requests (often tied to one large customer or "
            "one key range) time out consistently while the rest of the "
            "system is healthy."
        ),
        root_cause=(
            "Data or load was not evenly distributed across shards or "
            "partitions - one key, customer, or partition receives far more "
            "traffic or data volume than the others were sized for."
        ),
        resolution=(
            "Look for load concentrated on a specific key range or shard "
            "before assuming a system-wide capacity problem, since "
            "aggregate dashboards can hide this entirely. Rebalance or "
            "re-shard the hot key, or isolate it to its own capacity."
        ),
        tags=["sharding", "data-skew", "hot-key"],
    ),
]
