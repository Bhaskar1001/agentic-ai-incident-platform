# AI Engineering & Incident Resolution Platform — Updates Log

> Single source of truth for this project: what it is, why it exists, what has been built in each
> phase, and the reasoning behind every significant decision.
>
> **Last updated:** 2026-10-10 · **Current phase:** Phase 7 complete — escalating incidents now get
> a synthesized root cause analysis (primary cause, alternatives, cited findings, knowledge-base
> consistency), produced as a single structured-output call rather than a second agent loop. The
> investigation agent from Phase 5 (with the optional knowledge-base retrieval from Phase 6)
> remains the only looping agent in the system; no Supervisor exists yet, since two steps in a fixed
> sequence give it no real decision to make. Phase 7 continuation (more agents / a Supervisor) or
> Phase 8 (memory) next.

---

## 1. Introduction

### What is this project?

An **agentic AI platform for production incident resolution**. When a production service breaks —
elevated error rates, timeouts, crashing pods — this system investigates the incident the way a
human on-call engineer would: it forms a hypothesis, gathers evidence from observability tools
(logs, metrics, pod status), reasons over what it found, proposes a root cause, and recommends (or
with approval, performs) a remediation. It then verifies whether the remediation actually worked.

It is deliberately built as a **production-oriented** system rather than a demo: every component is
designed with failure modes, security boundaries, testability, and human oversight in mind.

### What problem are we solving?

Production incidents are expensive in a specific way: the costly part is usually not the *fix*, it's
the **time between "something is wrong" and "we understand what is wrong."** During that window a
human engineer is doing largely mechanical work — pulling up dashboards, grepping logs, checking
recent deploys, correlating timestamps — before any real judgment is applied.

That investigation phase is:

- **Repetitive** — the same handful of checks for most incident classes.
- **Slow under pressure** — context-switching at 3am, hunting for the right dashboard.
- **Inconsistent** — quality depends on who is on-call and how much of the system they know.
- **Poorly captured** — the reasoning usually lives in a Slack thread and is lost afterward.

### How are we solving it?

An **agent** that performs the investigation loop autonomously, with strict boundaries:

```
Incident reported
      ↓
Assess severity            (LLM, structured output)
      ↓
Investigate                (agent loop: reason → call tool → observe → repeat)
      ↓
Collect Evidence           (raw tool output, preserved verbatim)
      ↓
Produce Findings           (LLM inference over evidence)
      ↓
Recommend action           (LLM proposes; deterministic policy decides)
      ↓
Human approval gate        (for anything risky)
      ↓
Remediate → Verify         (did it actually work? if not, re-investigate)
```

The central design principle, applied everywhere:

> **The LLM controls strategy. Deterministic code controls boundaries.**

The LLM decides *which tool to call*, *what the evidence means*, and *what to recommend*. Plain
Python decides *how many steps are allowed*, *whether an action is permitted*, *what counts as valid
data*, and *when a human must approve*. An LLM is never trusted with a safety decision.

### Why solve it this way?

Because the alternatives are worse:

- **A pure script/runbook** can't handle novel incidents or reason over unstructured log text.
- **A pure chatbot over logs** (naive RAG) can answer questions but can't *act* — it can't decide to
  check metrics after seeing something suspicious in logs.
- **An unconstrained autonomous agent** is unsafe. It could restart production pods based on a
  hallucinated diagnosis.

The agentic-with-guardrails approach keeps the LLM where it's genuinely strong (reasoning over
ambiguous natural-language evidence, planning what to check next) and keeps it away from where it's
weak (anything requiring correctness guarantees, authorization, or irreversible action).

### Learning goal

This project doubles as a structured curriculum in production AI engineering: LangGraph, agent
architecture, RAG, memory, MCP, guardrails, evaluation, observability, and deployment — each
introduced only when a real problem justifies it, never for its own sake.

---

## 2. Tech Stack — what and why

| Technology | Why chosen | Alternatives rejected |
|---|---|---|
| **Python 3.10** | Ecosystem for AI/LLM tooling is Python-first. | — |
| **uv** | Fast dependency manager with a real lockfile (`uv.lock`) for reproducible environments; also manages the Python version per-project. | `pip` + `requirements.txt` (no lockfile → non-reproducible installs); `poetry` (mature but slower, heavier CLI). |
| **Pydantic v2** | Validation *at the boundary*. Used for domain invariants **and** as the schema contract for LLM structured output — one tool for both jobs. | Dataclasses (no validation); manual `if` checks (verbose, easy to forget). |
| **OpenRouter (default) + Ollama (kept)** | Hosted, OpenAI-compatible API with free-tier models — chosen after Ollama's local disk footprint became impractical on the development machine. Sits behind `LLMClient`, an abstraction introduced at this point specifically because the project's own stated "provider-agnostic" intent (see below) had not actually been enforced in code until a second real provider existed to prove it against. | Staying Ollama-only — ruled out by disk space, not by design; the project otherwise has no objection to a local model and keeps Ollama working behind the same interface. |
| **httpx** | Already a transitive dependency via `ollama`; used directly for the OpenRouter HTTP client rather than adding a second HTTP library. | `requests` — no reason to add a second dependency for the same job. |
| **python-dotenv** | Loads `OPENROUTER_API_KEY` from a local, gitignored `.env` file rather than requiring it to be exported in every shell session. | Plain environment variables only — more portable across shells, but easy to forget to set and harder to keep consistent across a team. |
| **sentence-transformers (`all-MiniLM-L6-v2`)** | Local embedding model for knowledge-base retrieval — free, no network dependency once cached, and small enough (~80MB) not to reintroduce the disk-space problem that moved the project off Ollama, which was about multi-GB LLM checkpoints specifically. | A hosted embedding API — rejected to avoid a second paid-API dependency beyond OpenRouter for a corpus this small. |
| **numpy** | Brute-force cosine similarity over ~20 embedded documents. | A vector database (Chroma/pgvector) — rejected per the project's own "no infrastructure until a concrete problem demands it" principle; neither scale nor persistence-across-restarts is a real problem yet for a static ~20-document corpus. |
| **LangGraph** | Explicit, inspectable **state** across steps; **cycles** (agent loops back to gather more evidence); **interrupts** for human-in-the-loop approval. | LangChain chains alone — fundamentally a linear pipeline; awkward for loops and stateful branching. Plain Python — would work, but we'd hand-roll state management, checkpointing, and interrupts. |
| **pytest** | Standard, minimal ceremony. Dev-only dependency so it never ships to production. | — |
| **`src/` layout** | Package is only importable if properly installed, so packaging bugs surface immediately instead of being masked by `sys.path` accidents. | Flat layout — "works on my machine" risk. |

### Planned but deliberately not yet added

`PostgreSQL` (persistence), `pgvector` (RAG), `FastAPI` (API layer), `MCP` (tool protocol), `Docker`
/ `Kubernetes` / `AWS` / `Terraform` (deployment). **Each will be added only when a concrete need
arises.** Adding infrastructure before there's a problem it solves is the most common way these
projects collapse under their own weight.

---

## 3. Architecture

```
src/agentic_ai/
├── domain/       Business entities, their invariants, and safety policy
│                (Incident lifecycle, Evidence/Finding, escalation thresholds)
├── llm/          Stateless LLM capabilities (one call in, structured output out)
├── tools/        The agent's perceptual world — what it can observe
├── agents/       Autonomous reasoning: the bounded investigation loop
├── workflows/    LangGraph orchestration: state, nodes, edges, routing
└── api/          (empty) HTTP layer — later phase
```

**Why these boundaries:**

- `llm/` vs `agents/` — `assess_severity()` is a *capability*: one call, no loop, no decisions about
  what to do next. An **agent** has a loop, state it reasons over, and chooses its own next action.
  A capability is something an agent *uses*.
- `workflows/` vs `agents/` — `workflows/` owns graph wiring and deterministic orchestration;
  `agents/` owns autonomous decision-making. The boundary held up when the two were connected: the
  workflow's `investigate_node` *calls* `investigate()` rather than embedding the agent's graph, so
  each side keeps its own state shape and neither leaks into the other. ⚠️ **Still provisional** — in
  Phase 7 a Supervisor agent will itself likely be a LangGraph graph, which may force this line to be
  redrawn.
- `domain/` stays free of LLM and framework imports — business rules must be testable without a
  model or a graph.

---

## 4. Phase Log

### Phase 1 — Engineering Foundations ✅

**Goal:** A project skeleton that won't need rebuilding later.

**Built:** `uv`-managed project · `src/` layout with `domain`/`llm`/`tools`/`workflows`/`agents`/`api`
subpackages · `tests/` package · git repository · public GitHub remote.

**Key decisions:**

- **`uv` over `pip`/`poetry`** — the deciding factor was the **lockfile**. `requirements.txt` records
  *what you asked for* (`langchain>=0.2.0`); a lockfile records *what was actually resolved*, pinned
  exactly, including every transitive dependency. Without it, two installs of the same file can
  produce different environments. That matters more here than in ordinary software, because an
  agentic system already has non-determinism from the model — the dependency layer must not add more.
- **`src/` layout** — with a flat layout, Python often adds the working directory to `sys.path`, so
  tests can import a package that was never properly installed. That masks real packaging bugs until
  deployment. `src/` makes correct installation the only way to import.
- **`[project.dependencies]` vs `[dependency-groups].dev`** — `pytest` is dev-only; it must never be
  installed into a production container. Established this split at the first opportunity.

**Commits:** `ea9b554` Initialize agentic AI project structure

---

### Phase 2 — Incident Domain Model ✅

**Goal:** Define what an incident *is*, structurally, before anything reasons about one.

**Built:** `src/agentic_ai/domain/incident.py` — `Severity` enum, `IncidentStatus` enum, `Incident`
model with an enforced lifecycle. `tests/test_incident.py` — 11 tests.

**Key decisions:**

- **Enums, not strings, for `Severity` and `IncidentStatus`.** A free-text severity produces
  `"critical"` / `"Critical"` / `"CRITICAL"` chaos, and downstream a guardrail asking
  *"is this CRITICAL?"* needs a reliable answer. Both inherit from `(str, Enum)` so they serialise
  cleanly to JSON later. **This is deterministic validation's job, not the LLM's.**
- **A lifecycle with illegal transitions.** Not every status change is valid:

  ```
  DETECTED → INVESTIGATING → MITIGATION → RESOLVED → CLOSED
                    ↑                         │
                    └─────── reopen ──────────┘
  ```

  `DETECTED → CLOSED` is forbidden (skips investigation). `RESOLVED → INVESTIGATING` is *allowed* —
  incidents genuinely recur, and a system that can't reopen one is lying about reality. `CLOSED` is
  terminal (empty transition set).
- **The entity guards its own invariant.** Status changes go through `transition_to()`, which
  validates against a class-level `_LEGAL_TRANSITIONS` map and updates `updated_at` only on success.
  This is the "protect invariants inside the entity" principle from domain-driven design: an
  `Incident` should never be *able* to hold an invalid state, rather than merely *asking* callers to
  behave.
- **`status` is not a public field.** It's stored as a `PrivateAttr` (`_status`) and exposed through
  a read-only `@property`. A `model_validator(mode="before")` intercepts the constructor input and
  routes `status=...` into `_status`, so construction still works normally.

  *Why not `Field(frozen=True)`?* Tried it — it blocks **all** assignment including `transition_to()`'s
  own internal mutation. Pydantic can't distinguish internal from external callers. The property
  approach can.

  *Honest limitation:* `incident._status = X` still works. Python's underscore convention is social,
  not enforced. There is a test that **documents this explicitly** rather than pretending the
  encapsulation is absolute.
- **Timezone-aware timestamps** (`datetime.now(timezone.utc)`), asserted in tests. Naive datetimes
  are a silent-bug factory once incidents are compared or stored across systems.
- **`Evidence` / `TimelineEntry` deliberately excluded from `Incident`.** They reference the incident
  by ID rather than being nested inside it, because an incident can accumulate hundreds of evidence
  records over its life and most reads (list view, status check) need none of them. Loading them
  eagerly would be the classic ORM N+1 / over-fetching problem, designed in from the start.

**Bugs found and fixed during review:**

1. Raw `KeyError` escaping `transition_to()` on an unmapped status — leaked an implementation detail
   (a dict lookup) as the error contract. A caller writing `except ValueError:` would not catch it.
   Replaced with a defensive `.get()` and an explicit `ValueError`.
2. The `_LEGAL_TRANSITIONS` dict was being **rebuilt on every call** inside the method body. Moved to
   a class-level constant — built once, not once per transition.
3. `status` was directly assignable, bypassing the entire invariant. Fixed via the `PrivateAttr` +
   property approach above.

**Commits:** `1676d8e` Implement incident domain model

---

### Phase 3 — LLM Foundation ✅

**Goal:** Get **structured, validated** output from an LLM — not free text.

**Built:** `src/agentic_ai/llm/severity.py` — `SeverityAssessment` model, `assess_severity()`,
custom exceptions. `tests/test_severity.py` — 4 tests including a live model call.

**Environment:** Ollama v0.33.3 installed locally; `llama3.2:latest` (2.0 GB) pulled.

**Key decisions:**

- **Why structured output instead of parsing free text.** Asking *"what severity is this?"* and
  regex-matching the reply fails in at least three ways:
  1. The reply mentions several severities (*"initially looked high, now critical"*) — regex grabs
     the wrong one.
  2. The model uses different words — *"highest priority"*, *"crit"* — and the pattern misses.
  3. **The reply contains no severity at all.** A regex can't express *"I don't know"*; it either
     matches something wrong or matches nothing, silently. Structured output makes invalid or absent
     answers **detectable** rather than swallowed.
- **Pydantic as the LLM contract.** `SeverityAssessment.model_json_schema()` is passed straight to
  Ollama's `format=` parameter, and the response goes through `model_validate_json()`. The same tool
  that guards the domain model guards the model boundary. Notably, `severity` reuses the *existing*
  `Severity` enum — the LLM cannot return a severity the domain doesn't recognise.
- **Retry once, then fail loudly — never default.** On `ValidationError`, retry a second time; if it
  fails again, raise. **Do not silently substitute a default severity.** A `MEDIUM` invented because
  the model hiccuped could mean nobody gets paged for a genuine `CRITICAL`. Escalating to a caller
  who can involve a human is correct; guessing is not.
- **Two failure categories, deliberately kept separate.**

  ```
  Invalid LLM output    → retry once → InvalidAssessmentError   (data-quality problem)
  Ollama unreachable    → propagates untouched                  (infrastructure problem)
  ```

  These warrant different upstream handling: infrastructure failures might be retried with backoff;
  repeated invalid output from a *reachable* model signals a bad prompt or an undersized model, which
  retrying won't fix. `except ValidationError` deliberately does **not** catch connection errors.
- **`raise InvalidAssessmentError(...) from exc`** — preserves the original `ValidationError` as
  `__cause__`, so field-level validation detail is never lost. A test asserts this explicitly.

**Bug found and fixed:** an unreachable `raise RuntimeError(...)` after the retry loop — every path
through the loop either returns or raises, so it was dead code. Removed. (Contrast with Phase 5's
router guard, which *looks* similar but is genuinely reachable via direct calls — see below.)

**Testing technique established:** `monkeypatch` `ollama.chat` with a fake returning known-bad data,
then assert the call count. This proves the retry actually fires **twice** without needing a real
model to cooperate in failing — deterministic and fast. Reused in every later phase.

**Commits:** `e40961f` Add structured LLM severity assessment

---

### Phase 4 — LangGraph Workflow ✅

**Goal:** First stateful graph — state, nodes, conditional routing.

**Built:** `src/agentic_ai/workflows/incident.py` — `IncidentWorkflowState`, `assess_severity_node`,
`route_by_severity`, placeholder `escalate_node` / `investigate_node`, `build_incident_workflow()`.
`tests/test_incident_workflow.py` — 3 tests.

```
START → assess_severity → route_by_severity ─┬─ "urgent" → escalate    → END
                                             └─ "normal" → investigate → END
```

**Key decisions:**

- **Why LangGraph and not a LangChain chain.** A chain is a pipeline: the path is known in advance. An
  agent needs to choose its next action based on what it just observed — check logs, or metrics, or
  the recent deploy? — and to **loop** until it has enough. LangGraph provides explicit inspectable
  state, cycles, and interrupt points for human approval (needed in Phase 10). Without those, all
  three would be hand-rolled.
- **The graph state does *not* contain the `Incident` entity.** It holds `incident_description` and
  the assessment; later it will hold `incident_id`. Two reasons: (1) orchestration state and domain
  entities have different lifecycles and shouldn't be coupled; (2) LangGraph merges partial state
  updates between nodes, and the `PrivateAttr` + property encapsulation built in Phase 2 should not
  be run through serialisation machinery whose behaviour we haven't verified. **Reference domain
  objects; don't embed them.**
- **`path_map` indirection.** `route_by_severity` returns abstract keys — `"urgent"` / `"normal"` —
  and `path_map` maps those to node names. The routing function therefore doesn't know the graph's
  topology: nodes can be renamed, or the same routing logic reused in a different graph, without
  touching the routing logic. Arguably overkill at this size; used deliberately to establish the
  pattern.
- **Verified the API rather than assuming it.** `inspect.signature(StateGraph.add_conditional_edges)`
  showed `path: Callable[..., Hashable]` and `path_map: dict[Hashable, str] | list[str] | None` —
  confirming the routing function returns a *routing key*, not a business value.
- **A defensive `ValueError` that was deliberately kept.** `route_by_severity` raises if
  `severity_assessment` is `None`. Tracing the graph shows this is unreachable — `assess_severity_node`
  always runs first and always sets it. But unlike the Phase 3 `RuntimeError`, this function is
  *directly callable*: in a unit test, in a future graph, or after a rewiring. The distinction is
  **"unreachable through this graph today"** vs **"unreachable through any call, ever."** Only the
  second justifies deletion. A direct-call test proves the guard fires.

  A useful side effect: this same guard would also catch a typo in `assess_severity_node`'s return
  key (e.g. `"severity"` instead of `"severity_assessment"`), because the state would arrive
  unpopulated. Defensive code often catches failures it wasn't written for.

- **Graph tests mock the LLM.** Routing logic is tested with `monkeypatch`ed `assess_severity` — fast
  and deterministic. The real model is already covered by the live test in Phase 3; duplicating that
  cost per graph test buys nothing.

**Commits:** `e5ae2e2` Implement LangGraph incident workflow · `a3ca9ba` Update gitignore

---

### Phase 5 — Investigation Agent 🚧 *in progress*

**Goal:** Replace the `investigate` placeholder with a real ReAct-style agent that selects tools,
gathers evidence, and produces a structured conclusion.

#### 5a. Design contract — agreed before writing code ✅

Settled on paper first, because these decisions are cheap to change now and expensive once an agent
loop is built around them.

**Core invariants:**

1. **"Evidence must always be traceable back to an unmodified tool result."** Raw tool output is
   authoritative; the LLM never rewrites it.
2. **"Tool failure is part of the agent's environment, not automatically a terminal workflow
   failure."** A failed tool is an observation the agent reacts to — *"logs unavailable, try
   metrics"* — not a crash.
3. **"A finding can be wrong while the underlying evidence is still correct."** This is the entire
   justification for separating the two.
4. **LLM controls strategy; deterministic code controls boundaries.**

**Planned models:**

```
InvestigationResult                Evidence                    TimelineEntry
├── summary                        ├── id                      ├── timestamp
├── evidence[]                     ├── source_tool             ├── event_type (→ enum)
├── findings[]                     ├── raw_output  ← verbatim  ├── description
├── model_assessed_confidence      ├── observation ← rendered  └── metadata
├── unresolved_questions[]         └── metadata
├── recommended_next_action
└── termination_reason
```

| Concept | Answers | Produced by |
|---|---|---|
| **Evidence** | "What did we observe?" | Tool (value) + LLM (selection/phrasing) |
| **Finding** | "What do we infer from it?" | LLM |
| **TimelineEntry** | "What happened during the investigation?" | Deterministic process record |

**Field classification — deterministic vs LLM vs hybrid:**

| Field | Source | Note |
|---|---|---|
| `summary` | LLM | Must only use information in context |
| `evidence[]` | **Hybrid** | Value is tool-derived and authoritative; selection/phrasing is LLM |
| `findings[]` | LLM | Inference — separable from evidence, and separately wrong |
| `model_assessed_confidence` | **Hybrid** | LLM proposes, code validates range |
| `unresolved_questions[]` | LLM | Gaps in current evidence |
| `recommended_next_action` | **Hybrid** | LLM proposes, policy engine decides if permitted |

**Named `model_assessed_confidence`, not `confidence`, on purpose.** LLMs produce plausible-sounding
numbers with no calibrated relationship to accuracy. The field name itself is the guardrail against a
future reader treating `0.92` as a 92% probability of correctness. Whether it correlates with real
accuracy is an open question for Phase 13 (evaluation).

**Termination:**

```python
MAX_INVESTIGATION_STEPS = 5   # one step = reason → select tool → execute → observe
```

One limit, not two. `MAX_ITERATIONS` and `MAX_TOOL_CALLS` were both considered and collapsed — with
one tool call per step they are the same bound under two names, and redundant limits can silently
disagree.

```
termination_reason:
  COMPLETED          agent decided it had enough evidence
  MAX_STEPS_REACHED  system stopped it at the boundary
  TIMEOUT            ⚠️ reserved — NOT reachable yet, no wall-clock enforcement exists
  TOOL_ERROR         unrecoverable only — a single failed tool must NOT trigger this
```

Kept **separate from confidence** on purpose: `confidence=0.60` does not mean "incomplete", and
`MAX_STEPS_REACHED` does not mean "low quality". Conflating them would let a truncated investigation
masquerade as a confident answer.

**Timeline persistence:** written incrementally *during* execution (not reconstructed afterwards),
held in in-memory state for this phase. **Architectural boundary: the agent must never gain a direct
database dependency.** When persistence arrives, it goes
`Agent → state/event → repository layer → PostgreSQL`. The agent investigates; it does not store.

**Deliberately deferred (recorded, not forgotten):** `Finding.supporting_evidence_ids` — linking each
finding to the specific evidence supporting it. Premature now; important later for RCA, human review,
auditability, and evaluation.

#### 5b. Mock tool layer ✅

**Built:** `src/agentic_ai/tools/mock_tools.py` — `get_mock_logs()`, `get_mock_metrics()`,
`get_mock_pod_status()`, `MockService` enum. `tests/test_mock_tools.py` — 14 tests.

**Why tools before the agent:** these tools are the agent's **entire perceptual world**. Everything it
will ever know arrives through these signatures. Sloppy tool contracts get inherited by every
downstream decision, and you end up debugging your *tools* while believing you're debugging your
*agent*.

**Verified first, not assumed:** a direct probe confirmed `llama3.2` genuinely honours Ollama's
native `tools=` parameter — it selected `get_mock_logs` and extracted `{"service": "database"}` from
plain English. No fallback design needed. (Caveat: that proved *one tool, one obvious call, one
turn*. Reliable multi-tool selection and knowing when to **stop** are still unproven and remain the
real risk with a small local model.)

**Key decisions:**

- **Return `str` containing JSON.** This resolves a genuine tension: `Evidence.raw_output` needs the
  output preserved verbatim; the LLM needs to read it as tool-result text; tests need deterministic
  comparison; and later code may need to parse it. A JSON-formatted string satisfies all four rather
  than compromising between them.
- **Three distinct failure categories:**

  ```
  Valid request      → success result
  Unknown service    → structured error result   ← agent can react and recover
  Programming bug    → exception
  ```

  Collapsing the middle case into an exception is the common mistake — it leaves the agent unable to
  distinguish *"this data doesn't exist"* from *"the system is broken."*
- **A single coherent incident scenario.** All three tools describe **one** story: the `payment`
  service is suffering **database connection-pool exhaustion**, causing connection timeouts, elevated
  latency and error rates, and unhealthy restarting pods. Metrics show `active == pool_size == 100`;
  logs show *"connection pool exhausted"* and timeout errors; pods show `ready: False` with restart
  counts of 4–5.

  Coherence is the point. If metrics said "connection exhaustion", logs said "timeout", and pods said
  "memory leak", we'd have three unrelated facts rather than a test case — and no way to evaluate
  whether the agent correctly *connected* evidence into a hypothesis.
- **Realistic fixture values.** Latency percentiles have a proper long tail (`p50: 850`, `p95: 2400`,
  `p99: 5100`); error rate is a plausible `12.5%`, not a cartoonish 100%. Agents tuned on unrealistic
  data only work on unrealistic data.
- **Docstrings are written for an LLM reader.** The model sees only the name, docstring, and
  parameter schema — that *is* the contract. Validation logic inside the function body is invisible
  to the caller.

**Issue found and fixed in review — the contract discoverability gap:**

The original code had `if service != "payment"` inside the function, while the docstring said only
*"Name of the service."* The LLM cannot see that comparison. Given an incident described as
*"Payment API is failing"*, it would reasonably guess `"payment-api"`, `"payment_service"`, or
`"payments"` — all silently returning `service_not_found`. The agent would burn three of its five
steps and conclude "no data available", looking stupid when the real fault was an underspecified
contract.

> **General principle:** anything the LLM must know to call a tool correctly has to be visible in the
> tool's contract — name, docstring, or parameter schema. Nothing else exists as far as the model is
> concerned.

Fixed with **two complementary mechanisms**, because they cover different moments:

1. **`MockService` enum + `AVAILABLE_SERVICES`** — one source of truth for valid values, ready to
   emit `"enum": ["payment"]` into the tool's JSON schema. *Prevents* the wrong guess.
2. **Self-teaching error response** — the error now includes `available_services` as structured data
   *and* names them in the message. *Recovers* from a wrong guess in one step. This is invariant #2
   applied directly: tool failure as a recoverable observation.

The parameter is **deliberately still typed `service: str`, not `service: MockService`.** A real
`get_logs()` querying a live cluster cannot have valid services baked into a type — the set is
dynamic. Typing it as the enum would make the mock behave *unlike* its eventual replacement. The
enum constrains the data; the signature stays honest about the real contract.

**Other review outcomes:**

- **Duplicated guard extracted** to `_is_known_service()`. Not for the three saved lines — so the
  *rule* has one home. Adding a service now means editing `MockService` only.
- **Scenario-pinning tests added.** The original tests asserted JSON shape (`status`, `service`) but
  not the incident-defining facts. Editing `utilization_percent` from `100` to `12` would have passed
  every test while silently destroying the scenario. New tests assert *relationships*
  (`active == pool_size`) rather than literals, so tuning numbers doesn't break tests spuriously but
  changing the **story** does.
- **Determinism tests left as-is — a conscious "no".** `first == second` only proves idempotence
  within one process, not true determinism. With hardcoded literals and no computation, time, I/O, or
  randomness, there is no mechanism by which output could vary, so golden-file assertions would add
  maintenance cost for no additional protection. **Revisit if these tools ever gain logic.**

**Open minor item:** `AVAILABLE_SERVICES` is a module-level mutable `list`; a `tuple` would make
accidental mutation structurally impossible.

#### 5c. Investigation agent ✅

**Built:** `src/agentic_ai/domain/investigation.py` (the models) · `src/agentic_ai/tools/registry.py`
(tool registry) · `src/agentic_ai/agents/investigation.py` (the agent subgraph).
`tests/test_investigation.py` — 14 tests.

```
START → reason ─┬─ "continue"         → execute_tools ──┐
                ├─ "stop"             → END             │
                └─ "budget_exhausted" → budget_exhausted → END
                       ↑                                 │
                       └── the cycle ────────────────────┘
```

**Key decisions:**

- **Implemented as a LangGraph subgraph, not a plain `while` loop.** A plain loop would have been
  simpler to write and debug, but Phase 10 needs **interrupts** for human approval mid-investigation,
  and LangGraph provides those plus checkpointing. Going straight to the graph avoids a rewrite.
- **`Annotated[list, _append]` reducers on `messages`, `evidence` and `timeline`.** Without a reducer,
  each node's return value *overwrites* that state key. The reducer makes it *accumulate* — which is
  what makes a cycle actually build up context rather than discarding it each pass. This is the
  mechanism that turns a graph into a loop with memory.
- **The step budget is enforced by the graph, not the model.** `should_continue()` checks
  `steps_taken >= MAX_INVESTIGATION_STEPS` **before** honouring the model's wish to continue. A model
  that never stops is stopped anyway, and lands in `budget_exhausted_node`, which records
  `MAX_STEPS_REACHED`. Directly implements "LLM controls strategy, code controls boundaries."
- **The summariser strips code-owned fields from the JSON schema.** Before the final structured call,
  `evidence`, `timeline` and `termination_reason` are removed from the schema handed to the model,
  then injected afterwards by deterministic code. The model **physically cannot author them** — this
  is the "raw output is authoritative" invariant enforced structurally rather than by instruction.
- **Tool failures are converted, never raised.** `ToolNotFoundError` (hallucinated tool) and
  `TypeError` (bad arguments) are both caught and turned into error-shaped tool *results* the model
  reads and reacts to, with a `TOOL_FAILED` timeline entry. Invariant #2 made real. Note the failed
  call produces **no Evidence** — only successful observations become evidence.
- **One registry, two consumers.** `tools/registry.py` produces both the JSON schema sent to the model
  and the dispatch target, from the same dict. They cannot drift. The schema emits
  `"enum": ["payment"]` derived from `AVAILABLE_SERVICES`, so the Phase 5b contract fix actually
  reaches the model.

**Live result — first run (poor):**

The graph worked; the model did not. It called **one** tool, ignored metrics and pod status, and
produced `summary: "Incident investigation"`, `confidence: 0.0`, and **zero findings** — despite
having logs in front of it that said *"connection pool exhausted"*.

One thing did go right: it recommended *"Escalate to human"* with confidence `0.0` rather than
fabricating a diagnosis. The **"never silently default"** principle held under a weak model.

**Diagnosis and fix — prompt engineering:**

Rather than immediately reaching for a bigger model, the cheap diagnostic was run first: is this a
*prompting* problem or a *capability ceiling*? Two prompts were rewritten:

1. **System prompt** — added the *reason* one source is insufficient (logs show symptoms, metrics
   show magnitude, pod status shows instance health), an explicit numbered procedure naming all three
   tools, and explicit negatives: *"do NOT stop after a single tool call"*, *"do NOT repeat a tool
   call"*.
2. **Summary instruction** — was merely *"summarise the investigation as JSON"*. Rewritten to specify
   what each field should contain and to require every finding to cite a concrete signal (an exact
   metric value or log message).

**Live result — after (good):**

| | Before | After |
|---|---|---|
| Tools called | 1 (logs only) | **3** — logs, metrics, pod status |
| Summary | `"Incident investigation"` | Correct root cause, cites 100% utilisation |
| Findings | 0 | **4**, each naming its supporting signal |
| Confidence | 0.0 | 0.8 |
| Next action | "Escalate to human" | "Review and adjust connection pool configuration" |

The model correctly identified **database connection pool exhaustion** — the exact scenario the
fixtures encode — and corroborated across sources rather than parroting one: finding 1 from metrics,
findings 2–3 from logs, finding 4 combining both.

Two details worth recording:

- **The confidence gradient was sensible.** `0.9` for a directly-observed metric, `0.8` for an
  explicit log message, `0.7` for an inference *from* logs, `0.6` for a causal claim. Not calibrated
  probability — the `model_assessed_confidence` name still earns its keep — but the *ordering* tracks
  each claim's distance from raw evidence.
- **The unresolved questions were genuine.** *"Is it a configuration problem or a traffic surge?"* is
  exactly what the available tools cannot answer. It identified the boundary of its own evidence
  instead of speculating past it.

> **Transferable lesson: small models need *procedural* instruction, not *goal-level* direction.**
> "Investigate the incident" is a goal; "call these three tools in order, then cite the specific
> values you saw" is a procedure. Larger models bridge that gap themselves; a 3B model does not.
> **Try proceduralising the prompt before reaching for a bigger model.**

Assessment: the **summary instruction was probably the bigger lever** — the original gave the model
no idea what a good summary or finding looks like, which is why it produced filler text and an empty
findings list.

#### 5d. Second scenario, and enforcing evidence coverage ✅

**Goal:** find out whether the agent *reasons* or merely *pattern-matches*. One validated scenario
proves nothing — the prompt at that point literally listed the three tools in order, so the model may
simply have been following a script toward a memorised answer.

**Built:** a second scenario — **upstream dependency failure** — where the payment service is
*healthy* and a downstream API (`card-authorization-api`) is returning 503s with 10-second timeouts.
A second, *healthy* dependency (`ledger-api`) sits alongside it, so the agent must isolate which one.

| Signal | Pool exhaustion | Upstream failure |
|---|---|---|
| DB connections | 100/100 (**saturated**) | 12/100 (**healthy**) |
| Pods | 2 unready, 4–5 restarts | **all ready, 0 restarts** |
| Latency p50 | 850 ms (degraded) | **60 ms (fine)** |
| Latency p95 | 2 400 ms | 10 200 ms (**upstream timeout**) |
| Logs | "connection pool exhausted" | "503 from card-authorization-api" |

An agent that pattern-matches the first scenario will reach for pool exhaustion. The evidence here
actively contradicts that.

**Fixture design decision — an explicit `scenario` parameter, not module-level state.** A
`set_scenario()` switch would have been more convenient, but it makes the tools return different data
depending on invisible state: no longer pure functions of their arguments, and tests must remember to
reset between runs. An explicit parameter with a default keeps determinism intact.

Critically, **the model never sees it.** `build_tool_schemas()` advertises only `service`, and
`dispatch_tool()` strips any `scenario` key the model might invent before injecting the caller's
value. The agent cannot choose its own reality — the same structural-not-instructional principle as
stripping `evidence` from the summary schema.

**Result — the agent does not pattern-match.** On the upstream scenario it correctly identified the
fault as external and named the right dependency, with this line in its findings:

> *"The local database query completed normally, suggesting that the issue is not with the database."*

It actively ruled out the *other* scenario's root cause rather than defaulting to it.

**But it called only one tool** — logs — and concluded from that alone, reporting findings at
confidence `1.0`, `1.0`, `1.0`, including a summary claim that *"the upstream service has a high
error rate"*, a fact that lives in the metrics it never retrieved. **Right answer, incomplete
evidence, false confidence** — a worse failure mode than being wrong loudly, because nothing
distinguishes it from the case where the guess is wrong.

**The regression run changed the diagnosis.** Scenario 1, run with the *same* generalised prompt,
called all three tools — and in a different order than the old numbered list. So this was never "the
prompt broke compliance":

> **Compliance is non-deterministic.** Same prompt, same model, different scenario, different
> behaviour. You cannot instruct your way out of non-determinism; you can only bound it in code.

**Fix — prompt *and* deterministic enforcement:**

1. **Prompt re-enumerates the three tool names**, but explicitly does *not* prescribe an order
   (*"call them in whatever order the evidence suggests, but you must call all three"*), so it does
   not re-overfit to a sequence. Added a rule: *"Only state facts you actually observed in a tool
   result."*
2. **`MIN_DISTINCT_TOOLS = 3`, enforced in the graph.** A premature conclusion routes to a new
   `require_more_evidence` node, which clears the termination, names the specific tools not yet
   called, and re-enters the cycle. **The model's "I'm done" became a request, not a decision** —
   sufficiency of evidence is a boundary, and boundaries belong to code.
3. **Confidence capping.** If an investigation ends with partial coverage, the summariser is told
   which sources are missing, instructed not to state facts from them, and capped at ≤ 0.5.

**A bug introduced by the fix, and worth recording.** The first version produced a
`GraphRecursionError`: `require_more_evidence` cleared `termination_reason` and looped back to
`reason`, but `steps_taken` only advanced inside `execute_tools_node` — so a model that kept claiming
to be done without calling tools cycled forever. **A guard against unbounded model behaviour that was
itself unbounded.** Fixed by charging a rejection one step, which then forced
`MAX_INVESTIGATION_STEPS` from 5 to 8: a stubborn model needs 3 calls + 2 rejections = exactly 5,
leaving no headroom for a failed or hallucinated call. The arithmetic is documented in the constant.

**Live verification — both scenarios now call all three tools**, in *different* orders
(`logs→pods→metrics` and `pods→metrics→logs`), confirming exploration rather than script-following.
Both reach correct, evidence-grounded diagnoses and every cited number is real.

The confidence values became honest too. Scenario 1 assigns `1.0` to four directly-observed facts and
drops to `0.8` for the causal claim *"connection pool exhaustion is the root cause"* — the model
separating observation from inference in its own output, mirroring the Evidence/Finding split the
domain model encodes.

#### 5e. Wiring the agent into the workflow ✅

**Built:** `investigate_node` in `workflows/incident.py` now calls the real agent, and the flow was
reordered.

```
START → assess_severity → investigate ─┬─ urgent → escalate → END
                                       └─ normal → monitor  → END
```

**Key decisions:**

- **Every incident is investigated before routing.** The previous order routed severity straight to
  `escalate`, which meant the *most serious* incidents were the ones nobody investigated — an
  escalation arriving with no evidence attached. Investigation now precedes routing, so escalation
  always carries findings, and the router can see them.
- **A failed or incomplete investigation escalates regardless of severity.**

  ```python
  if investigation is None or not investigation.is_complete:
      return "urgent"
  ```

  An incident nobody could explain is precisely the one a human should look at, even if it first
  appeared minor. This is the first place `is_complete` does real work: `MAX_STEPS_REACHED` now
  changes what the system *does*, not merely what it reports — which is why keeping
  `termination_reason` separate from confidence mattered.
- **Investigation failure does not sink the workflow, but real bugs still surface.**
  `investigate_node` catches `InvestigationError` and records it so routing can continue; a
  `RuntimeError` (Ollama unreachable, say) propagates untouched. Same failure-category discipline as
  Phase 3, with a test asserting the unexpected error is *not* swallowed.
- **Routing stays deterministic.** The model assessed severity and produced findings; whether that
  warrants paging a human is policy, expressed as `ESCALATION_SEVERITIES` in plain Python.

**Live end-to-end result — both scenarios correct:**

| | Pool exhaustion | Upstream failure |
|---|---|---|
| Severity | `high` | `high` |
| Tools called | all three | all three |
| Termination | `completed` | `completed` |
| Root cause | connection pool exhausted | `card-authorization-api` 503s |
| Routed to | `escalate` | `escalate` |
| Recommended action | increase pool size | investigate the upstream API |

An unplanned improvement appeared: the model began **citing its sources inline** —
*"The error rate is 12.5% (get_mock_metrics) … the logs indicate repeated errors (get_mock_logs)"* —
attributing each claim to the tool that produced it. That emerged from the *"only state facts you
actually observed"* rule, and it is exactly the traceability the `Evidence` model exists to provide.

#### 5f. The healthy-service scenario, and moving escalation into code ✅

**Goal:** both existing scenarios contain a real fault, so the agent has never had the option of
being wrong by *inventing* one. Models are strongly biased toward finding something when told to
investigate. This tests whether it can say "nothing is wrong".

**Built:** a `HEALTHY` scenario — but deliberately not a blank page. A perfectly silent fixture
would be an unrealistically easy test, so it carries the kind of benign noise real systems always
have:

| Signal | Value | Purpose |
|---|---|---|
| Error rate | 0.2% | Nominal, not zero |
| DB pool | 18/100 | Plenty of headroom |
| Latency p95 | 120 ms | Fine |
| Circuit breakers | all closed | Both dependencies healthy |
| Logs | 3× INFO, **1× WARN** | *"Retry succeeded on second attempt"* |
| Pods | all ready, **1 restart** | From a routine deploy 71 h ago |

That WARN and that restart are the bait. The correct answer is that a retry which succeeded and an
old restart are not an incident.

**Baseline first, on purpose.** Before changing anything, the scenario was run against the
*unmodified* prompt. Fixing the prompt pre-emptively would have made it impossible to know whether
the problem was real or imagined.

**Result — it fabricated a fault:**

> Finding: **"The payment service is not handling the logs correctly."** *(confidence 0.8)*
> Action: *"Investigate the log handling configuration…"*

No such problem exists, and it contradicted itself in the same summary: *"the logs do not contain
any explicit error messages."* Notably it did **not** take the planted bait — it invented an
entirely new *category* of fault out of the logs being clean. **Told to find a root cause, "there
isn't one" was not an available answer, so it manufactured one.**

Two things degraded gracefully and are worth crediting: confidence fell to `0.3` (against `0.8` on
real incidents), and the deterministic router still sent it to `monitor` rather than `escalate`.
**The guardrail held while the model's output was worthless** — the strongest argument so far for
keeping routing in code.

**Diagnosis: both prompts were leading.**

| | Before | After |
|---|---|---|
| System | *"Your job is to find the **ROOT CAUSE**"* | *"determine what the evidence **actually shows** … sometimes the service is healthy. **Both are valid conclusions.**"* |
| Summariser | *"stating **what is wrong** with the service and why"* | *"stating what the evidence shows … **if operating normally, say that plainly** rather than manufacturing a problem"* |

The summariser was probably the worse offender, since it authored the fabricated finding. Two
explicit guards were added: *"Reporting a problem the evidence does not support is a serious error,
worse than reporting nothing"*, and *"Healthy systems still produce routine noise: a retry that
succeeded, an old restart, a warning that resolved itself. Do not treat routine noise as a fault."*

**Result after the fix:** *"running and healthy … no error messages or concerning activity"*,
confidence `0.9`, action *"continue monitoring"*. Every finding a real observation. Confidence went
**up**, because the conclusion is now genuinely supported.

##### The re-test exposed two further problems

**1. Overcorrection in the model.** On the upstream scenario it wrote *"the payment service is
operating with a **low error rate of 34.0%**"*. It read the number correctly and attached the wrong
judgement — taught that "healthy" is a valid conclusion, it began seeing health that was not there.

**2. A routing bug — mine, not the model's.** A 97% upstream failure rate routed to `monitor`.

`route_by_severity` consulted only `severity_assessment`, which is produced from the incident
*description* **before any investigation runs**. The flow had been reordered specifically so routing
could use findings — and then the router was written to ignore them. Worse, the live runs showed all
three scenarios being assessed `low` from a neutral description, so under the old logic **every one
of them, including the 97% failure, would have gone to `monitor`.**

##### Fix: `domain/escalation.py`

Escalation now reads the **raw tool JSON** and applies explicit thresholds:

| Rule | Operator | Threshold |
|---|---|---|
| Error rate | `>` | 5.0 % |
| DB pool utilisation | `>=` | 90.0 % |
| p95 latency | `>` | 2 000 ms |
| Upstream dependency error rate | `>` | 10.0 % |
| Circuit breaker open | — | any |
| Pod not ready | — | any |
| No evidence / unreadable evidence | — | always escalates |

This also immunises the system against problem 1 above: code comparing `34.0 > 5.0` cannot decide
that 34% is low. The model's job is finding and explaining; whether numbers justify paging someone
is policy, and policy reads the numbers rather than the model's characterisation of them.

**A safety bug found while reviewing the new policy.** Unparseable output — malformed JSON, an empty
string, an error-shaped result — originally produced *no reasons*, so it fell through to "do not
escalate". An incident where every tool returned garbage would have been silently marked healthy.
**Absence of readable evidence had become evidence of absence** — the same class of error as the
model's fabrication, inverted, sitting in the layer written to be trustworthy. Unreadable evidence
now escalates, and says which source could not be read.

**Boundary tests.** `tests/test_escalation.py` (41 tests, no LLM) pins each rule at its exact
threshold — `5.0 → no`, `5.01 → yes`; `89.99 → no`, `90.0 → yes`; `2000 → no`, `2001 → yes` — so an
accidental `>` / `>=` swap fails loudly. It also covers malformed JSON, wrongly-typed fields, missing
keys, error-shaped output, healthy evidence, and several rules tripping at once.

**A flaky test was fixed, not weakened.** `test_assess_severity_live` asserted the model returns
exactly `CRITICAL`; it sometimes returns `HIGH` for the same input. It now asserts the severe *band*.
A suite that fails at random trains you to ignore failures — and what actually matters for routing is
that a total outage lands in the severe band, not which of two labels it picks.

##### The fix over-corrected, and a repeated run caught it

Re-running the same three scenarios a second time — same code, same prompts — produced a **worse**
failure than the one just fixed. On the upstream scenario, where the service's own error rate is 34%,
p95 latency is 10.2 seconds and the circuit breaker is open, the agent reported:

> *"The payment service is **not experiencing errors** and customers can complete checkout."*
> *"The upstream service 'card-authorization-api' is experiencing a high error rate (97.0%) but
> **this is not affecting the payment service**."*
>
> — both at confidence **1.0**, recommended action: *"Continue monitoring"*

A confident all-clear during an active outage. Materially worse than the earlier *"34% is low"*,
which was a mislabelled number rather than a wrong verdict.

**The healthy-scenario fix caused it.** The prompt had been given *"Evidence that the service's own
resources are healthy is meaningful, not a dead end"* — and in this scenario the pods genuinely
**are** all ready, because the fault is upstream. The model took one healthy signal and generalised
it to the whole service. One failure mode had been traded for another.

> **Healthy infrastructure is not a healthy service.** Pods can all be ready while a third of
> requests fail. These are different claims about different things, and a prompt that praises
> "evidence of health" without qualification invites the model to conflate them.

The prompt now states this explicitly — *"The service is only healthy if its own error rate and
latency are normal too. If requests are failing or slow, there IS an incident, whatever is causing it
and however healthy the pods look"* — and names *"reporting that a failing service is fine"* as an
error of the same seriousness as fabricating a fault. After the fix the same scenario produced
*"The payment service is **not healthy** … The service's own error rate is 34.0%, indicating that it
is experiencing errors"*, severity `critical`, confidence 0.9.

##### What repeated runs revealed: behavioural variance

The single most useful thing about running the same three scenarios more than once was discovering
that **they do not fail the same way twice**:

| Scenario | Run 1 | Run 2 | Run 3 (after fix) |
|---|---|---|---|
| Pool exhaustion | Partly right; 2 factual errors | **Correct** | Correct; p95/p99 slip |
| Upstream | Correct | **False all-clear at 1.0** | **Correct**, "not healthy" |
| Healthy | Correct, 3 findings | Correct, **0 findings** | Correct, 3 findings |

No run was clean, and each failed in a different place. A single passing run would have justified
shipping any of these states. The lesson generalises beyond this project: **with a non-deterministic
component, one green run is an anecdote.** This is the concrete argument for the golden-test suites
planned in Phase 13 — a behaviour that appears fixed may simply not have recurred yet.

##### The deterministic layer earned its place

Across every run above, including the one that declared an active outage healthy, **routing was
correct every single time.** `escalate` fired on both failure scenarios and stayed off for the
healthy one, because `domain/escalation.py` reads `error_rate_percent: 34.0` from the raw tool JSON
and compares it to a threshold, rather than reading the model's sentence about it.

This is the clearest vindication so far of the project's central principle. The model's narrative was
wrong in three separate ways across three runs — a fabricated fault, a false all-clear, a
percentile misread — and not one of them reached the routing decision. **The LLM's semantic
interpretation can flatly contradict the numeric evidence it just read; the safety boundary must
therefore consume the numbers, never the interpretation.**

The residual exposure is worth naming plainly: routing is protected, but the human still reads the
model's `summary` and `recommended_next_action`. A wrong narrative attached to a correct route can
still mislead the person acting on it.

##### A duplicate-reason bug the tests did not catch

The confirmation run surfaced a defect in the new policy itself. Scenario 1 returned **five**
escalation reasons, with `"2 of 3 pods not ready"` appearing twice.

The agent had called `get_mock_pod_status` twice — the prompt asks it not to repeat calls, but that
is a request, not a guarantee — producing two identical `Evidence` records, and `evaluate` reported
the same observation once per record.

Safety impact was nil: duplication can only add reasons, never remove them, and the incident
escalated correctly. But a reader seeing that line twice would reasonably infer two separate
problems. Reasons are a set of distinct facts, not a log of checks performed.

Worth recording *why it slipped through*: the existing test
`test_all_tripped_rules_are_reported_together` asserts `len(decision.reasons) == 5`. **Counting
reasons is not the same as checking they are distinct** — the assertion would have passed on the
duplicated output. Fixed with an order-preserving `dict.fromkeys()`, plus two tests: one driving
repeated evidence, one confirming that deduplication does not collapse genuinely different
observations.
### Phase 5.6 — Provider abstraction, and switching from Ollama to OpenRouter ✅

**Goal:** Ollama's local disk footprint (multiple GB per model, plus the runtime itself) became
impractical on the development machine. Move to a hosted, free-tier API — OpenRouter — without
quietly re-coupling the codebase to a single vendor the way it had been coupled to Ollama.

**The gap this closed.** `UPDATESLOG.md`'s own tech-stack table had claimed, since Phase 3, that
Ollama "forces provider-agnostic design from day one." That was the *intent*, but both LLM call sites
(`llm/severity.py`, `agents/investigation.py`) imported `ollama` directly and called `ollama.chat()`
against its specific response shape (`response.message.content`, `.tool_calls[].function.arguments`
as an already-parsed dict). Nothing had ever tested whether the design was actually
provider-agnostic, because only one provider had ever existed to test it against.

**Built:**

```
src/agentic_ai/llm/
├── client.py                      LLMClient protocol, ChatResponse, ToolCall, get_client()
└── providers/
    ├── ollama_provider.py         wraps ollama.chat(), synthesises a tool_call id
    └── openrouter_provider.py     OpenAI-compatible HTTP client via httpx
```

`LLMClient` has exactly one method, `complete(messages, tools=None, json_schema=None) ->
ChatResponse`, because that is the entire surface both call sites ever needed — a one-shot
structured-output call (severity assessment) and a multi-turn tool-calling loop (investigation).
Each provider translates its own wire format into `ChatResponse` and back; callers never see
Ollama's `response.message` or OpenRouter's `response.choices[0].message`.

**Key decisions:**

- **Ollama was kept, not deleted**, specifically so the abstraction would have two real
  implementations to prove it against. An interface validated by only ever having one provider
  behind it is not actually validated — the next section shows exactly what that would have hidden.
- **OpenRouter is the new default** (`LLM_PROVIDER=openrouter` in `.env`), with a free-tier model
  (`nvidia/nemotron-3-super-120b-a12b:free`, switched from an initial choice of
  `google/gemma-4-26b-a4b-it:free` after that model's free pool was rate-limited on first live test —
  confirmed via OpenRouter's own error body, not assumed). Free-tier models on OpenRouter are
  explicitly documented by OpenRouter as being shared across all users and subject to disappearing
  or being rate-limited without notice; this is accepted as a known, named operational risk rather
  than solved with retry logic in this pass — it stays in Known Issues below.
- **Secrets handling, done at the moment it first mattered.** `.env.example` (committed, no real
  values) documents the required variables; `.gitignore` now excludes `.env` itself. This closes a
  gap flagged explicitly as far back as Phase 1 — *"must be added the moment any API key / secret is
  introduced, before the key touches disk in a tracked file"* — at the actual moment it became true,
  not after the fact.

**A real security incident during this work, worth recording honestly.** The user's first attempt
at providing the API key placed it in `.env.example` — the committed template file, not the
gitignored `.env`. This was caught before any commit or push (`git ls-files .env.example` confirmed
it was never tracked), but the key had already been typed into the conversation, so the safe
assumption is that it should be treated as compromised regardless of whether it reached git. The
file was corrected immediately (secret moved to `.env`, template restored with no value), and key
rotation was recommended but left as the user's explicit choice to do after confirming the
integration worked — a judgment call the user made deliberately, not an oversight.

**A real cross-provider bug found by having two providers, exactly as the abstraction was designed
to surface.** The first live end-to-end investigation against OpenRouter failed with `400 Bad
Request`. The actual response body (fetched directly, not guessed at) read:

> `messages[3]: tool messages must include a non-empty string tool_call_id`

OpenRouter's OpenAI-compatible API strictly enforces the tool-calling protocol: every assistant
`tool_calls` entry must carry an `id`, and the matching `tool`-role result message must echo it back
as `tool_call_id`. Ollama's tool calls have no `id` field at all (confirmed by inspecting
`ollama._types.Message.ToolCall.model_fields` directly rather than assumed) — this had silently never
mattered while Ollama was the only provider in existence.

Fixed by making `ToolCall.id` a required field: the Ollama provider synthesises a stable per-response
id (`call_0`, `call_1`, ...) since it has none of its own, and OpenRouter's provider passes through
the real id it returns. `agents/investigation.py`'s message construction now threads that id through
both the assistant message and its matching tool-result message.

> **This is the whole reason Ollama was kept behind the interface rather than deleted.** A
> single-provider abstraction would have shipped this bug invisibly — there would have been no
> second wire format to disagree with the first.

**Live verification, not just the mocked suite.** Both the `connection_pool_exhaustion` and
`healthy` scenarios were re-run end-to-end against the real OpenRouter API after the fix:

- Pool exhaustion: all three tools called, correct diagnosis citing the actual retrieved numbers
  (100% pool utilisation, 12.5% error rate, p95/p99 reported correctly and not confused — a
  recurring weakness under the smaller local model in earlier phases), confidence 0.9, routed to
  `escalate`.
- Healthy: all three tools called, correctly reported no incident, correctly treated a benign retry
  warning as routine noise rather than a fault, routed to `monitor` — the Phase 5.5 fabrication fix
  holds under the new provider, not just under the model it was originally tuned against.

**123 tests passing**, including the one live test, after updating the two test files that had
monkeypatched `ollama.chat` on the module directly (`test_severity.py`, `test_investigation.py`) to
instead inject a fake `LLMClient` or patch `get_client()` — a more direct seam that does not need to
change again if a third provider is ever added.
### Phase 6 — RAG: a knowledge base of past incidents ✅

**Goal:** the investigation agent reasoned from raw evidence alone every single time, with no benefit
from "we have seen this exact failure pattern before." RAG gives it access to a corpus of past
incidents it can retrieve from and ground its reasoning in, without changing what kind of evidence
the escalation policy trusts.

**Built:**

```
src/agentic_ai/
├── domain/knowledge.py           KnowledgeEntry, KnowledgeSearchResult
└── tools/
    ├── knowledge_corpus.py       18 synthetic past-incident records
    ├── knowledge_retrieval.py    embed-once, in-memory cosine similarity search
    └── search_tool.py            search_knowledge_base(), the agent-facing tool
```

**Key decisions:**

- **In-memory, not a vector database.** The corpus is ~20 static Python objects; brute-force cosine
  similarity over that many vectors costs microseconds. Argued against a real vector DB (Chroma /
  pgvector) explicitly against this project's own stated principle — *"add complexity only when a
  real problem demands it"* — since neither problem a vector database actually solves (a corpus too
  large to hold in memory, or one that changes at runtime and needs persistence) exists here. Revisit
  if either becomes true.
- **Local embeddings (`sentence-transformers`, `all-MiniLM-L6-v2`), not a hosted embedding API.**
  Keeps the corpus searchable with zero additional paid-API dependency beyond OpenRouter, and the
  ~80MB model checkpoint is a non-issue on this machine (156GB free) — the disk-space constraint that
  moved the project off Ollama was about multi-GB LLM checkpoints, not a blanket rule against any
  local model.
- **The embedding model loads lazily, on first search, not at import time.** Importing
  `knowledge_retrieval` happens as a side effect of importing the tools package at all; paying a
  ~70-second one-time model-load cost (confirmed by direct measurement) on every import, including
  test collection, would have been a real cost for no benefit until retrieval is actually used.
- **Retrieval matches on symptoms only, not on root_cause or resolution.**
  `KnowledgeEntry.as_search_text()` embeds title + symptom_description exclusively. If the answer
  text itself were embedded, a query that happened to share vocabulary with a *resolution* (e.g.
  "restart the pod") could outrank an entry with a genuinely similar *symptom* but a differently-worded
  fix. This mirrors the same discipline that kept `Evidence` and `Finding` separate in Phase 5:
  observation and inference must not be allowed to contaminate each other.
- **The corpus was deliberately written to contain near-duplicate pairs**, not just obviously distinct
  entries, because a retrieval system tested only on easy cases proves nothing. RB-001 (local database
  exhaustion) vs RB-002 (upstream, locally healthy) share surface vocabulary; RB-006 vs RB-007 are
  both Kubernetes CrashLoopBackOff with different causes (bad config vs insufficient memory); RB-002
  and RB-009 both read as "the problem is upstream" but differ in symptom (timeout/5xx vs HTTP 429).
  Live-verified: the corpus and model correctly discriminate all of these when the query describes
  the actual positive symptom.

**A real finding about the embedding model, caught by a failing test rather than hidden.** An early
version of the RB-001/RB-002 discrimination test phrased its query as a negation — *"our database is
completely healthy"* — and it mis-ranked RB-001 (a database-themed entry) above RB-002, even though
the query was specifically trying to rule the database **out**. Diagnosed directly rather than
assumed: `all-MiniLM-L6-v2` is measurably weaker at negation than at topical/lexical matching —
mentioning "database" at all, even to deny it, pulled database-themed entries up in the ranking.
Rewriting the query to describe the actual positive symptom (timeouts calling an external dependency)
rather than negating the alternative fixed it with a clear margin. Kept as a documented finding and a
reusable principle for writing retrieval queries generally, not quietly tuned away.

**The mandatory-vs-optional tool question, decided explicitly rather than defaulted.** The existing
Phase 5.5 coverage-enforcement logic (`MIN_DISTINCT_TOOLS`, `should_continue`,
`require_more_evidence_node`) computed "which required tools are still missing" as
`set(INVESTIGATION_TOOLS) - used` — every *registered* tool was treated as mandatory. Adding
`search_knowledge_base` to the registry naively would have forced the agent to consult past incidents
on every single investigation, including ones with no relevant precedent, burning a step for no
benefit. Introduced `REQUIRED_TOOLS` (the three observability tools only) as the actual floor the
coverage logic enforces, leaving the knowledge-base tool registered, callable, and mentioned in the
prompt as explicitly **optional** — the agent decides whether a precedent is worth checking. The
registry itself was generalised at the same time: `dispatch_tool` previously assumed every tool took
`service` + `scenario`; it now distinguishes scenario-aware tools (the three mock observability tools,
which read whichever incident scenario is active) from the knowledge base (one real corpus,
independent of any scenario) via an explicit `_SCENARIO_AWARE_TOOLS` set, rather than passing
`scenario` to every tool uniformly and hoping it is ignored.

**Live-verified end to end.** Across three live runs (one per scenario):
- `connection_pool_exhaustion`: the agent called all three required tools, **then chose to call
  `search_knowledge_base`** on its own judgement, retrieved RB-001 at 0.722 similarity (the correct,
  directly matching runbook), and produced a correct diagnosis.
- `upstream_dependency_failure` and `healthy`: the agent concluded confidently from the three
  required tools alone, **without** calling the knowledge base — it judged the precedent lookup
  unnecessary. This is the "optional, agent decides" design working as intended, not a gap: the tool
  is available, not force-fed into every run.
- One run on the pool-exhaustion scenario produced `model_assessed_confidence = 0.0` on every finding
  despite a correct, well-evidenced diagnosis. Re-run immediately with identical inputs and it
  produced sensible confidences (0.92) instead — consistent with the already-documented
  "behavioural variance across runs" finding from Phase 5.5, not a defect introduced by this change.
  Confirmed by direct comparison that the knowledge-base retrieval itself was correct and unchanged
  across both runs.

**142 tests passing** (up from 123): 17 new tests in `tests/test_knowledge.py` covering corpus
integrity, the symptom/answer separation invariant, real-embedding retrieval quality on the
deliberately-hard pairs, and the agent-facing tool's JSON contract and error handling; plus 2 tests in
`test_investigation.py` split and extended to cover the generalised registry (one asserting the full
4-tool schema set, one confirming the knowledge-base tool's schema has no `service` parameter).
### Phase 7 — Root cause analysis: the second reasoning step, deliberately not a second agent ✅

**Goal:** the Investigation Agent's `findings[]` are per-observation statements produced *while*
gathering evidence, under the constraint of also deciding what to look at next. Synthesizing those
findings into one coherent root cause, weighing alternatives, and checking consistency against a
retrieved precedent is a genuinely distinct reasoning task - but it is not another evidence-gathering
loop, and building it as a second agent would have been complexity the task does not need.

**Scoped deliberately narrow before any code.** The project's roadmap lists five potential agents
(Investigation, Knowledge, RCA, Remediation Planner, Verification) plus a Supervisor. Built one new
piece: RCA. No Supervisor yet, since with only two steps in the pipeline (Investigation, then RCA)
there is no actual routing decision between multiple agents for a Supervisor to make - introducing one
now would be exactly the premature-infrastructure pattern this project has repeatedly rejected
elsewhere (the vector-database decision in Phase 6, the "no database until a problem demands one"
principle stated from Phase 1 onward).

**Built:**

```
src/agentic_ai/
├── domain/rca.py       RootCauseAnalysis, AlternativeHypothesis, KnowledgeBaseConsistency
└── llm/rca.py          analyze_root_cause() - one structured-output call, no tools, no loop
```

**The `llm/` vs `agents/` placement, decided on shape rather than the roadmap's word "agent".**
RCA takes a finished `InvestigationResult`, makes one structured-output LLM call, and returns a
result - architecturally identical to `assess_severity()` in `llm/severity.py`, not to the looping
`agents/investigation.py`. The roadmap calls it an "RCA Agent" because that is its conceptual role in
the pipeline; the code is named for what it actually is. This is also what the Phase 4 `workflows/`
vs `agents/` boundary question - left open since LangGraph was introduced - was actually resolved by:
`workflows/` is graph wiring, `agents/` means a loop that reasons over accumulating state and chooses
its own next action, `llm/` is a stateless one-shot capability. RCA fits the third category cleanly,
and forcing it into `agents/` to match the roadmap's vocabulary would have made that boundary mean
nothing.

**RCA has no tool access, enforced structurally.** The prompt built in `llm/rca.py` never offers
`tools=` to the LLM client - there is nothing for the model to call even if it wanted to re-query the
knowledge base or re-check a metric. Investigation gathers; RCA synthesizes. This mirrors the same
discipline that keeps `InvestigationResult`'s own summariser from authoring `evidence` or
`termination_reason`: the boundary is in what the model is offered, not in an instruction asking it
to behave.

**Revisited a deliberate Phase 5 deferral, now that something concrete needed it.** `Finding` gained
an `id: UUID` field. Phase 5's `UPDATESLOG` entry had explicitly flagged `supporting_evidence_ids` as
"needed for RCA... premature now" - this is that exact deferral being closed at the moment RCA
actually required it, not speculatively earlier. The field has a `default_factory`, so no prompt or
LLM output needed to change to accommodate it; all 142 then-existing tests passed unchanged.

**Citations are validated, not trusted.** `RootCauseAnalysis.supporting_finding_ids` is checked
against the real `Finding.id` values present in the input before being accepted; an unresolvable
citation is treated as invalid structured output and retried, the same way an invalid enum value or a
malformed JSON body already are elsewhere in this project. The alternative - logging a warning and
accepting it anyway - would let a fabricated citation sit in the result looking exactly as
authoritative as a real one.

**The mandatory-vs-advisory distinction from Phase 6 generalises here.** Just as
`search_knowledge_base` is optional for the Investigation Agent, RCA's own use of knowledge-base
results is read from whatever `Evidence` the investigation happened to produce
(`source_tool == "search_knowledge_base"`), not fetched fresh. If no search was performed,
`knowledge_base_consistency` is expected to be `no_relevant_precedent` rather than the model
inventing a precedent to discuss.

**Where RCA sits in the workflow, decided to preserve the Phase 5.5 escalation principle.**

```
assess_severity -> investigate -> route_by_severity -+- urgent -> analyze_root_cause -> escalate
                                                       `- normal -> monitor
```

RCA runs strictly *after* routing, on the escalation path only. Two reasons, both load-bearing:
- An incident routed to `monitor` was not an incident; there is no root cause to explain, and running
  RCA anyway would spend an LLM call for nothing.
- `route_by_severity` must keep deciding escalation from raw evidence alone, the principle Phase 5.5
  fought hard to establish after watching an LLM call a 34% error rate "low". Running RCA before
  routing - or worse, letting routing read RCA's output - would reopen exactly that risk by handing a
  safety decision to narrative instead of numbers. A test
  (`test_routing_is_unaffected_by_rca_regardless_of_its_content`) pins this directly: a
  deliberately low-confidence RCA result must not change a routing decision severity has already
  made, since RCA runs strictly after that decision in the graph.
- `analyze_root_cause_node` also skips RCA whenever the investigation produced no findings or failed
  outright - nothing to synthesize in either case - and a failure inside RCA itself does not block
  escalation: a human is already being paged, and an unexplained incident with no RCA is still better
  handed to a human than silently dropped. Only genuinely unexpected errors (not
  `RootCauseAnalysisError`) propagate, the same failure-category discipline used everywhere else in
  this project.

**Live-verified end to end, with one honest gap.** On `connection_pool_exhaustion`: the agent
investigated, retrieved RB-001 from the knowledge base on its own judgement (Phase 6 behaviour
unchanged), and RCA produced a root cause citing 5 real finding ids, correctly judged the knowledge
base result `supported`, and proposed two alternative hypotheses (thread-pool starvation, TLS
certificate expiry) with specific discriminating evidence for ranking each below the primary
conclusion rather than generic hedging. On `upstream_dependency_failure`: correct diagnosis, `escalate`,
knowledge base `supported` at 0.98 confidence. The `healthy` scenario could not be re-verified live in
this session - OpenRouter's shared free-tier pool returned `429 Too Many Requests` on repeated
attempts, the same documented risk flagged in Phase 5.6 ("no retry/fallback logic exists yet"), not a
Phase 7 regression. The "RCA does not run on the monitor path" behaviour is still fully verified at
the unit level (`test_rca_does_not_run_on_the_monitor_path`, deterministic, no LLM involved) - this
gap is specifically "not reconfirmed live this session," not "unverified."

**161 tests total, 160 passing in this session** (the one failure being the live-model rate limit
above, environmental, not a code defect): 12 new in `tests/test_rca.py` (domain model validation,
prompt-section rendering, citation validation including the retry-then-raise path), 7 new in
`test_incident_workflow.py` covering the full RCA integration matrix (runs when expected, skipped on
the monitor path, skipped with no findings, skipped when investigation itself failed, a failure inside
RCA does not block escalation, unexpected errors still propagate, and routing is provably unaffected
by RCA's content).


---

## 5. Current State

**Test suite: 161 tests** (43 escalation · 39 mock tools · 18 investigation · 17 knowledge ·
12 rca · 17 workflow · 11 incident · 4 severity). Exactly one touches the live LLM directly
(`test_assess_severity_live`); the knowledge tests load a real local embedding model but no LLM; the
rest are fully deterministic. 160 passed as of this writing - the live test failed on a transient
OpenRouter free-tier 429, not a code defect (see Known Issues).

```
src/agentic_ai/
├── domain/
│   ├── incident.py          Incident, Severity, IncidentStatus, transition_to()
│   ├── investigation.py     Evidence, Finding (now with id), TimelineEntry,
│   │                        InvestigationResult, TerminationReason, TimelineEventType
│   ├── escalation.py        Deterministic escalation policy over raw evidence
│   ├── knowledge.py         KnowledgeEntry, KnowledgeSearchResult
│   └── rca.py               RootCauseAnalysis, AlternativeHypothesis,
│                            KnowledgeBaseConsistency
├── llm/
│   ├── client.py            LLMClient protocol, ChatResponse, ToolCall, get_client()
│   ├── severity.py          SeverityAssessment, assess_severity(), custom exceptions
│   ├── rca.py                analyze_root_cause() - one call, no tools, no loop
│   └── providers/
│       ├── ollama_provider.py       local model, synthesises a tool_call id
│       └── openrouter_provider.py   hosted, OpenAI-compatible, default provider
├── tools/
│   ├── mock_tools.py          get_mock_logs/metrics/pod_status(), MockService
│   ├── registry.py            build_tool_schemas(), dispatch_tool(), REQUIRED_TOOLS
│   ├── knowledge_corpus.py    18 synthetic past-incident records
│   ├── knowledge_retrieval.py embed-once, in-memory cosine similarity search
│   └── search_tool.py         search_knowledge_base(), the agent-facing tool
├── agents/investigation.py  bounded ReAct subgraph, investigate(), coverage enforcement
├── workflows/incident.py    assess -> investigate -> route -> [RCA] -> escalate/monitor
└── api/                     (empty — later)
```

**Dependencies:** `langgraph>=1.2.11` · `httpx>=0.28.1` · `python-dotenv>=1.2.4` ·
`sentence-transformers` · `numpy` · `pydantic>=2.13.5` ·
`ollama>=0.6.2` (kept, no longer the default) · *dev:* `pytest>=9.1.1`

**LLM provider:** OpenRouter by default (`LLM_PROVIDER=openrouter` in `.env`, see `.env.example`),
model `nvidia/nemotron-3-super-120b-a12b:free`. Ollama remains available via `LLM_PROVIDER=ollama`.

**Branches:** everything through Phase 5.5 has been merged to `main` and pushed; Phase 5.6
(this provider switch) is committed locally on `main`, not yet pushed.

---

## 6. Known Issues & Deferred Work

| Item | Status | Notes |
|---|---|---|
| `Finding.supporting_evidence_ids` | Deferred, recorded | Needed for RCA, audit, explainability |
| `TIMEOUT` termination reason | Reserved, unreachable | No wall-clock enforcement implemented yet |
| `TimelineEntry.event_type` | Not built | Must be an **enum**, not free text |
| `workflows/` vs `agents/` boundary | Provisional | Held up so far: the workflow node *calls* the agent rather than embedding it. Will be tested again in Phase 7 by the Supervisor |
| `escalate` / `monitor` are stubs | Open | They set `next_step` but take no action. Real behaviour arrives with Phase 10 approval gates and Phase 11 remediation |
| `AVAILABLE_SERVICES` mutability | Minor | `list` → `tuple` would prevent accidental mutation |
| Linter / formatter | Not set up | `ruff` or `black`; stray-whitespace artifacts already seen |
| Git author identity | Unconfigured | Commits carry an auto-guessed name/email on a public repo |
| Model-assessed confidence calibration | Unknown | Ordering looks sensible; true calibration is a Phase 13 question |
| Prompt robustness | Improved | Verified on three scenarios including a healthy service. A fourth (partial/ambiguous evidence) remains untested |
| Behavioural variance across runs | Open | The same scenarios do not fail the same way twice. One green run proves little; this is the concrete case for Phase 13 golden tests |
| Knowledge corpus is synthetic and static | Open | 18 hand-written runbooks, no real incident history, no mechanism to add entries without a code change. Fine for proving the retrieval mechanism; not a real knowledge base yet |
| Embedding model weak at negation | Open | `all-MiniLM-L6-v2` mis-ranked a database-themed entry above the correct one when a test query negated "database" rather than describing the actual positive symptom. Noted as a property of this model size, not fixed - queries should describe what IS happening |
| Escalation thresholds | Provisional | The numbers (5%, 90%, 2000ms, 10%) are reasonable defaults, not tuned against real incident data |
| Model narrative misstates observed values | Open | Recurring, not a one-off. Across live runs it reported p95 as 5100ms (the fixture's p99; p95 is 2400), called a 12.5% error rate "within normal ranges" at 0.8 confidence, and called 34% "low". Routing is unaffected because escalation reads raw numbers, but the summary a human reads can be wrong, and `recommended_next_action` is model-authored. A Phase 13 evaluation target |
| Free-tier model availability | Open | OpenRouter free models are shared across all users and can be rate-limited or withdrawn without notice (observed directly: the first model chosen hit a 429 within the same session it was picked). No retry/fallback logic exists yet; a sustained outage of the default model will surface as a 429 to the caller |
| Free-tier rate limit recurred in Phase 7 | Open | Confirmed again, not just a one-off: adding an RCA call per escalating incident increases free-pool consumption, and a live verification run hit 429 on retry within the same session. No retry/backoff exists; this is now two independent confirmations of the same documented risk, worth addressing before relying on live runs for anything time-sensitive |
| Exposed API key | Resolved, flagged | A key was briefly typed into `.env.example` (the committed template) instead of `.env`. Caught before any commit or push, corrected immediately, but the key reached the conversation and should be rotated — left as the user's explicit choice, not yet done as of this writing |

---

## 7. Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Engineering foundations | ✅ |
| 2 | Incident domain model | ✅ |
| 3 | LLM foundation — structured output | ✅ |
| 4 | LangGraph — state, nodes, routing | ✅ |
| 5 | Investigation agent + mock tools | ✅ |
| 6 | RAG — engineering knowledge retrieval | ✅ |
| 7 | Multi-agent — RCA done; supervisor, remediation planner remain | 🚧 |
| 8 | Memory — short-term, long-term, vector | ⬜ |
| 9 | MCP & tool engineering | ⬜ |
| 10 | Guardrails — risk tiers, approval gates | ⬜ |
| 11 | Remediation (mock first) | ⬜ |
| 12 | Verification loop | ⬜ |
| 13 | Evaluation — golden cases, regression | ⬜ |
| 14 | Observability — tracing, spans, tokens | ⬜ |
| 15 | Production — Docker, K8s, AWS, Terraform, CI/CD | ⬜ |

**Guardrail model planned for Phase 10:**

```
LOW      → automatic
MEDIUM   → additional validation
HIGH     → human approval required
CRITICAL → blocked
```

---

## 8. Principles

Extracted from decisions actually made in this project, not aspirational:

1. **The LLM controls strategy; deterministic code controls boundaries.** Step limits, timeouts,
   authorization, validation, and policy are never the model's job.
2. **Validate at every boundary.** LLM output is untrusted input. So is tool output.
3. **Never silently default on uncertain safety-relevant data.** Escalate or fail loudly; a
   fabricated `MEDIUM` can mean nobody gets paged.
4. **Distinguish failure categories.** Invalid data ≠ unavailable infrastructure ≠ programming bug.
   Each warrants different handling.
5. **Observations and inferences are different things.** Evidence can be right while a finding built
   on it is wrong.
6. **Preserve raw output verbatim.** Paraphrase is a separate, clearly-labelled layer.
7. **Entities protect their own invariants** — make invalid states unrepresentable rather than merely
   discouraged.
8. **Name things honestly.** `model_assessed_confidence`, not `confidence`. Naming is a guardrail.
9. **Add complexity only when a real problem demands it.** No database, no API, no MCP, no
   abstraction layer until something concrete requires it.
10. **Verify; don't assume.** Read the actual signature. Run the actual probe. Check the commit
    actually landed.
11. **Anything the LLM must know has to be in the contract.** Logic inside a function body is
    invisible to the model calling it.
12. **Document limitations honestly, including in tests.** A test that proves `_status` is still
    mutable is more useful than pretending encapsulation is absolute.
13. **Small models need procedural instruction, not goal-level direction.** Try proceduralising the
    prompt before reaching for a bigger model — it is far cheaper, and it tells you whether you have
    a prompting problem or a capability ceiling.
14. **Make invariants structural, not instructional.** Stripping `evidence` from the schema means the
    model *cannot* author it. Asking it politely not to would have been a hope, not a guarantee.
15. **You cannot instruct your way out of non-determinism.** The same prompt produced full tool
    coverage on one scenario and a single call on another. Prompts shift probabilities; only code
    provides guarantees.
16. **Test a second case before believing the first.** One passing scenario cannot distinguish
    reasoning from pattern-matching, and a prompt tuned on one example is a prompt overfitted to it.
17. **A leading question gets a leading answer.** "Find the root cause" made "nothing is wrong" an
    unavailable answer, and the model invented a fault rather than return empty-handed. Prompts must
    make the negative conclusion explicitly available.
18. **Absence of evidence is not evidence of absence.** Unreadable tool output must escalate, not
    fall through to "nothing tripped a threshold". This bug was in the deterministic layer written to
    be trustworthy — safety code needs the same scrutiny as model output.
19. **Safety decisions read the numbers, not the narrative.** The model called a 34% error rate
    "low". Code comparing `34.0 > 5.0` cannot. Where a threshold exists, compare against it directly.
20. **Fix flaky tests by asserting the right thing, not by loosening until green.** A live LLM test
    asserting one exact label became a band assertion — because the band is what actually matters
    downstream, not because the strict version was inconvenient.
21. **With a non-deterministic component, one green run is an anecdote.** The same three scenarios
    failed in three different places across three runs. Re-run before believing a fix.
22. **Fixing one failure mode can create its opposite.** Teaching the agent that "healthy" is a valid
    conclusion made it declare a failing service healthy. Re-test the cases the previous behaviour
    got right, not only the one being fixed.
23. **Counting is not checking.** A test asserting `len(reasons) == 5` passed on output containing
    the same reason twice. Assert the property you care about — distinctness — not a proxy for it.
24. **An abstraction with one implementation behind it is untested.** `LLMClient` existed in intent
    from Phase 3 onward, but nothing enforced it until a second provider existed to disagree with the
    first. The `tool_call_id` bug was invisible for the entire time Ollama was the only provider — it
    could only be a bug once something else cared.
25. **A secret typed into the wrong file is compromised the moment it is typed, not the moment it is
    committed.** Catching a leak before `git commit` is real and worth doing, but it does not undo the
    exposure; rotate the credential regardless of whether it reached version control.
26. **Not every registered capability is mandatory.** Generalising "all tools are required" to "all
    registered tools are required" would have forced a knowledge-base lookup into every investigation,
    including ones with no relevant precedent. A tool being available to the agent and a tool being
    part of the evidence floor are different facts; conflating them costs a wasted step every time the
    optional tool has nothing useful to contribute.
27. **Test retrieval on the hard pairs, not the easy ones.** A corpus where every document is
    trivially distinct proves nothing about ranking quality. Near-duplicate entries that share
    vocabulary but differ in root cause are what actually exercises whether retrieval works.
28. **A second "agent" in the roadmap is not automatically a second agent in the code.** RCA is named
    an agent in the project's own plan but has no loop and no tools; it is architecturally identical
    to a Phase 3 one-shot LLM call. Naming code for its actual shape, not for the vocabulary a plan
    used to describe its role, is what keeps an architectural boundary (`llm/` vs `agents/`) meaning
    something.
29. **A safety-relevant decision must not move even when a new, more confident-sounding voice joins
    the pipeline.** Escalation was already correctly decided before RCA existed. Adding RCA after
    that decision, with a test proving a deliberately low-confidence RCA result cannot change it, is
    what keeps a new source of narrative from quietly becoming a new way to override deterministic
    judgement.
