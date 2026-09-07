# AI Engineering & Incident Resolution Platform — Updates Log

> Single source of truth for this project: what it is, why it exists, what has been built in each
> phase, and the reasoning behind every significant decision.
>
> **Last updated:** 2026-09-07 · **Current phase:** Phase 5 complete — the full pipeline (assess →
> investigate → route) runs end-to-end against a local model on two scenarios. Phase 6 (RAG) next.

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
| **Ollama + `llama3.2`** | Runs a real open-weight model **locally at zero cost**. Also forces provider-agnostic design from day one rather than coupling to one vendor's SDK. | Anthropic/OpenAI hosted APIs — rejected on cost; revisit if capability becomes the bottleneck. |
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
├── domain/       Business entities and their invariants (Incident, Severity, lifecycle rules)
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

---

## 5. Current State

**Test suite: 67 passing** (11 incident · 4 severity · 8 workflow · 16 mock tools · 16 investigation ·
12 investigation domain/registry)

```
src/agentic_ai/
├── domain/
│   ├── incident.py          Incident, Severity, IncidentStatus, transition_to()
│   └── investigation.py     Evidence, Finding, TimelineEntry, InvestigationResult,
│                            TerminationReason, TimelineEventType
├── llm/severity.py          SeverityAssessment, assess_severity(), custom exceptions
├── tools/
│   ├── mock_tools.py        get_mock_logs/metrics/pod_status(), MockService
│   └── registry.py          build_tool_schemas(), dispatch_tool(), ToolNotFoundError
├── agents/investigation.py  bounded ReAct subgraph, investigate(), coverage enforcement
├── workflows/incident.py    assess -> investigate -> route, handle_incident()
└── api/                     (empty — later)
```

**Dependencies:** `langgraph>=1.2.11` · `ollama>=0.6.2` · `pydantic>=2.13.5` · *dev:* `pytest>=9.1.1`

**Branches:**

```
main
 └── feature/incident-domain-and-severity   (Phases 2–3)
      └── feature/langgraph-incident-workflow   (Phase 4, current) ← mock tools uncommitted here
```

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
| Prompt robustness | Improved | Verified on two scenarios; a third (e.g. "nothing is wrong") would test whether it can decline to find a problem |

---

## 7. Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Engineering foundations | ✅ |
| 2 | Incident domain model | ✅ |
| 3 | LLM foundation — structured output | ✅ |
| 4 | LangGraph — state, nodes, routing | ✅ |
| 5 | Investigation agent + mock tools | ✅ |
| 6 | RAG — engineering knowledge retrieval | ⬜ |
| 7 | Multi-agent — supervisor, RCA, remediation planner | ⬜ |
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
