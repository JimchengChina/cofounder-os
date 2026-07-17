# CoFounder OS Development Guardrails

## Objective

Develop CoFounder OS without destabilizing the verified Mac and DGX Spark
runtime.

The governing principle is:

> Solve product problems at the product layer. Do not redesign infrastructure
> to accommodate an isolated implementation detail.

## Mandatory Workflow

Every implementation step must follow this sequence:

1. Inspect current state.
2. Define the exact module and files allowed to change.
3. Create a rollback point.
4. Make the smallest viable change.
5. Run module tests.
6. Run product smoke tests.
7. Confirm infrastructure health.
8. Record changed files and test results.
9. Stop and roll back when acceptance fails.

## Required Execution Summary

Development scripts should end with machine-readable fields:

```text
FINAL_RESULT=PASS|FAIL
CHANGED_FILES=...
BACKUP_PATH=...
TEST_RESULT=...
CURRENT_SERVICES=...
NEXT_ACTION=...
```

## Layer Ownership

### Infrastructure Layer

Includes:

- Qwen model weights
- vLLM environment and startup parameters
- Gateway provider connectivity
- systemd services
- launchd SSH tunnel
- SSH configuration
- Fixed ports

Feature work must not modify this layer.

### Product Platform Layer

Includes:

- Domain models
- Run and Task state
- JSONL Ledger
- Gateway Client
- Agent contracts
- Workflow Controller
- Policy Gate and Approval Gate
- Artifact generation
- Evaluation framework

This is the primary development layer.

### Presentation Layer

Includes:

- FastAPI product endpoints
- HTML, CSS, and JavaScript
- Demo timeline
- Route visualization
- Approval controls
- Artifact viewer
- Audit view

The presentation layer must consume Product APIs and must not contain
authoritative workflow logic.

## MVP Scope Discipline

The preliminary MVP supports one complete Founder workflow:

```text
Founder objective
    -> orchestration
    -> structured tasks
    -> Product and Finance agents
    -> explicit model routing
    -> deterministic policy gate
    -> human approval when required
    -> synthesized artifacts
    -> replayable audit trail
```

Completing this workflow is more important than adding more agents, models, or
infrastructure.

## Prohibited Shortcuts

Do not:

- Call Qwen directly from product modules
- Call Step directly from product modules
- Let agents mutate global state
- Hide routing decisions inside prompts
- Store authoritative state only in memory
- Treat natural-language output as a state transition
- Allow unbounded recursive task decomposition
- Execute high-risk actions without approval
- Log credentials or full environment values
- Build complex UI logic before the Product API contract exists
- Fix application errors by reinstalling vLLM
- Add frameworks because they may be useful later

## Initial Implementation Constraints

Until the preliminary MVP is frozen, use only:

- One Executive Orchestrator
- Product Agent
- Finance Agent
- Deterministic Policy Gate
- Artifact Synthesizer
- Three Gateway virtual models
- Append-only JSONL persistence
- FastAPI
- Static HTML, CSS, and JavaScript

Do not add:

- A production database
- A message queue
- A second local model
- External write integrations
- Automatic email or message delivery
- Automatic financial transactions

## State Authority

Only the Workflow Controller may change:

- Run status
- Task status
- Approval status
- Dependency resolution
- Retry count
- Terminal failure state

Agents may propose actions and return structured results, but they are not
authoritative state controllers.

## Model Output Discipline

All model output must be treated as untrusted input.

Every structured response must have:

- A versioned schema
- Pydantic validation
- Bounded repair attempts
- A terminal failure path
- An audit event
- Original provider and routing metadata

Invalid model output must never silently become valid system state.

## Risk Control

The Deterministic Policy Gate must block or require approval for:

- External writes
- Destructive file operations
- Production configuration changes
- Dangerous shell commands
- Private-data uploads
- External email or message delivery
- Material budget commitments
- Irreversible or high-impact operations

Approval is a formal state transition, not a natural-language instruction.

## Logging

Development log locations:

```text
Mac:
~/Library/Logs/CoFounderOS/dev/

DGX Spark:
~/cofounder-os/logs/dev/
```

Logs must provide enough diagnostic information to avoid large screenshot sets.

Logs may contain:

- IDs
- State transitions
- Timing
- Provider names
- Routing reasons
- Error classes
- Redacted payload summaries

Logs must not contain:

- API keys
- SSH private keys
- Gateway key values
- Full `.env` contents
- Unredacted user secrets

## Definition of Done

A development task is complete only when:

- Acceptance criteria pass
- Changed files are listed
- Test results are recorded
- Infrastructure remains healthy
- No secret or generated runtime data enters Git
- A rollback path exists
- The next dependency is explicit
