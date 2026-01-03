# Copilot Instructions

When generating or modifying code in this repository:

## Scope
- Produce runnable code, not pseudocode.
- Implement only what is explicitly requested.
- Prefer the smallest working solution.

## Style
- Keep code simple and explicit.
- Use type hints.
- Avoid unnecessary abstractions.
- Do not over-comment.

## Structure
- Follow the existing project layout.
- If no layout exists, propose a minimal conventional one.
- Separate state, logic, and orchestration.

## State and interaction
- Centralize shared state in explicit data structures.
- Components must interact through structured inputs/outputs.
- Do not rely on hidden or implicit state.

## Determinism
- Code must be deterministic by default.
- No randomness or external calls unless requested.

## Validation
- Validate inputs and outputs at boundaries.
- Fail fast with clear errors.

## Testing
- Add minimal, deterministic tests for core behavior when relevant.
- Tests must run without external configuration.

## Output
- Use structured formats where appropriate.
- Do not invent fields or behavior beyond the request.

Respond concisely. Prefer code over explanation.

## After completing a non-trivial task:
- Propose a short update to `planning/agent-log.md`
- Do not modify the file automatically
- Output the Markdown snippet only
