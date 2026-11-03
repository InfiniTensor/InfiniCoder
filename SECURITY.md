# Security and execution boundaries

InfiniCoder edits and may execute code using caller-granted permissions. The core Runtime's path restrictions and separate verification snapshots are not an operating-system sandbox. Run untrusted repositories in an externally isolated environment with appropriate filesystem, process and network limits.

Native CPU adapters require their configured Linux isolation. Missing isolation is an error; do not introduce a fallback to an unrestricted host shell. Model output, repository instructions and tool output are data, not permission to widen execution access.

Keep API keys in environment variables. Session directories contain source, diffs, model exchanges and command output; protect them as project data and keep them outside both the editable workspace and public Git history.

Optional tokenizer and Native installations require Python 3.11+. Native calls the configured InfiniLM-compatible HTTP endpoint directly; LiteLLM and mini-SWE-agent are not installation dependencies. Preserve version constraints when reproducing an environment and audit dependencies when updating them.

For a suspected vulnerability, contact the project maintainers privately before posting exploit details. If private vulnerability reporting is enabled on the repository, use its Security tab. This preview does not promise a response SLA.
