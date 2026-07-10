# Agent Prompts

Role prompts for multi-agent execution (Volume 4). Each prompt assumes
the agent has read `FMROS/00_README.md` + `01_Constitution.md` and has
repo access. In single-agent mode these become sequential passes; use
the pass order in Volume 4's closing note.

Files:
* `chief_scientist.md` — agenda, priority queue, cycle reports
* `statistician.md` — design sign-off, blocking/multiplicity policing
* `quant_researcher.md` — experiment design + implementation
* `reviewer.md` — adversarial pre-verdict review
* `risk_manager.md` — tail hunts, ruin analysis, house risk rules
* (remaining roles operate directly from Volume 4 specs; dedicated
  prompt files are added when a role is first exercised in multi-agent
  mode — an intentionally lazy policy to keep prompts grounded in use)
