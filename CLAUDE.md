# Working notes for Claude

## Writing documents (METHODS.md, PLAN.md, write-ups)

These are read by reviewers and, in this competition, by a selection model. They
record what was done and what was found. They are not a record of how the work
went.

- State findings, not process. "X was run; 14 of 20 are below moderate" — not
  "I realised X mattered, so I built a screen, and it turned out that...".
- Do not justify decisions in prose. Record the decision and the number behind
  it. Reasoning about alternatives belongs in the commit message.
- No signposting. Cut "Two caveats.", "The key insight is", "Notably",
  "It is worth noting", "This is important because".
- Match the length of the sections already there. In METHODS.md a limitation is
  one paragraph of 4-6 lines. If a new section is three times the longest
  existing one, it is wrong regardless of content.
- A negative result gets one or two sentences, not a table and a narrative. If a
  measurement changed nothing, say what it was and that it changed nothing.
- Numbers inline in prose. Reserve tables and fixed-width blocks for per-design
  data that a reader will actually look up.
- Check for an existing paragraph on the topic before adding one. Prefer
  revising it. Duplicating a point at greater length is the common failure.

Rule of thumb: after drafting, cut it in half. If halving loses nothing a
reviewer needs, the first version was self-indulgent.

## Compute

All GPU work goes through `sbatch --gpus=a100:1 -p cnu -A cosc020762`. The
`lbatch` A100 option is broken and the login node silently falls back to CPU.

## Repo

`origin` is ThomasHeap/mosaic, `upstream` is escalante-bio. No `gh` CLI.
