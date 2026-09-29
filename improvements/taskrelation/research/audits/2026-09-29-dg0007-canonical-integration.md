# DG-0007 — canonical integration note

Status: integrated into `feature/mssl-task-relation-study` on branch `integrate/dg-0007`;
DG-0007 remains `BLOCKED` on authorization. No compute, no worker, no job, no
S3 or Network Volume mutation, no authorization created.

## Lineage (actual SHAs, read from Git ancestry — none assumed)

| Role | Ref | SHA |
|---|---|---|
| Integration base (canonical HEAD the worktree was cut from) | `feature/mssl-task-relation-study` | `664c572102f91b253358dcf8699e43a3c9c8e916` |
| Audit base (merge-base of audit branch and canonical) | — | `63b639f6eafff906a63ee0b8f80f6cecbdbb9984` |
| Accepted audit/remediation head | `research/mtrl-theory-audit` | `46fc0f95ea021c4e313d339e8db4c48ea1cfa8fd` |
| Independent review, final micro-verify | `review/mtrl-theory-audit` | `6be2b8b49bcee6a99ed462899528ab69854f1ab6` |

The audit base was *not* guessed: `git merge-base 46fc0f9 664c572` returns
`63b639f6eaff…`, which is an ancestor of canonical HEAD, and `664c572` is **not**
an ancestor of `46fc0f9`. The two lines diverge at `63b639f` and neither contains
the other.

Canonical commits newer than the audit base (ten, all excluded from the audit
branch):

```
664c572 Classify a failure during job preparation as infrastructure, not a code defect
d13e82e Walk the worker readiness ladder instead of crashing on it
62c9786 Grant the TR-0007 screen its narrow compute authorization
b0ba0f9 Fix the TR-0007 screen at a researcher-fixed lambda_2 and log the worker GPU
e6d5353 Record the TR-0007 publication, preflight and tracking readiness
8b40eed Give the TR-0007 controls the study's identity instead of untagged sweep runs
5dd12c6 Record the TR-0007 Option A decision and correct the MSSL equation
386b6e4 Return the MSSL solver tests to the gate and validate the TR-0007 screen
6c4ebab Pre-register the TR-0007 screen and its deterministic compute plan
bde6034 Implement the published MSSL Omega step faithfully for TR-0007
```

## Conflict classification, file by file

Every file the accepted delta touches is classified below. This is an additive
port, not a merge: the audit branch was **not** merged or cherry-picked.

### TAKE_CANONICAL — the newer canonical text wins outright

| File | Why |
|---|---|
| `Makefile` | Canonical `make research-check` became test *discovery* (`discover -s $(RESEARCH_TESTS) -t $(RESEARCH_TESTS)`, `research-check-all` an alias), and the MSSL exclusion was removed once DEC-0015 fixed both sides. The audit branch's three-name explicit list is obsolete by construction: new test modules are picked up automatically. Verified byte-identical to canonical. |
| `research/literature/goncalves-2016-mssl.md` | The audit's "repair" and canonical's correction are the **same blob** (`9d0ebfd`); the audit branch had already replaced its copy with canonical's. No delta exists. |
| `research/studies/TR-0007/NOTE.md` | Canonical carries the fuller record: the historical gate text left intact, plus append-only `DEC-0015` (Option A), the primary-source equation correction, the λ₂ open question and the `DEC-0016` screen authorization. The audit's added block said the same thing with less canonical context. Verified byte-identical to canonical. |
| `research/STATE.md` — the TR-0007 paragraph | The audit's own correction log (§9.5) instructs this: `STATE.md` "will conflict with the canonical branch in these regions; the canonical branch's newer text wins in every case". Canonical states the DEC-0015 Option-A resolution, the corrected Eq. (4b)/(8) reading and the solver fixes; the audit text records only the pre-resolution repair. |
| `research/STATE.md` — the "Latest/Previous iteration" chain | Canonical's TR-0007 iteration record is newer and stays. |

### TAKE_ACCEPTED_DG0007_DELTA — the accepted version survives verbatim

Each of these is byte-identical to `46fc0f9` in the integrated tree, because
canonical's blob at `664c572` equals the audit base's blob for that path, so the
accepted patch applied with no divergence:

| File | Content |
|---|---|
| `improvements/run_identity.py` | `research_identity()` — the study/arm identity contract |
| `improvements/base/run_base.py` | baseline entry point emits the same contract |
| `improvements/taskrelation/01-mtrl/README.md` | the audit section: `normalize_w` is a deviation, with the measured numbers, the label rule and the `U3` provenance caveat |
| `improvements/taskrelation/01-mtrl/mtrl_norm_corrected_25L_config.yml` | the corrected arm config (value-identical to the historical control apart from `model.normalize_w`, its output roots and its comment header) |
| `research/audits/2026-09-29-mtrl-theory-to-implementation-audit.md` | the audit report and its correction log |
| `research/studies/DG-0007/{PLAN.md,NOTE.md}` | pre-registration and run note |
| `research/studies/DG-0007/configs/*.yml` (6) | screen and confirmation execution configs for the three arms |
| `research/tests/test_mtrl_theory_faithfulness.py` | the mathematical regression suite |
| `research/tests/test_dg0007_run_identity.py` | the run-identity/provenance suite |

`research/studies/DG-0007/check_runtime_faithfulness.py` and
`research/tests/test_mtrl_runtime_faithfulness_checker.py` are also
`TAKE_ACCEPTED_DG0007_DELTA` **plus** the micro-verify MEDIUM fix recorded below.

### SEMANTIC_MERGE — port only the accepted hunk

| File | Merge |
|---|---|
| `improvements/run_improvements.py` | Canonical added the `mssl` model type, its config override, its `build_model`/`build_trainer` branches, the `layer_count`/`worker_gpu` tags and the `--model` choice — none of which the audit branch had. The accepted delta touches only the `emit_run_identity` call and its import. The union is the integrated file: the identity hunk is applied and every canonical MSSL line is preserved. Mechanically verified — the whole diff against canonical is 9 added and 4 removed lines, all of them the identity contract. |
| `research/STUDIES.jsonl` | Canonical's `TR-0007` record (with `DEC-0015`/`DEC-0016`, resolver-fixed λ₂, publication and preflight) is preserved **byte-for-byte**; the accepted `DG-0007` record is appended. The audit branch deliberately did not touch `TR-0007` here, and this integration does not either. |
| `research/STATE.md` | Canonical's `TR-0007`/`DEC-0015`/`DEC-0016` text is preserved. The accepted `DG-0007` registration is inserted into *Current Pending Work* and into the `DG-xxxx` list, the *Current active Study* line is extended to name it, and one iteration entry records this integration. |

### Not imported

The independent review and micro-verify documents
(`2026-09-29-mtrl-theory-audit-independent-review.md`,
`2026-09-30-mtrl-theory-audit-rereview.md`,
`2026-09-30-mtrl-dg0007-final-rereview.md`,
`2026-09-30-dg0007-final-micro-verify.md`) live in the reviewer's own worktree
and are **not** copied here; they remain independent evidence, as instructed.
The audit report's cross-reference to the independent review therefore points
outside this branch by design.

## The micro-verify MEDIUM, fixed here

The independent micro-verify recorded one non-blocking MEDIUM. The exit-2 report
was assembled as

```python
payload["detail"] = json.loads(_bounded(json.dumps(error.detail)))
```

— bounding the *serialized* JSON at 240 characters and parsing the truncated
string back. Truncation is not valid JSON by construction, so any `BadEvidence`
whose detail serialized longer than 240 characters raised `JSONDecodeError` from
inside the `except BadEvidence` handler, escaped `main()` as a traceback and
exited `1`: a malformed artifact presented as a *scientific* gate failure, which
is exactly what the exit contract forbids.

Reproduced against the accepted checker, then fixed:

| Case | Accepted checker | Integrated checker |
|---|---|---|
| arbitrary text/bytes checkpoint (long run path) | `1` + traceback | `2`, no traceback |
| foreign identity value larger than the bound | `1` + `JSONDecodeError` | `2`, no traceback |
| foreign identity integer larger than the bound | `1` + `JSONDecodeError` | `2`, no traceback |
| near-empty identity record | `1` + `JSONDecodeError` | `2`, no traceback |

The fix bounds the detail **by value** before serialization (`_bounded_detail`):
strings are truncated to 240 characters, containers to 16 entries and 3 levels
below the root, rendered leaves to 4096 characters in total, and an oversized
integer to `<int with N bits>` rather than its full decimal expansion.
Unrecognized objects become `<TypeName>`, never `str(obj)`, so no tensor or whole
object is ever materialized. The result is plain JSON by construction, the
`json.dumps` in `main` cannot fail on it, and field-level diagnostics are
preserved — the near-empty case still names all nine missing fields, and the
foreign-value case still reports `{"mismatched": {field: {expected, found}}}`.

Regression tests for the four classes are in
`research/tests/test_mtrl_runtime_faithfulness_checker.py`, class
`BadInputDiagnosticsAreBounded`; each asserts exit `2`, a bounded JSON
diagnostic, that the case genuinely exceeded the 240-character bound (so the
test cannot go vacuous), and — as a subprocess — exit `2` with no `Traceback` on
stderr.

## Frozen protocol — preserved, not restated

Study `DG-0007`; independent variable `model.normalize_w` (`true` historical,
`false` corrected); screen seed `42`; three arms (normalization-corrected MTRL,
historical MTRL control, matched wavCSE baseline); confirmation seeds `0,1,2,3,4`
(15 runs); all 25 layers; layer pooling `smp` 0.5; LOSO a conditional escalation
with a separate, **unmeasured** budget. Ω mathematics, λ terms, optimizer,
learning rate, epochs, batch size, datasets, embeddings, metrics, thresholds and
decision rules are untouched by the integration. The arm is
**normalization-corrected MTRL — not fully faithful Zhang & Yeung MTRL**: only
the audit's D2 is removed; D1, D3, D4, D6 and D7 remain.

## Blocked

No authorization envelope covers DG-0007, and the only envelope on the record
(`authorizations/TR-0007.yaml`, `DEC-0016`) excludes it **by name**: "no TR-0008,
no DG-0007". Nothing may be provisioned, submitted or launched for this study
until the researcher answers the authorization question and the
`VARIANT_BENCHMARK_PROTOCOL.md` §2 control question recorded in the study's
`NOTE.md`.

## Recorded discrepancies not resolved here

1. `research/STATE.md`'s `TR-0007` bullet still calls the screen commit
   "local-only (``origin`` publishes … at ``05fa10c``)" and still lists the λ₂
   rule and the compute envelope as the two outstanding human inputs, while the
   same file's later iteration text, `STUDIES.jsonl` and
   `authorizations/TR-0007.yaml` all record the publication and `DEC-0016`.
   Canonical-internal staleness, unrelated to DG-0007; left untouched rather
   than silently rewritten.
2. The audit report cross-references
   `audits/2026-09-29-mtrl-theory-audit-independent-review.md`, which exists only
   in the reviewer's worktree (see *Not imported*).
3. `STUDIES.jsonl`'s `DG-0007.independent_variable` says the two MTRL configs are
   "otherwise identical apart from run output directories"; `research.method`
   also differs, and must, because it is the field that separates the two arms
   in `ARC_RUN_IDENTITY` (`test_dg0007_run_identity.py` asserts exactly that).
   The scientific difference remains exactly `model.normalize_w`.
