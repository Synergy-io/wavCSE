research_family: task_relation_learning

tasks:
  - ks
  - si
  - er

formal_starting_method:
  id: mtrl
  location: improvements/taskrelation/01-mtrl
  role: primary existing Task Relation Learning baseline method

method_scope:
  active: [mtrl]
  diagnostic_only: [lnp]
  archived: [gbc]
  quarantined_unvalidated: [tsm, pmr]

progression_gate:
  # Amended by DEC-0013 (2026-09-22, human): diagnostic-first sequencing is
  # deferred, so a named diagnostic is no longer a precondition for a mechanism
  # study. The published-method gate is unchanged.
  mechanism_requires_diagnostic_evidence: false
  mechanism_requires_published_method: true
  allow_arbitrary_architecture_generation: false
  amended_by: DEC-0013

optimization:
  type: multi_objective

screening:
  seeds: 1

confirmation:
  seeds: [0, 1, 2, 3, 4]

requirements:
  no_material_regression_pp: 0.20

promotion:
  require_multiseed: true
  require_matched_protocol: true

er:
  serious_claim_requires_loso: true

plateau:
  consecutive_failed_studies: 5

literature_mode_on_plateau: true