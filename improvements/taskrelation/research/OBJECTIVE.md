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
  mechanism_requires_diagnostic_evidence: true
  mechanism_requires_published_method: true
  allow_arbitrary_architecture_generation: false

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