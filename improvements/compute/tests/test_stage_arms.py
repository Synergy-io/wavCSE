"""Per-stage arm selection: the bounded JobSpec extension for DG-0008.

The bug this covers: the plan model multiplied every declared arm across every
stage's seeds, so "200 matrix jobs plus exactly one same-arm repeat" could not be
expressed without either repeating the repeat across five seeds or encoding
experiment topology as worker concurrency. A stage may now name the arms it
expands; a stage that names none keeps the old behaviour exactly.

The DG-0008 fixture below encodes the approved topology (2 cells x 2 arm classes
x 10 LOSO folds x 5 seeds = 200, plus one `si_er`/`control`/fold-0/seed-0 repeat).
It is a test fixture for the *schema*, not DG-0008's committed compute plan:
`studies/DG-0008/compute/plan.json` remains unwritable until the controller-time
`worker` binding exists (see that Study's `PLAN.md` blocker 5). No existing
planner test is modified.
"""

import unittest

from improvements.compute import jobspec, run_study, worker_stage
from improvements.compute.errors import ConfigurationError
from improvements.compute.tests.fakes import ComputeTestCase, sample_plan

COMMIT = "a" * 40
CELLS = ("ks_er", "si_er")
ARM_CLASSES = ("pair", "control")
FOLDS = tuple(range(10))
SEEDS = (0, 1, 2, 3, 4)
CONFIG = "configs/dg0008.yaml"


def matrix_arm_name(cell, arm_class, fold):
    return "%s_%s_f%d" % (cell, arm_class, fold)


def dg0008_plan():
    """The approved DG-0008 topology, expressed with a per-stage arm selection."""

    arms = []
    for cell in CELLS:
        for arm_class in ARM_CLASSES:
            for fold in FOLDS:
                arms.append({
                    "arm": matrix_arm_name(cell, arm_class, fold),
                    "method": "%s_%s" % (cell, arm_class),
                    "config": CONFIG,
                    "argv": ["python", "run_dg0008.py", "--cell", cell,
                             "--arm", arm_class, "--fold", str(fold),
                             "--config", "{config}", "--seed", "{seed}"],
                })
    arms.append({
        "arm": "si_er_control_f0_repeat",
        "method": "si_er_control",
        "config": CONFIG,
        "argv": ["python", "run_dg0008.py", "--cell", "si_er", "--arm", "control",
                 "--fold", "0", "--determinism-repeat",
                 "--config", "{config}", "--seed", "{seed}"],
    })
    matrix_names = [arm["arm"] for arm in arms if arm["arm"] != "si_er_control_f0_repeat"]
    return {
        "schema_version": 1,
        "study": "DG-0008",
        "repository": "https://github.com/Synergy-io/wavCSE.git",
        "task_type": "ks_si_er",
        "environment_secrets": ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"],
        "timeout_seconds": 3600,
        "device_index": 0,
        "arms": arms,
        "stages": {
            "stage1_screen": {"seeds": list(SEEDS), "arms": matrix_names},
            "determinism_repeat": {"seeds": [0], "arms": ["si_er_control_f0_repeat"]},
        },
        "outputs": [
            {"name": "checkpoint_best", "kind": "checkpoint", "tag": "best",
             "required": True},
            {"name": "eval_metrics_opt", "kind": "results_file", "required": True},
        ],
        "worker": {
            "gpu_type": "NVIDIA L4",
            "cloud": "SECURE",
            "gpu_count": 1,
            "image": "runpod/pytorch:example",
            "container_disk_gb": 100,
        },
    }


class StageSelectionValidationTests(unittest.TestCase):
    def test_the_dg0008_topology_validates(self):
        jobspec.validate_plan(dg0008_plan())

    def test_unknown_arm_reference_fails_closed(self):
        plan = dg0008_plan()
        plan["stages"]["stage1_screen"]["arms"].append("does_not_exist_f0")
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(plan)

    def test_empty_selection_fails(self):
        plan = dg0008_plan()
        plan["stages"]["stage1_screen"]["arms"] = []
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(plan)

    def test_non_list_selection_fails(self):
        plan = dg0008_plan()
        plan["stages"]["stage1_screen"]["arms"] = "ks_er_pair_f0"
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(plan)

    def test_repeated_arm_in_a_selection_fails(self):
        plan = dg0008_plan()
        plan["stages"]["stage1_screen"]["arms"] = ["ks_er_pair_f0", "ks_er_pair_f0"]
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(plan)

    def test_unknown_stage_key_fails(self):
        plan = dg0008_plan()
        plan["stages"]["stage1_screen"]["concurrency"] = 1
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(plan)

    def test_a_stage_without_a_selection_still_validates(self):
        jobspec.validate_plan(sample_plan())


class BackwardCompatibilityTests(unittest.TestCase):
    def test_old_plans_expand_identically(self):
        plan = sample_plan()
        self.assertEqual([("mssl", 42)], run_study.plan_jobs(plan, "screen"))
        self.assertEqual([("mssl", 0), ("mssl", 1)], run_study.plan_jobs(plan, "confirm"))

    def test_stage_arm_names_defaults_to_every_declared_arm(self):
        plan = sample_plan()
        self.assertEqual(["mssl"], jobspec.stage_arm_names(plan, "screen"))
        self.assertEqual(["mssl"], jobspec.stage_arm_names(plan, "confirm"))

    def test_a_selection_of_every_arm_is_equivalent_to_no_selection(self):
        plan = sample_plan()
        implicit = run_study.plan_jobs(plan, "screen")
        plan["stages"]["screen"]["arms"] = ["mssl"]
        jobspec.validate_plan(plan)
        self.assertEqual(implicit, run_study.plan_jobs(plan, "screen"))

    def test_unknown_stage_errors(self):
        plan = sample_plan()
        with self.assertRaises(ConfigurationError):
            jobspec.stage_arm_names(plan, "nope")


class StageSelectionIdentityTests(ComputeTestCase):
    """Deterministic identity must not move when a stage selects arms."""

    def setUp(self):
        super(StageSelectionIdentityTests, self).setUp()
        self.make_repo()
        self.plan = jobspec.load_plan(self.write_plan(sample_plan()))
        self.commit()

    def build(self, plan=None, **overrides):
        kwargs = dict(stage="screen", arm="mssl", seed=42, commit=COMMIT,
                      scope="TR-0007", envelope_digest="digest")
        kwargs.update(overrides)
        return jobspec.build_spec(plan or self.plan, **kwargs)

    def test_generation_is_still_byte_identical(self):
        self.assertEqual(jobspec.render(self.build()), jobspec.render(self.build()))

    def test_selection_does_not_change_job_identity(self):
        implicit = self.build()
        explicit_plan = sample_plan()
        explicit_plan["stages"]["screen"]["arms"] = ["mssl"]
        explicit = jobspec.load_plan(self.write_plan(explicit_plan, name="plan2.json"))
        self.commit()
        other = self.build(plan=explicit)
        self.assertEqual(jobspec.job_key_of(implicit), jobspec.job_key_of(other))
        self.assertEqual(implicit["name"], other["name"])

    def test_an_arm_the_stage_does_not_select_fails_closed(self):
        plan = sample_plan()
        plan["arms"].append({"arm": "other", "method": "other", "config": CONFIG,
                             "argv": ["python", "-m", "x", "--config", "{config}",
                                      "--seed", "{seed}"]})
        plan["stages"] = {"screen": {"seeds": [42], "arms": ["mssl"]}}
        loaded = jobspec.load_plan(self.write_plan(plan, name="plan3.json"))
        self.commit()
        with self.assertRaises(ConfigurationError):
            self.build(plan=loaded, arm="other")


class WorkerStageGuardTests(ComputeTestCase):
    def setUp(self):
        super(WorkerStageGuardTests, self).setUp()
        self.make_repo()
        self.plan_path = self.write_plan(dg0008_plan())
        self.commit()

    def run_wrapper(self, stage, arm, seed):
        return worker_stage.main(["--plan", self.plan_path, "--stage", stage,
                                  "--arm", arm, "--seed", str(seed),
                                  "--outputs-root", "outputs/probe"])

    def test_wrapper_refuses_an_arm_the_stage_does_not_select(self):
        self.assertEqual(2, self.run_wrapper("determinism_repeat", "ks_er_pair_f0", 0))

    def test_wrapper_refuses_a_seed_the_stage_does_not_carry(self):
        self.assertEqual(2, self.run_wrapper("determinism_repeat",
                                             "si_er_control_f0_repeat", 3))


class DG0008ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.plan = dg0008_plan()
        jobspec.validate_plan(self.plan)

    def jobs(self):
        rows = []
        for stage in self.plan["stages"]:
            for arm, seed in run_study.plan_jobs(self.plan, stage):
                rows.append((stage, arm, seed))
        return rows

    def test_expands_to_exactly_201_jobs(self):
        self.assertEqual(201, len(self.jobs()))

    def test_exactly_200_are_matrix_members(self):
        matrix = [row for row in self.jobs() if row[0] == "stage1_screen"]
        self.assertEqual(200, len(matrix))
        names = {row[1] for row in matrix}
        self.assertEqual(40, len(names))
        for arm in names:
            self.assertEqual(5, sum(1 for row in matrix if row[1] == arm))
        for seed in SEEDS:
            self.assertEqual(40, sum(1 for row in matrix if row[2] == seed))

    def test_exactly_one_job_is_the_approved_same_arm_repeat(self):
        repeats = [row for row in self.jobs() if row[0] == "determinism_repeat"]
        self.assertEqual([("determinism_repeat", "si_er_control_f0_repeat", 0)], repeats)

    def test_the_repeat_is_not_multiplied_across_five_seeds(self):
        repeats = [row for row in self.jobs() if row[1] == "si_er_control_f0_repeat"]
        self.assertEqual(1, len(repeats))
        self.assertEqual({0}, {row[2] for row in repeats})

    def test_no_unintended_job_exists(self):
        rows = self.jobs()
        names = [jobspec.job_name(self.plan["study"], stage, arm, seed)
                 for stage, arm, seed in rows]
        self.assertEqual(201, len(set(names)))
        declared = set(jobspec.arm_names(self.plan))
        self.assertTrue({arm for _, arm, _ in rows} <= declared)

    def test_job_identity_is_deterministic(self):
        first = self.jobs()
        second = self.jobs()
        self.assertEqual(first, second)
        keys = [jobspec.job_key("DG-0008", "DG-0008", stage, arm, seed, COMMIT)
                for stage, arm, seed in first]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
