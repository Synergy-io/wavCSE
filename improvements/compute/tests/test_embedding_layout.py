"""B08: the verified artifact must reach the loader, or the job must not run.

Every test builds a *fresh worker workspace*: a job directory holding the artifacts the
control plane materialized, a checkout holding the plan's own inputs file, and — the
point of the exercise — an unrelated, stale embedding tree under the worker's ``$HOME``,
exactly the tree a careless run would silently read instead. The mapping must resolve to
the declared, digest-verified input and to nothing else.
"""

import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest

from improvements import embedding_root
from improvements.compute import embedding_layout, jobspec, worker_stage
from improvements.compute.tests.fakes import REPO_ROOT, ComputeTestCase

MTRL_CONFIG = os.path.join("improvements", "taskrelation", "01-mtrl", "mtrl_config.yml")
DATASETS = ("speechcommand", "voxceleb", "iemocap")
TASKS = "ks_si_er"

# A tiny stand-in for one training run: it reports the run identity the wrapper reads,
# records the embedding root it was handed, and writes the declared outputs.
TRAINING_SCRIPT = r"""
import json, os, sys
seed = sys.argv[1]
results = os.path.join(os.environ["ARC_RESULTS"], "run_s{}".format(seed))
checkpoints = os.path.join(os.environ["ARC_CHECKPOINTS"], "ckpt_s{}".format(seed))
os.makedirs(results, exist_ok=True)
os.makedirs(checkpoints, exist_ok=True)
# What the loader would expand: recorded so the test can prove which root the run saw.
with open(os.path.join(results, "resolved_root.txt"), "w") as handle:
    handle.write(os.environ.get("WAVCSE_ROOT_EMB_PATH", "<unset>"))
with open(os.path.join(results, "eval_metrics_opt.txt"), "w") as handle:
    handle.write("loss_all=1.0 | acc_all=0.5\n")
    for task in ("ks", "si", "er"):
        handle.write("{} | loss=1.0 | acc=0.5 | samples=10\n".format(task))
with open(os.path.join(checkpoints, "checkpoint_best.pth"), "wb") as handle:
    handle.write(b"weights")
record = {"results_dir": results, "checkpoints_dir": checkpoints,
          "run_id": os.path.basename(results),
          "checkpoint_run_id": os.path.basename(checkpoints)}
destination = os.environ.get("ARC_RUN_IDENTITY_FILE", "")
if destination:
    with open(destination, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
"""


class LayoutTestCase(ComputeTestCase):
    def setUp(self):
        super(LayoutTestCase, self).setUp()
        self.job_directory = os.path.join(self.home, "job-0000000000000000")
        self.checkout = os.path.join(self.home, "checkout")
        os.makedirs(self.checkout, exist_ok=True)
        # The checkout carries the loader's own contract, so the mapping is derived from
        # it rather than from a restatement of it.
        os.symlink(os.path.join(REPO_ROOT, "downstream"),
                   os.path.join(self.checkout, "downstream"))
        self.home_directory = os.path.join(self.home, "worker-home")
        self.stale_root = os.path.join(self.home_directory, "embedding")
        self.write_stale_tree()
        self.training_script = os.path.join(self.checkout, "training.py")
        with open(self.training_script, "w", encoding="utf-8") as handle:
            handle.write(TRAINING_SCRIPT)
        self.objects = {}
        self.requirements = []
        self.write_inputs_file()
        self.materialize()

    # ------------------------------------------------------------------ the worker

    def dataset_bytes(self, dataset, *, files=("a.wav", "b.wav")):
        """One dataset archive: a plain TAR of the dataset directory."""

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as bundle:
            for relative in files:
                payload = "{}:{}\n".format(dataset, relative).encode("utf-8")
                info = tarfile.TarInfo("{}/{}_wavlm_large_mean.pt".format(dataset, relative))
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
        return buffer.getvalue()

    def write_stale_tree(self):
        """The unrelated tree a careless run would read: present, plausible, wrong."""

        directory = os.path.join(self.stale_root, "wavlm_large", "mean", "speechcommand")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "STALE_wavlm_large_mean.pt"), "wb") as handle:
            handle.write(b"embeddings from another machine, never verified")

    def write_inputs_file(self, *, artifacts=None):
        artifacts = artifacts or {
            "embeddings/wavlm_large/mean/{}.tar".format(dataset): self.dataset_bytes(dataset)
            for dataset in DATASETS
        }
        requirements = []
        for artifact, payload in sorted(artifacts.items()):
            destination = "embeddings/{}".format(artifact.rsplit("/", 1)[-1])
            requirements.append({
                "artifact": artifact,
                "destination": destination,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "required": True,
            })
            self.objects[destination] = payload
        self.requirements = requirements
        path = os.path.join(self.checkout, "inputs.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "requirements": requirements}, handle)
        self.inputs_relative = "inputs.json"

    def materialize(self, *, contents=None):
        """Place each declared input where the control plane would have put it."""

        for requirement in self.requirements:
            destination = os.path.join(self.job_directory, "inputs",
                                       requirement["destination"])
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            payload = (contents or {}).get(requirement["artifact"])
            if payload is None:
                payload = self.objects[requirement["destination"]]
            with open(destination, "wb") as handle:
                handle.write(payload)

    def plan(self, **overrides):
        layout = {
            "root": "embedding",
            "datasets": [
                {"dataset": dataset,
                 "input": "embeddings/wavlm_large/mean/{}.tar".format(dataset)}
                for dataset in DATASETS
            ],
        }
        document = {
            "schema_version": 1,
            "study": "TR-0007",
            "repository": "https://github.com/example/wavCSE.git",
            "task_type": TASKS,
            "timeout_seconds": 3600,
            "device_index": 0,
            "environment_secrets": ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"],
            "arms": [{
                "arm": "mtrl",
                "method": "mtrl",
                "config": os.path.join(REPO_ROOT, MTRL_CONFIG),
                "argv": [sys.executable, self.training_script, "{seed}"],
            }],
            "stages": {"screen": {"seeds": [42]}},
            "outputs": [
                {"name": "checkpoint_best", "kind": "checkpoint", "tag": "best",
                 "required": True},
                {"name": "eval_metrics_opt", "kind": "results_file", "required": True},
            ],
            "inputs_file": self.inputs_relative,
            "worker": {"gpu_type": "NVIDIA GeForce RTX 4090", "cloud": "SECURE",
                       "gpu_count": 1, "image": "runpod/pytorch:example",
                       "container_disk_gb": 60},
            "embedding_layout": layout,
        }
        document.update(overrides)
        path = os.path.join(self.home, "plan.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
        return jobspec.load_plan(path)

    def prepare(self, plan=None, **kwargs):
        return embedding_layout.prepare(
            plan or self.plan(), stage="screen", arm="mtrl", seed=42,
            checkout_root=self.checkout, job_directory=self.job_directory,
            commit="a" * 40, job_id="job-0000000000000000", **kwargs
        )

    # ------------------------------------------------------------------ assertions

    def prepared_tree(self, dataset):
        return os.path.join(self.job_directory, "embedding", "wavlm_large", "mean",
                            dataset)

    def tree_files(self, root):
        found = []
        for directory, _dirs, files in os.walk(root):
            for name in files:
                found.append(os.path.relpath(os.path.join(directory, name), root))
        return sorted(found)


class FreshWorkerMappingTests(LayoutTestCase):
    def test_the_prepared_layout_is_exactly_the_verified_archive(self):
        marker = self.prepare()

        for dataset in DATASETS:
            self.assertEqual(
                self.tree_files(self.prepared_tree(dataset)),
                ["a.wav_wavlm_large_mean.pt", "b.wav_wavlm_large_mean.pt"],
            )
        self.assertEqual(sorted(entry["dataset"] for entry in marker["datasets"]),
                         sorted(DATASETS))
        for entry in marker["datasets"]:
            requirement = next(item for item in self.requirements
                               if item["artifact"] == entry["artifact"])
            self.assertEqual(entry["sha256"], requirement["sha256"])

    def test_the_stale_worker_tree_is_never_reachable_from_the_prepared_root(self):
        self.prepare()

        prepared = os.path.join(self.job_directory, "embedding")
        names = self.tree_files(prepared)
        self.assertNotIn("STALE_wavlm_large_mean.pt", " ".join(names))
        self.assertFalse(os.path.exists(
            os.path.join(prepared, "wavlm_large", "mean", "speechcommand",
                         "STALE_wavlm_large_mean.pt")))

    def test_every_dataset_the_protocol_loads_is_present(self):
        """ks_si_er needs all three trees, not one toy path."""

        self.assertEqual(embedding_layout.required_datasets(self.plan()),
                         ["speechcommand", "voxceleb", "iemocap"])
        marker = self.prepare()
        for dataset in embedding_layout.required_datasets(self.plan()):
            extracted = next(entry for entry in marker["datasets"]
                             if entry["dataset"] == dataset)
            self.assertTrue(os.path.isdir(extracted["extracted_to"]))

    def test_the_loader_resolves_the_prepared_root_and_not_the_stale_one(self):
        """The real loader, constructed exactly as the arm constructs it."""

        marker = self.prepare()
        prepared = marker["loader_root"]
        environment = {embedding_root.ENV_ROOT: prepared}
        resolved = embedding_root.resolve_root("~/embedding", environ=environment)

        self.assertEqual(resolved, prepared)
        self.assertFalse(resolved.startswith(self.home_directory))

        load_embedding = self.real_loader(resolved)
        self.assertEqual(load_embedding.emb_speechcommand_path,
                         os.path.join(prepared, "wavlm_large", "mean", "speechcommand"))
        self.assertEqual(load_embedding.emb_voxceleb_path,
                         os.path.join(prepared, "wavlm_large", "mean", "voxceleb"))
        self.assertEqual(load_embedding.emb_iemocap_path,
                         os.path.join(prepared, "wavlm_large", "mean", "iemocap"))
        self.assertNotIn(self.home_directory, load_embedding.root_emb_path)

    def real_loader(self, root):
        """Construct the frozen downstream loader, as `run_improvements` does."""

        sys.path.insert(0, os.path.join(REPO_ROOT, "downstream"))
        from dataset.load_embedding import LoadEmbedding

        return LoadEmbedding(
            root_emb_path=root,
            root_data_path=os.path.join(self.home, "data"),
            upstream_model_type="wavlm_large",
            frame_pooling_type="mean",
            frame_pooling_param=None,
            transformer_layer_array=list(range(25)),
            device="cpu",
        )

    def test_an_unprepared_root_is_refused_rather_than_used(self):
        """A root without the layout marker is never accepted, and never replaced."""

        environment = {embedding_root.ENV_ROOT: self.stale_root}

        with self.assertRaises(embedding_root.EmbeddingRootError) as caught:
            embedding_root.resolve_root("~/embedding", environ=environment)

        self.assertIn("no prepared-layout marker", str(caught.exception))

    def test_without_the_override_the_configured_root_is_returned_unchanged(self):
        self.assertEqual(embedding_root.resolve_root("~/embedding", environ={}),
                         os.path.expanduser("~/embedding"))


class LayoutRefusalTests(LayoutTestCase):
    def test_materialized_bytes_that_do_not_match_the_declared_digest_refuse(self):
        self.materialize(contents={
            requirement["artifact"]: b"not the verified bytes"
            for requirement in self.requirements
        })

        with self.assertRaises(embedding_layout.EmbeddingLayoutError) as caught:
            self.prepare()

        self.assertIn("not the declared", str(caught.exception))
        self.assertFalse(os.path.exists(os.path.join(self.job_directory, "embedding")))

    def test_an_archive_containing_another_dataset_refuses(self):
        # Overwrite the voxceleb input with iemocap's bytes, and declare the digest it
        # really has, so only the archive's *contents* are wrong.
        payload = self.dataset_bytes("iemocap")
        target = next(item for item in self.requirements
                      if item["artifact"].endswith("voxceleb.tar"))
        target["sha256"] = hashlib.sha256(payload).hexdigest()
        target["size_bytes"] = len(payload)
        with open(os.path.join(self.checkout, "inputs.json"), "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "requirements": self.requirements}, handle)
        self.materialize(contents={target["artifact"]: payload})

        with self.assertRaises(embedding_layout.EmbeddingLayoutError) as caught:
            self.prepare()

        self.assertIn("belongs to another dataset", str(caught.exception))

    def test_an_archive_with_an_escaping_member_refuses(self):
        payload = self.dataset_bytes("speechcommand")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as bundle:
            member = tarfile.TarInfo("speechcommand/../../escaped.pt")
            member.size = 4
            bundle.addfile(member, io.BytesIO(b"nope"))
        payload = buffer.getvalue()
        target = next(item for item in self.requirements
                      if item["artifact"].endswith("speechcommand.tar"))
        target["sha256"] = hashlib.sha256(payload).hexdigest()
        target["size_bytes"] = len(payload)
        with open(os.path.join(self.checkout, "inputs.json"), "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "requirements": self.requirements}, handle)
        self.materialize(contents={target["artifact"]: payload})

        with self.assertRaises(embedding_layout.EmbeddingLayoutError) as caught:
            self.prepare()

        self.assertIn("unsafe member", str(caught.exception))
        self.assertFalse(os.path.exists(os.path.join(self.home, "escaped.pt")))

    def test_a_layout_missing_a_required_dataset_refuses(self):
        plan = self.plan(embedding_layout={
            "root": "embedding",
            "datasets": [{"dataset": "speechcommand",
                          "input": "embeddings/wavlm_large/mean/speechcommand.tar"}],
        })

        with self.assertRaises(Exception) as caught:
            self.prepare(plan)

        self.assertIn("does not cover", str(caught.exception))

    def test_a_layout_naming_an_undeclared_artifact_refuses(self):
        layout = {
            "root": "embedding",
            "datasets": [{"dataset": dataset,
                          "input": "embeddings/other/{}.tar".format(dataset)}
                         for dataset in DATASETS],
        }
        with self.assertRaises(Exception) as caught:
            self.prepare(self.plan(embedding_layout=layout))

        self.assertIn("does not declare it", str(caught.exception))

    def test_a_second_preparation_of_the_same_job_refuses(self):
        self.prepare()

        with self.assertRaises(embedding_layout.EmbeddingLayoutError):
            self.prepare()

    def test_the_plan_schema_rejects_a_layout_with_no_inputs_file(self):
        with self.assertRaises(Exception):
            self.plan(embedding_layout={"root": "embedding", "datasets": []})


class JobWrapperEndToEndTests(LayoutTestCase):
    """The wrapper itself: prepare, export, run, and record the mapping it used."""

    def run_job(self, plan):
        environment = dict(os.environ)
        environment.update({
            "WAVCSE_JOB_DIRECTORY": self.job_directory,
            "WAVCSE_JOB_ID": "job-0000000000000000",
            "WAVCSE_JOB_COMMIT": "a" * 40,
            "ARC_RESULTS": os.path.join(self.job_directory, "results"),
            "ARC_CHECKPOINTS": os.path.join(self.job_directory, "checkpoints"),
            "HOME": self.home_directory,
        })
        previous = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        cwd = os.getcwd()
        os.chdir(self.checkout)
        try:
            return worker_stage.main([
                "--plan", plan["_path"], "--stage", "screen", "--arm", "mtrl",
                "--seed", "42", "--outputs-root", "outputs/mtrl_s42",
            ])
        finally:
            os.chdir(cwd)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_the_run_consumes_the_prepared_root_not_the_stale_tree(self):
        code = self.run_job(self.plan())

        self.assertEqual(code, 0)
        prepared = os.path.join(self.job_directory, "embedding")
        resolved_file = os.path.join(self.job_directory, "results", "run_s42",
                                     "resolved_root.txt")
        with open(resolved_file, "r", encoding="utf-8") as handle:
            resolved = handle.read().strip()
        self.assertEqual(resolved, prepared)
        self.assertNotEqual(resolved, self.stale_root)

        outputs = os.path.join(self.job_directory, "outputs", "mtrl_s42")
        with open(os.path.join(outputs, "MANIFEST.json"), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["study"], "TR-0007")
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["commit"], "a" * 40)
        self.assertEqual(manifest["training_exit_code"], 0)
        self.assertIsNotNone(manifest["staged_at"])
        declared = manifest["embedding_layout"]
        self.assertEqual(declared["root"], prepared)
        self.assertEqual(declared["frame_pool_id"], "mean")
        self.assertEqual(declared["upstream_model_type"], "wavlm_large")
        self.assertEqual(sorted(entry["dataset"] for entry in declared["datasets"]),
                         sorted(DATASETS))
        for entry in declared["datasets"]:
            self.assertTrue(os.path.isdir(os.path.join(
                prepared, "wavlm_large", "mean", entry["dataset"])))
        staged = {item["target"] for item in manifest["files"]}
        self.assertEqual(staged, {"checkpoint_best.pth", "eval_metrics_opt.txt"})

    def test_a_job_that_cannot_prepare_its_layout_runs_nothing(self):
        self.materialize(contents={
            requirement["artifact"]: b"wrong bytes"
            for requirement in self.requirements
        })

        code = self.run_job(self.plan())

        self.assertEqual(code, 4)
        self.assertFalse(os.path.exists(
            os.path.join(self.job_directory, "results", "run_s42", "resolved_root.txt")))


if __name__ == "__main__":
    unittest.main()
