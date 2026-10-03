"""The Literature Agent vertical slice stays read-only and reference-bounded.

These tests cover the parts a unit test can hold: the agent's declared authority,
its skill, the tool adapter's declared surface, and the deterministic modules'
mutation behaviour. Model reasoning quality is evaluated by real runs, not here.
"""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
RESEARCH_DIR = REPO_ROOT / "improvements" / "taskrelation" / "research"
AGENT_PATH = REPO_ROOT / ".omp" / "agents" / "literature-reviewer.md"
TOOLS_PATH = REPO_ROOT / ".omp" / "tools" / "literature.ts"
MCP_PATH = REPO_ROOT / ".omp" / "mcp.json"
TRANSCRIPT_CHECKER = REPO_ROOT / "scripts" / "agents" / "literature_agent_transcript.py"

# The approved evidence surface: the harness primitive plus four capabilities.
REQUIRED_LITERATURE_TOOLS = (
    "literature_resolve", "literature_query", "literature_read", "literature_primary",
)
# Historical recall is not an evidence capability in any spelling.
RECALL_TOOL_NEEDLES = ("deja", "recall", "retain", "reflect", "history", "memory", "mcp__")
# The native project OMP settings file; the recall *content* channel (a
# user-scope extension) is disabled here, not in mcp.json.
OMP_CONFIG_PATH = REPO_ROOT / ".omp" / "config.yml"
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "wavcse-literature-review"
SKILL_PATH = SKILL_DIR / "SKILL.md"

# The authority contract for V1: read-only, reference-bounded, no shell, no
# mutation, no infrastructure, no credential access.
FORBIDDEN_AGENT_TOOLS = (
    "bash", "write", "edit", "python", "notebook", "browser", "computer",
    "task", "glob", "grep", "find", "lsp", "web_search", "ask",
    # The evidence surface is the retained corpus only: a generic reader would
    # let the literature agent answer questions about our own code and state.
    "read",
)
ALLOWED_AGENT_TOOLS = (
    "literature_resolve", "literature_query", "literature_read",
    "literature_primary", "literature_discover", "literature_record", "yield",
)
FORBIDDEN_TOOL_SOURCE = (
    "child_process", "spawnSync", "shell: true", "boto3", "botocore",
    "aws_access", "secret_access", "session_token", "s3://", "AWS_",
)
EXPECTED_TOOL_NAMES = (
    "literature_discover", "literature_primary", "literature_query",
    "literature_read", "literature_record", "literature_resolve",
)

def parse_frontmatter(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise AssertionError("{} has no frontmatter block".format(path.name))
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return lines[1:index], lines[index + 1:]
    raise AssertionError("{} frontmatter is never closed".format(path.name))


def run_module(module, *argv):
    result = subprocess.run(
        [sys.executable, "-m", "improvements.taskrelation.research." + module, *argv],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result


class AgentDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries, body = parse_frontmatter(AGENT_PATH)
        cls.body = "\n".join(body)
        cls.fields = {}
        for line in cls.entries:
            key, _, value = line.partition(":")
            cls.fields[key.strip()] = value.strip().strip('"').strip("'")

    def test_agent_declares_identity_and_a_model_role(self):
        self.assertEqual(self.fields.get("name"), "literature-reviewer")
        self.assertTrue(self.fields.get("description"))
        self.assertTrue(self.fields.get("model", "").startswith("@"))

    def test_agent_grants_only_read_only_tools(self):
        declared = [
            name.strip() for name in self.fields["tools"].split(",") if name.strip()
        ]

        for forbidden in FORBIDDEN_AGENT_TOOLS:
            with self.subTest(tool=forbidden):
                self.assertNotIn(forbidden, declared)
        for name in declared:
            with self.subTest(tool=name):
                self.assertIn(name, ALLOWED_AGENT_TOOLS)

    def test_agent_autoloads_a_skill_that_exists(self):
        declared = self.fields.get("autoloadSkills", "")

        self.assertEqual(declared, SKILL_DIR.name)
        self.assertTrue(SKILL_PATH.is_file())

    def test_agent_forbids_decisions_experiments_and_state_changes(self):
        lowered = self.body.lower()

        for phrase in (
            "authorize", "decision", "findings.md", "decisions.md",
            "studies.jsonl", "manifest", "credential",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, lowered)


class CapabilityDeclarationDriftTests(unittest.TestCase):
    """INC-016: declaration, adapter surface and approved grant must not drift.

    The first real Literature Agent investigation observed `literature_discover`
    declared in the agent frontmatter but absent from the runtime grant. A
    declared-but-ungranted capability is a defect class of its own, so this test
    pins the agent's declaration to the adapter's exported surface and the
    approved constant; if any layer loses a capability the others still declare,
    it fails. Runtime reachability is proved separately by the transcript
    checker (`scripts/agents/literature_agent_transcript.py`), which reads the
    real `session_init.tools` of a run.
    """

    @classmethod
    def setUpClass(cls):
        entries, _ = parse_frontmatter(AGENT_PATH)
        fields = {}
        for line in entries:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip('"').strip("'")
        cls.declared = tuple(
            sorted(name.strip() for name in fields["tools"].split(",") if name.strip())
        )
        cls.adapter = tuple(
            sorted(re.findall(r'name:\s*"(literature_[a-z_]+)"', TOOLS_PATH.read_text(encoding="utf-8")))
        )
        cls.code = code_only(TOOLS_PATH.read_text(encoding="utf-8"))

    def test_agent_declaration_equals_the_adapter_surface(self):
        self.assertEqual(self.declared, tuple(sorted(EXPECTED_TOOL_NAMES)))
        self.assertEqual(self.adapter, tuple(sorted(EXPECTED_TOOL_NAMES)))

    def test_every_required_capability_is_declared_and_granted(self):
        for capability in REQUIRED_LITERATURE_TOOLS + ("literature_discover",):
            with self.subTest(capability=capability):
                self.assertIn(capability, self.declared)
                self.assertIn(capability, self.adapter)
                self.assertIn(capability, ALLOWED_AGENT_TOOLS)

    def test_adapter_query_operations_cover_the_intended_surface(self):
        for operation in (
            "list", "resolve", "paper_studies", "study_papers", "study",
            "assessment", "paper_claims", "claim", "synthesis_list", "synthesis",
        ):
            with self.subTest(operation=operation):
                self.assertIn('"{}"'.format(operation), self.code)
        # synthesis_list and synthesis resolve to the existing Python commands.
        self.assertIn('["literature_query", "syntheses"]', self.code)
        self.assertIn('"synthesis", params.synthesisId', self.code)

    def test_adapter_read_sources_cover_every_artifact_class(self):
        for source in ("card", "study", "survey", "synthesis"):
            with self.subTest(source=source):
                self.assertIn('"{}"'.format(source), self.code)

    def test_adapter_discovery_operations_cover_the_provider_surface(self):
        for operation in (
            "doi", "arxiv", "title", "search", "references", "citations", "providers",
        ):
            with self.subTest(operation=operation):
                self.assertIn('"{}"'.format(operation), self.code)


class SkillAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries, body = parse_frontmatter(SKILL_PATH)
        cls.body = "\n".join(body)
        cls.fields = {}
        for line in cls.entries:
            key, _, value = line.partition(":")
            cls.fields[key.strip()] = value.strip().strip('"').strip("'")

    def test_skill_name_matches_its_directory_and_has_a_bounded_description(self):
        self.assertEqual(self.fields.get("name"), SKILL_DIR.name)
        description = self.fields.get("description", "")
        self.assertTrue(description)
        self.assertLessEqual(len(description), 200)

    def test_skill_teaches_progressive_disclosure_and_evidence_levels(self):
        lowered = self.body.lower()

        for phrase in (
            "cheapest sufficient evidence", "card-derived", "study-derived",
            "primary", "never present card-derived",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, lowered)

    def test_skill_separates_claim_evidence_interpretation_and_implication(self):
        lowered = self.body.lower()

        for phrase in (
            "paper claim", "reported evidence", "your interpretation",
            "research implication",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, lowered)

    def test_skill_states_the_authority_limits(self):
        lowered = self.body.lower()

        for phrase in (
            "findings.md", "decisions.md", "studies.jsonl",
            "authoriz", "investigation-scoped", "cannot create or close",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, lowered)


def declared_parameters(code):
    """Collect parameter names from each `parameters: z.object({...})` block."""

    blocks = []
    current = None
    for line in code.splitlines():
        if "z.object({" in line:
            current = set()
            continue
        if current is None:
            continue
        if line.strip().startswith("})"):
            blocks.append(current)
            current = None
            continue
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9]*)\s*:", line)
        if match:
            current.add(match.group(1))
    return blocks


def code_only(source):
    """Strip comments so capability assertions test code, not prose."""

    without_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return "\n".join(
        line.split("//", 1)[0] for line in without_block.splitlines()
    )


class ToolAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TOOLS_PATH.read_text(encoding="utf-8")
        cls.code = code_only(cls.source)

    def test_adapter_exposes_exactly_the_semantic_surface(self):
        declared = sorted(re.findall(r'name:\s*"(literature_[a-z_]+)"', self.source))

        self.assertEqual(declared, list(EXPECTED_TOOL_NAMES))

    def test_adapter_declares_no_shell_infrastructure_or_credentials(self):
        for needle in FORBIDDEN_TOOL_SOURCE:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, self.code)
        # The only executable reached is the fixed Python interpreter, never a shell.
        self.assertNotIn("Bun.$", self.code)
        self.assertEqual(self.code.count("pi.exec("), 1)

    def test_adapter_exposes_recorded_claims_and_survey_reads(self):
        self.assertIn('"paper_claims"', self.code)
        self.assertIn('"claim"', self.code)
        self.assertIn('"literature_claims"', self.code)
        self.assertIn('"survey"', self.code)

    def test_adapter_exposes_the_canonical_assessment_query(self):
        # One canonical (investigation, paper) assessment read, part of the
        # existing query tool rather than a tool of its own.
        self.assertIn('"assessment"', self.code)
        self.assertIn('"assessment", params.studyId, params.paperId', self.code)
        self.assertEqual(len(re.findall(r'name:\s*"(literature_[a-z_]+)"', self.source)), 6)

    def test_adapter_exposes_primary_read_but_never_registration(self):
        # The agent may read a retained primary artifact by page, and may never
        # register one: registration is an operator-side act.
        self.assertIn('"read"', self.code)
        self.assertIn("--page", self.code)
        self.assertIn("--page-end", self.code)
        self.assertIn("--role", self.code)
        self.assertNotIn('"register"', self.code)

    def test_adapter_selects_a_primary_version_by_role_never_by_path(self):
        blocks = declared_parameters(self.code)
        declared = set().union(*blocks) if blocks else set()

        self.assertIn("role", declared)
        for forbidden in ("path", "key", "bucket", "url", "credential"):
            with self.subTest(param=forbidden):
                self.assertNotIn(forbidden, declared)

    def test_agent_declares_no_file_access_and_claim_first_provenance(self):
        lowered = "\n".join(parse_frontmatter(AGENT_PATH)[1]).lower()

        self.assertIn("no general file access", lowered)
        self.assertIn("paper_claims", lowered)
        self.assertIn("claim_ref", lowered)

    def test_adapter_declares_paper_addressed_parameters_only(self):
        blocks = declared_parameters(self.code)
        declared = set().union(*blocks) if blocks else set()

        self.assertEqual(len(blocks), 6, "one parameter block per exposed tool")
        for name in ("paperId", "studyId", "source", "operation"):
            with self.subTest(param=name):
                self.assertIn(name, declared)
        for forbidden in ("path", "filePath", "objectKey", "bucket", "command", "url"):
            with self.subTest(param=forbidden):
                self.assertNotIn(forbidden, declared)


class MutationBoundaryTests(unittest.TestCase):
    """Reader modules never write; the scoped writer writes only its bounded set."""

    READ_ONLY_MODULES = (
        "literature_catalog.py", "literature_query.py", "literature_read.py",
        "literature_claims.py", "literature_assessment.py",
    )
    WRITE_PATTERNS = ("write_text(", "open(", ".unlink(", "mkdir(", "os.replace(")

    def test_read_only_modules_contain_no_write_operations(self):
        for name in self.READ_ONLY_MODULES:
            source = (RESEARCH_DIR / name).read_text(encoding="utf-8")
            for pattern in self.WRITE_PATTERNS:
                with self.subTest(module=name, pattern=pattern):
                    self.assertNotIn(pattern, source)

    def test_scoped_writer_cannot_reach_other_research_state(self):
        """literature_record is the one Literature Agent mutator, and it is bounded."""

        source = (RESEARCH_DIR / "literature_record.py").read_text(encoding="utf-8")

        self.assertIn("OUTSIDE_DELEGATED_SCOPE", source)
        self.assertIn("assessments.jsonl", source)
        self.assertIn("claims.jsonl", source)
        # No write-bearing line reaches a Study, finding, decision, backlog,
        # proposal, authorization or admission ledger, and it never shells out,
        # admits a paper or acquires an artifact.
        write_lines = [
            line
            for line in source.splitlines()
            if any(
                call in line
                for call in (
                    "write_text(", "open(", "os.replace(", "os.unlink(", "unlink(",
                    "mkdir(", "NamedTemporaryFile", "subprocess", "os.system",
                    "os.popen", "boto3", "botocore", "urllib.request", "requests",
                )
            )
        ]
        for forbidden in (
            "STUDIES.jsonl", "FINDINGS", "FAILURES", "DECISIONS", "BACKLOG",
            "proposals", "authorizations", "canonicalizations",
            "literature_admit", "literature_acquire",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(
                    any(forbidden in line for line in write_lines),
                    "a writer line reaches {!r}".format(forbidden),
                )

    def test_lifecycle_owns_study_registry_writes_and_nothing_else(self):
        source = (RESEARCH_DIR / "literature_investigation.py").read_text(encoding="utf-8")

        self.assertIn("STUDIES.jsonl", source)
        self.assertIn("INVESTIGATION_ALREADY_DELEGATED", source)
        write_lines = [
            line
            for line in source.splitlines()
            if any(
                call in line
                for call in (
                    "write_text(", "open(", "os.replace(", "os.unlink(", "unlink(",
                    "mkdir(", "NamedTemporaryFile", "subprocess", "os.system",
                    "boto3", "botocore", "requests",
                )
            )
        ]
        for forbidden in (
            "FINDINGS", "FAILURES", "DECISIONS", "BACKLOG", "proposals",
            "authorizations", "claims.jsonl", "assessments.jsonl",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(
                    any(forbidden in line for line in write_lines),
                    "a lifecycle write line reaches {!r}".format(forbidden),
                )

    def test_primary_module_writes_only_paths_derived_from_paper_id(self):
        source = (RESEARCH_DIR / "literature_primary.py").read_text(encoding="utf-8")

        self.assertIn("cache_path", source)
        # A cache root inside the repository is refused outright.
        self.assertIn("outside the repository", source)
        self.assertIn("os.replace", source)
        for pattern in ("boto3", "botocore", "requests", "urllib.request"):
            with self.subTest(pattern=pattern):
                self.assertNotIn(pattern, source)

    def test_discovery_package_is_metadata_only_and_writes_nothing(self):
        """Structured discovery (INC-012) must not touch artifacts or research state."""

        discovery_dir = RESEARCH_DIR / "literature_discovery"
        modules = sorted(discovery_dir.glob("*.py"))

        self.assertTrue(modules, "the discovery package is missing")
        for path in modules:
            source = path.read_text(encoding="utf-8")
            for pattern in ("write_text(", "write_bytes(", "os.replace(", "import shutil",
                            "import boto3", "import botocore", "import requests",
                            "literature_primary", "literature_ingest"):
                with self.subTest(module=path.name, pattern=pattern):
                    self.assertNotIn(pattern, source)
        # Only the HTTP layer may reach the network primitive; the model and the
        # orchestration are transport-free.
        for name in ("model.py", "core.py", "providers.py"):
            source = (discovery_dir / name).read_text(encoding="utf-8")
            self.assertNotIn("urllib.request", source)

    def test_agent_visible_modules_expose_no_screening_status_field(self):
        for name in ("literature_catalog.py", "literature_read.py"):
            source = (RESEARCH_DIR / name).read_text(encoding="utf-8")
            for pattern in ('"status"', "'status'"):
                with self.subTest(module=name, pattern=pattern):
                    self.assertNotIn(pattern, source)


class AcquisitionBoundaryTests(unittest.TestCase):
    """Public acquisition (INC-013) is deterministic, operator-side tooling.

    The Literature Agent's evidence surface stays read-only: a candidate location
    can be selected, but acquisition itself is never exposed as a model tool.
    """

    ACQUIRE_PATH = RESEARCH_DIR / "literature_acquire.py"

    def test_acquisition_module_converges_on_the_one_ingest_pipeline(self):
        source = self.ACQUIRE_PATH.read_text(encoding="utf-8")

        self.assertTrue(self.ACQUIRE_PATH.is_file())
        self.assertIn("literature_ingest", source)
        self.assertIn("PUBLIC_ACQUIRED", source)
        # No second HTTP stack, no shell and no cloud SDK.
        for forbidden in ("subprocess", "os.system", "os.popen", "eval(", "exec(",
                          "__import__", "boto3", "botocore"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_public_acquisition_reuses_the_discovery_http_machinery(self):
        source = self.ACQUIRE_PATH.read_text(encoding="utf-8")

        self.assertIn("literature_discovery", source)
        self.assertIn("HttpFetcher", source)
        self.assertNotIn("urllib.request", source)

    def test_model_facing_adapter_never_exposes_acquisition(self):
        adapter = TOOLS_PATH.read_text(encoding="utf-8")
        agent = AGENT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("literature_acquire", adapter)
        self.assertNotIn("acquire", adapter)
        self.assertNotIn("literature_acquire", agent)


class AdmissionBoundaryTests(unittest.TestCase):
    """Canonical admission (INC-017) is deterministic, operator-side tooling.

    The Literature Agent may select a discovered paper, but the catalog write is
    owned by the deterministic layer; admission is never a model-facing tool.
    """

    ADMIT_PATH = RESEARCH_DIR / "literature_admit.py"

    def test_admission_module_converges_on_the_canonical_sources(self):
        source = self.ADMIT_PATH.read_text(encoding="utf-8")

        self.assertTrue(self.ADMIT_PATH.is_file())
        self.assertIn("literature_catalog", source)
        # One source policy, reused: admission classifies a source URL with the
        # same acquisition policy rather than a second host convention.
        self.assertIn("literature_acquire", source)
        # No shell, no dynamic code, no cloud SDK and no second HTTP stack.
        for forbidden in ("subprocess", "os.system", "os.popen", "eval(", "exec(",
                          "__import__", "boto3", "botocore", "urllib.request",
                          "requests"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_admission_writes_only_canonical_literature_state(self):
        source = self.ADMIT_PATH.read_text(encoding="utf-8")

        # It may write the catalog, the canonical card and the ledger; it must
        # never reach another research record.
        for forbidden in ("FINDINGS", "FAILURES", "DECISIONS", "BACKLOG",
                          "STUDIES.jsonl", "proposals", "authorizations",
                          "assessments.jsonl", "claims.jsonl", "primary_manifest"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_model_facing_adapter_never_exposes_admission(self):
        adapter = TOOLS_PATH.read_text(encoding="utf-8")
        agent = AGENT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("literature_admit", adapter)
        self.assertNotIn("admit", adapter)
        self.assertNotIn("literature_admit", agent)
        # The agent may *describe* that discovery cannot admit; it must never be
        # granted an admission capability.
        self.assertNotIn("tools: literature_admit", agent)


class BoundedReadCliTests(unittest.TestCase):
    def test_card_read_returns_card_derived_text_for_a_known_paper(self):
        result = run_module("literature_read", "card", "goncalves-2016-mssl", "--max-chars", "400")

        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(result.stdout)
        self.assertEqual(document["source_kind"], "card")
        self.assertEqual(document["evidence_level"], "card-derived")
        self.assertEqual(document["paper_id"], "goncalves-2016-mssl")

    def test_study_artifact_reads_are_limited_to_registered_kinds(self):
        ok = run_module("literature_read", "study", "LT-0001", "analysis", "--max-chars", "200")
        missing = run_module("literature_read", "study", "LT-0002", "result")

        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(json.loads(ok.stdout)["evidence_level"], "Study-derived")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(json.loads(missing.stderr)["kind"], "ARTIFACT_NOT_AVAILABLE")

    def test_survey_reads_are_limited_to_derived_survey_documents(self):
        ok = run_module("literature_read", "survey", "MSSL_SPARSITY_ANALYSIS.md", "--max-chars", "300")

        self.assertEqual(ok.returncode, 0, ok.stderr)
        document = json.loads(ok.stdout)
        self.assertEqual(document["source_kind"], "survey")
        self.assertEqual(document["evidence_level"], "survey-derived")
        self.assertEqual(document["characters"], 300)
        self.assertTrue(document["truncated"])

        # A survey document is derived knowledge one level below a card, so a
        # caller can never mistake it for the paper's own statement.
        self.assertNotEqual(document["evidence_level"], "card-derived")

        for reference in ("../DECISIONS.md", "/etc/passwd", "notes.txt"):
            with self.subTest(reference=reference):
                result = run_module("literature_read", "survey", reference)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(json.loads(result.stderr)["kind"], "INVALID_REFERENCE")

        missing = run_module("literature_read", "survey", "NO_SUCH_SURVEY.md")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(json.loads(missing.stderr)["kind"], "ARTIFACT_NOT_AVAILABLE")

    def test_path_like_reference_cannot_escape_the_literature_set(self):
        for reference in ("../DECISIONS.md", "/etc/passwd", "..", "INDEX"):
            with self.subTest(reference=reference):
                result = run_module("literature_read", "card", reference)
                self.assertEqual(result.returncode, 1)
                self.assertTrue(json.loads(result.stderr)["kind"])

    def test_primary_reports_a_precise_state_instead_of_credentials(self):
        # A paper the manifest does not retain: the state is deterministic and
        # never contains credential material, whatever the environment.
        result = run_module("literature_primary", "get", "zhang-yang-2021-mtl-survey")

        self.assertEqual(result.returncode, 1)
        document = json.loads(result.stderr)
        self.assertEqual(document["kind"], "PRIMARY_NOT_AVAILABLE")
        serialized = json.dumps(document).lower()
        for needle in ("secret", "access_key", "aws", "bucket"):
            with self.subTest(needle=needle):
                self.assertNotIn(needle, serialized)

    def test_primary_read_is_page_and_sha_addressed_when_cached(self):
        status = run_module("literature_primary", "status", "goncalves-2016-mssl")
        state = json.loads(status.stdout)
        artifacts = {artifact["role"]: artifact for artifact in state["artifacts"]}
        if "preprint" not in artifacts or artifacts["preprint"]["cache_state"] != "valid":
            self.skipTest("goncalves-2016-mssl preprint has no verified local copy")

        result = run_module(
            "literature_primary", "read", "goncalves-2016-mssl",
            "--role", "preprint", "--page", "6", "--max-chars", "400",
        )
        if result.returncode != 0:
            self.assertEqual(json.loads(result.stderr)["kind"], "EXTRACTOR_UNAVAILABLE")
            self.skipTest("no PDF text extractor is installed")

        page = json.loads(result.stdout)
        self.assertEqual(page["evidence_level"], "primary")
        self.assertEqual(page["locator"], "primary:page:6")
        self.assertEqual(page["role"], "preprint")
        self.assertEqual(page["sha256"], artifacts["preprint"]["sha256"])
        self.assertEqual(page["page"], 6)
        self.assertNotIn("cache_path", json.dumps(page))
        self.assertNotIn("secret", json.dumps(page).lower())

    def test_primary_refuses_to_choose_between_retained_versions(self):
        status = run_module("literature_primary", "status", "goncalves-2016-mssl")
        state = json.loads(status.stdout)
        if len(state["artifacts"]) < 2:
            self.skipTest("goncalves-2016-mssl retains fewer than two versions")

        result = run_module("literature_primary", "get", "goncalves-2016-mssl")

        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr)["kind"], "AMBIGUOUS_ARTIFACT")






class EvidenceAuthorityTests(unittest.TestCase):
    """Historical recall is removed at the configuration boundary, not by prose."""

    def test_project_mcp_config_disables_the_historical_recall_server(self):
        # The native project MCP config path is `<cwd>/.omp/mcp.json`; a root-level
        # `mcp.json` is not read, so a disable placed there would be inert.
        self.assertTrue(MCP_PATH.is_file(), "%s is missing" % MCP_PATH)
        config = json.loads(MCP_PATH.read_text(encoding="utf-8"))

        disabled = config.get("disabledServers") or []
        deja = (config.get("mcpServers") or {}).get("deja") or {}

        self.assertTrue(
            "deja" in disabled or deja.get("enabled") is False,
            "the project MCP config does not disable the recall server: %r" % config,
        )

    def test_project_omp_config_disables_the_recall_extension(self):
        # The recall *content* channel is the user-scope OMP extension written by
        # `deja install omp-auto` at `<omp-config>/extensions/deja/index.js`. It
        # hooks before_agent_start/context and injects <deja-recall> text into
        # every agent's context, which mcp.json cannot gate. The lever is the
        # `disabledExtensions` setting; the project settings file OMP reads is
        # `<cwd>/.omp/config.yml` and the id form is
        # `extension-module:<derivedName>` (derived name `deja`).
        self.assertTrue(OMP_CONFIG_PATH.is_file(), "%s is missing" % OMP_CONFIG_PATH)
        text = OMP_CONFIG_PATH.read_text(encoding="utf-8")
        entries = re.findall(r"^\s*-\s*(\S+)\s*$", text, re.MULTILINE)

        self.assertIn(
            "extension-module:deja",
            entries,
            "the project OMP config does not disable the recall extension: %r" % text,
        )

    def test_agent_declares_no_recall_or_mcp_capability(self):
        entries, _ = parse_frontmatter(AGENT_PATH)
        fields = {}
        for line in entries:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        declared = [n.strip() for n in fields.get("tools", "").split(",") if n.strip()]

        for name in declared:
            lowered = name.lower()
            for needle in RECALL_TOOL_NEEDLES:
                with self.subTest(tool=name, needle=needle):
                    self.assertNotIn(needle, lowered)

    def test_skill_fails_closed_when_evidence_tools_are_unavailable(self):
        lowered = SKILL_PATH.read_text(encoding="utf-8").lower()

        for phrase in (
            "required_literature_tool_unavailable",
            "evidence_status: unavailable",
            "not established from approved evidence",
            "fail closed",
            "missing_capability",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, lowered)

    def test_skill_and_agent_reject_recalled_text_as_provenance(self):
        for path in (SKILL_PATH, AGENT_PATH):
            lowered = path.read_text(encoding="utf-8").lower()
            with self.subTest(path=path.name):
                self.assertIn("never evidence", lowered)
                self.assertIn("recalled", lowered)
                self.assertIn("this run", lowered)
                self.assertNotIn("session-recalled", lowered)

    def test_skill_requires_claim_lookup_for_every_question_shape(self):
        lowered = SKILL_PATH.read_text(encoding="utf-8").lower()

        for phrase in ("every question shape", "comparative", "claim_ref", "even when"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, lowered)

    def test_agent_and_skill_forbid_undisclosed_recall_substitution(self):
        # Observed failure: with discovery unavailable the agent substituted
        # remembered arXiv identifiers without disclosing the provenance. Both
        # assets must state that a remembered identifier is not retrieved
        # evidence, and that model memory may only seed a search query.
        for path in (SKILL_PATH, AGENT_PATH):
            lowered = path.read_text(encoding="utf-8").lower()
            with self.subTest(path=path.name):
                self.assertIn("never substitute", lowered)
                self.assertIn("search query", lowered)
                self.assertIn("not retrieved evidence", lowered)

    def test_skill_requires_counts_from_the_structured_result(self):
        lowered = SKILL_PATH.read_text(encoding="utf-8").lower()

        for phrase in ("exact count", "structured result", "its length"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, lowered)

    def test_agent_requires_counts_from_the_structured_result(self):
        lowered = AGENT_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("exact count", lowered)
        self.assertIn("structured query result", lowered)
        self.assertIn("its length", lowered)


def write_transcript(path, *, tools, calls, with_init=True):
    lines = []
    if with_init:
        lines.append({"type": "session_init", "tools": list(tools)})
    for name in calls:
        lines.append({
            "type": "message",
            "message": {"role": "assistant", "content": [{"type": "toolCall", "name": name}]},
        })
    Path(path).write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


class TranscriptCheckerTests(unittest.TestCase):
    """An evaluation is invalid unless its own transcript proves the grant."""

    def run_checker(self, *argv):
        return subprocess.run(
            [sys.executable, str(TRANSCRIPT_CHECKER), *argv],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
        )

    def fixture(self, **kwargs):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.close()
        write_transcript(handle.name, **kwargs)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_accepts_a_grant_that_was_exercised(self):
        path = self.fixture(
            tools=REQUIRED_LITERATURE_TOOLS + ("yield",),
            calls=["literature_query", "literature_read", "yield"],
        )
        result = self.run_checker(path)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_rejects_the_provenance_bypass_run(self):
        """The observed failure: recall granted, called, and no corpus tool used."""

        path = self.fixture(
            tools=("yield", "mcp__deja_deja"),
            calls=["mcp__deja_deja", "mcp__deja_deja", "yield"],
        )
        result = self.run_checker("--json", path)
        report = json.loads(result.stdout.strip().splitlines()[0])

        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["verdict"], "INVALID")
        self.assertIn("mcp__deja_deja", report["forbidden_granted"])
        self.assertIn("mcp__deja_deja", report["forbidden_called"])
        self.assertEqual(report["missing_required"], list(REQUIRED_LITERATURE_TOOLS))
        self.assertTrue(any(r.startswith("NO_EVIDENCE_CALL") for r in report["reasons"]))

    def test_rejects_a_grant_without_an_evidence_call(self):
        path = self.fixture(tools=REQUIRED_LITERATURE_TOOLS + ("yield",), calls=["yield"])
        result = self.run_checker(path)

        self.assertEqual(result.returncode, 1)
        self.assertIn("NO_EVIDENCE_CALL", result.stdout)

    def test_rejects_a_transcript_that_never_recorded_a_grant(self):
        path = self.fixture(tools=(), calls=["literature_query"], with_init=False)
        result = self.run_checker(path)

        self.assertEqual(result.returncode, 1)
        self.assertIn("TRANSCRIPT_WITHOUT_TOOL_GRANT", result.stdout)

    def test_rejects_a_native_recall_device_in_the_grant(self):
        path = self.fixture(
            tools=REQUIRED_LITERATURE_TOOLS + ("yield", "recall"),
            calls=["literature_query"],
        )
        result = self.run_checker(path)

        self.assertEqual(result.returncode, 1)
        self.assertIn("FORBIDDEN_TOOL_GRANTED", result.stdout)

    def test_can_require_specific_calls(self):
        path = self.fixture(
            tools=REQUIRED_LITERATURE_TOOLS + ("yield",), calls=["literature_read"]
        )
        ok = self.run_checker("--require-call", "literature_read", path)
        bad = self.run_checker("--require-call", "literature_query", path)

        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("REQUIRED_CALL_ABSENT", bad.stdout)


if __name__ == "__main__":
    unittest.main()
