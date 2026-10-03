/**
 * Deterministic harness for the literature tool adapter.
 *
 * Loads `.omp/tools/literature.ts` with a minimal stand-in host, then exercises
 * the real Python modules through it. This proves the adapter builds correct
 * argv, resolves the repository root, parses output, and reports domain states —
 * without needing a live OMP session or any model call.
 *
 * Run: bun run scripts/agents/literature_tools_check.ts
 */
const REPO_ROOT = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
const TOOLS_PATH = `${REPO_ROOT}/.omp/tools/literature.ts`;

type ToolResult = {
	content: { type: "text"; text: string }[];
	details: Record<string, unknown>;
};

type Tool = {
	name: string;
	execute: (...args: unknown[]) => Promise<ToolResult>;
};

const failures: string[] = [];
let checks = 0;

function check(label: string, condition: boolean, detail?: unknown) {
	checks += 1;
	if (condition) {
		console.log(`ok   ${label}`);
		return;
	}
	console.log(`FAIL ${label}${detail === undefined ? "" : ` :: ${JSON.stringify(detail)}`}`);
	failures.push(label);
}

// Minimal host stand-in: only what the adapter actually uses.
const host = {
	cwd: REPO_ROOT,
	zod: {
		string: () => {
			const schema: Record<string, unknown> = {};
			schema.optional = () => schema;
			schema.describe = () => schema;
			return schema;
		},
		number: () => {
			const schema: Record<string, unknown> = {};
			schema.int = () => schema;
			schema.min = () => schema;
			schema.max = () => schema;
			schema.optional = () => schema;
			schema.describe = () => schema;
			return schema;
		},
		enum: () => {
			const schema: Record<string, unknown> = {};
			schema.describe = () => schema;
			schema.optional = () => schema;
			return schema;
		},
		object: (shape: Record<string, unknown>) => shape,
	},
	async exec(command: string, args: string[], options?: { cwd?: string }) {
		const proc = Bun.spawn([command, ...args], {
			cwd: options?.cwd ?? REPO_ROOT,
			stdout: "pipe",
			stderr: "pipe",
		});
		const [stdout, stderr, code] = await Promise.all([
			new Response(proc.stdout).text(),
			new Response(proc.stderr).text(),
			proc.exited,
		]);
		return { stdout, stderr, code, killed: false };
	},
};

// Dynamic import is required here: this harness verifies that a runtime-discovered
// tool module loads and behaves, which is exactly the boundary a static import cannot cover.
const module = await import(TOOLS_PATH);
const tools: Tool[] = await module.default(host);
// Map, not Record: the tool set is runtime-loaded, not a static literal table.
const byName = new Map(tools.map((tool) => [tool.name, tool]));

function parse(result: ToolResult): Record<string, unknown> {
	return JSON.parse(result.content[0].text) as Record<string, unknown>;
}

async function call(name: string, params: unknown): Promise<ToolResult> {
	const tool = byName.get(name);
	if (!tool) throw new Error(`tool ${name} was not exposed`);
	return tool.execute("check", params);
}

check(
	"exposes exactly the four semantic capabilities",
	tools.length === 4 &&
		["literature_resolve", "literature_query", "literature_read", "literature_primary"].every(
			(name) => byName.has(name),
		),
	tools.map((t) => t.name),
);

// 1. identity dedup: known
const known = await call("literature_resolve", { doi: "10.1145/3580305.3599261" });
check(
	"resolve reports a known paper with its canonical card",
	known.details.kind === undefined &&
		(parse(known).candidate as Record<string, unknown>).status === "known",
	known.details,
);

// 2. identity dedup: new
const fresh = await call("literature_resolve", { title: "A Paper We Do Not Retain" });
check(
	"resolve reports an unseen candidate as new",
	(parse(fresh).candidate as Record<string, unknown>).status === "new",
	parse(fresh),
);

// 3. identity dedup: ambiguous
const ambiguous = await call("literature_resolve", {
	doi: "10.1145/3580305.3599261",
	arxiv: "2009.05618",
});
check(
	"resolve reports disagreeing fields as ambiguous",
	(parse(ambiguous).candidate as Record<string, unknown>).status === "ambiguous",
	parse(ambiguous),
);

// 4. query: list filtered by metadata
const list = await call("literature_query", { operation: "list", year: 2018 });
const papers = (parse(list).papers as unknown[]) ?? [];
check("query lists retained papers for a year filter", papers.length === 3, papers.length);

// 5. query: Study-scoped assessment, never a global verdict
const relationships = await call("literature_query", {
	operation: "paper_studies",
	paperId: "chen-et-al-2018-gradnorm",
});
const first = (parse(relationships).relationships as Record<string, unknown>[])[0];
const assessment = first.assessment as Record<string, unknown>;
check(
	"query keeps the assessment Study-scoped with a resolvable pointer",
	assessment.study_id === "LT-0001" &&
		assessment.decision === "exclude_taxonomy" &&
		typeof assessment.detail_anchor === "string",
	assessment,
);

// 6. bounded read: card
const card = await call("literature_read", {
	source: "card",
	paperId: "goncalves-2016-mssl",
	maxChars: 300,
});
check(
	"read returns card-derived evidence with a truncation flag",
	parse(card).evidence_level === "card-derived" &&
		parse(card).truncated === true &&
		parse(card).characters === 300,
	parse(card),
);

// 7. bounded read: Study artifact
const study = await call("literature_read", {
	source: "study",
	studyId: "LT-0001",
	artifact: "analysis",
	maxChars: 200,
});
check("read returns Study-derived evidence", parse(study).evidence_level === "Study-derived", parse(study));

// 8. bounded read: arbitrary path is refused
const escaped = await call("literature_read", { source: "card", paperId: "../../DECISIONS.md" });
check(
	"read refuses a path-like reference",
	escaped.details.ok === false && escaped.details.kind === "UNKNOWN_REFERENCE",
	escaped.details,
);

// 9. missing required parameter is a structured refusal
const incomplete = await call("literature_read", { source: "study", studyId: "LT-0001" });
check(
	"read requires an artifact kind",
	incomplete.details.kind === "INVALID_REFERENCE",
	incomplete.details,
);

// 10. primary: precise state for an unretained paper, no credentials
const unretained = await call("literature_primary", {
	paperId: "zhang-yang-2021-mtl-survey",
	operation: "get",
});
check(
	"primary reports a precise unavailable state for an unretained paper",
	unretained.details.ok === false && unretained.details.kind === "PRIMARY_NOT_AVAILABLE",
	unretained.details,
);
check(
	"primary output never mentions credentials, buckets or keys",
	!/secret|access_key|aws_|bucket|s3/i.test(unretained.content[0].text),
	unretained.content[0].text,
);

// 11. primary: per-version retention status and a bounded, page-provenanced read
const status = await call("literature_primary", {
	paperId: "goncalves-2016-mssl",
	operation: "status",
});
const statusDoc = parse(status);
const artifacts = (statusDoc.artifacts as Record<string, unknown>[]) ?? [];
const preprint = artifacts.find((artifact) => artifact.role === "preprint");
check(
	"primary status enumerates every retained version without guessing one",
	statusDoc.retained === true &&
		artifacts.length >= 2 &&
		artifacts.every((artifact) => typeof artifact.sha256 === "string") &&
		statusDoc.sha256 === null,
	statusDoc,
);

const ambiguousPrimary = await call("literature_primary", {
	paperId: "goncalves-2016-mssl",
	operation: "get",
});
check(
	"primary refuses to choose between versions when none is named",
	ambiguousPrimary.details.ok === false &&
		ambiguousPrimary.details.kind === "AMBIGUOUS_ARTIFACT",
	ambiguousPrimary.details,
);

const read = await call("literature_primary", {
	paperId: "goncalves-2016-mssl",
	operation: "read",
	role: "preprint",
	page: 6,
	maxChars: 400,
});
const readDoc = parse(read);
check(
	"primary read names the artifact version, page and sha, or reports a precise reason",
	read.details.ok === true
		? readDoc.evidence_level === "primary" &&
			readDoc.locator === "primary:page:6" &&
			readDoc.role === "preprint" &&
			readDoc.sha256 === preprint?.sha256 &&
			readDoc.page === 6 &&
			!JSON.stringify(readDoc).includes("cache_path")
		: ["STORAGE_NOT_CONFIGURED", "EXTRACTOR_UNAVAILABLE", "INTEGRITY_MISMATCH"].includes(
				String(read.details.kind),
			),
	read.details.ok ? readDoc.locator : read.details,
);

// 12. recorded claims: the grid is attributed to the survey, not the card
const grid = await call("literature_query", {
	operation: "claim",
	paperId: "goncalves-2016-mssl",
	claimId: "published-lambda2-classification-grid",
});
const gridClaim = parse(grid);
const gridEvidence = (gridClaim.evidence as Record<string, unknown>[])[0];
check(
	"claim returns the recorded grid with its validated survey evidence",
	gridClaim.claim_ref === "goncalves-2016-mssl#published-lambda2-classification-grid" &&
		gridClaim.source_level === "survey" &&
		gridEvidence.kind === "survey" &&
		gridEvidence.document === "MSSL_SPARSITY_ANALYSIS.md" &&
		gridClaim.verification === "derived_existing_record",
	gridClaim,
);

// 13. recorded claims: paper-scoped listing with a type filter
const objectives = await call("literature_query", {
	operation: "paper_claims",
	paperId: "goncalves-2016-mssl",
	claimType: "method-objective",
});
const objectiveClaims = (parse(objectives).claims as Record<string, unknown>[]) ?? [];
check(
	"paper_claims filters by claim type and keeps each claim's own evidence reference",
	objectiveClaims.length === 4 &&
		objectiveClaims.every((claim) => {
			const evidence = claim.evidence as Record<string, unknown>[];
			return (
				claim.claim_type === "method-objective" &&
				Array.isArray(evidence) &&
				evidence.length >= 1 &&
				evidence.every(
					(ref) =>
						typeof ref.kind === "string" &&
						(ref.kind === "primary"
							? typeof ref.role === "string" && typeof ref.sha256 === "string"
							: typeof ref.anchor === "string"),
				)
			);
		}),
	objectiveClaims.map((c) => c.claim_ref),
);

// 14. recorded claims: an unknown reference is a structured refusal
const missingClaim = await call("literature_query", {
	operation: "claim",
	paperId: "goncalves-2016-mssl",
	claimId: "no-such-claim",
});
check(
	"an unknown claim reference is refused with the available set",
	missingClaim.details.ok === false &&
		missingClaim.details.kind === "UNKNOWN_CLAIM" &&
		parse(missingClaim).detail !== undefined,
	missingClaim.details,
);
check(
	"claim query requires a claimId",
	(await call("literature_query", { operation: "claim", paperId: "goncalves-2016-mssl" }))
		.details.kind === "INVALID_REFERENCE",
);

// 15. bounded read: derived survey document
const survey = await call("literature_read", {
	source: "survey",
	document: "MSSL_SPARSITY_ANALYSIS.md",
	maxChars: 250,
});
check(
	"read returns survey-derived evidence",
	parse(survey).evidence_level === "survey-derived" &&
		parse(survey).truncated === true,
	parse(survey),
);
check(
	"read refuses a survey path-like reference",
	(await call("literature_read", { source: "survey", document: "../../DECISIONS.md" }))
		.details.ok === false,
);

// 16. outside the repository the adapter refuses rather than guessing
const outside = await module.default({ ...host, cwd: "/tmp" });
const outsideResult = await outside[0].execute("check", { title: "x" });
check(
	"a session outside the wavCSE repository gets a structured refusal",
	outsideResult.details.ok === false &&
		outsideResult.content[0].text.includes("research modules were not found"),
	outsideResult,
);

console.log(
	failures.length === 0
		? `\nliterature tools: OK (${checks} checks)`
		: `\nliterature tools: ${failures.length}/${checks} FAILED`,
);
if (failures.length > 0) process.exit(1);
