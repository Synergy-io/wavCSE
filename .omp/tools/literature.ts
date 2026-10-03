/**
 * Semantic, read-only literature capabilities for the Literature Review agent.
 *
 * Each tool is a thin adapter over deterministic Python modules that already
 * validate their own inputs:
 *
 *   literature_resolve  -> improvements.taskrelation.research.literature_query
 *   literature_query    -> improvements.taskrelation.research.literature_query
 *                          (paper_claims / claim -> literature_claims: recorded,
 *                           source-bound literature claims)
 *   literature_read     -> improvements.taskrelation.research.literature_read
 *   literature_primary  -> improvements.taskrelation.research.literature_primary
 *
 * Authority: these tools READ canonical research state. The only write any of
 * them can perform is the disposable primary cache owned by literature_primary.
 * None of them can modify a card, the catalog, the primary manifest, a Study, a
 * finding, a decision, or authorize an experiment.
 *
 * The caller never supplies a bucket, object key, filesystem path, credential,
 * or shell command: the model addresses a `paper_id` (or a registered `LT-*`
 * Study artifact kind) and the deterministic layer resolves the rest.
 */
import type { CustomToolFactory } from "@oh-my-pi/pi-coding-agent";

const RESEARCH_MODULE_DIR = "improvements/taskrelation/research";
const MODULE = "improvements.taskrelation.research";
const PYTHON = "python3";

/** Domain states are normal results the model must branch on, not harness errors. */
type ToolResult = {
	content: { type: "text"; text: string }[];
	details: Record<string, unknown>;
};

/** `execute` argument order differs between host paths; locate the abort signal. */
function findSignal(args: unknown[]): AbortSignal | undefined {
	for (const candidate of args) {
		if (
			candidate !== null &&
			typeof candidate === "object" &&
			typeof (candidate as AbortSignal).addEventListener === "function"
		) {
			return candidate as AbortSignal;
		}
	}
	return undefined;
}

function invalid(field: string, operation: string): ToolResult {
	return {
		content: [{ type: "text", text: `${operation} requires '${field}'` }],
		details: { ok: false, kind: "INVALID_REFERENCE" },
	};
}

export interface ResolveParams {
	paperId?: string;
	title?: string;
	doi?: string;
	arxiv?: string;
	sourceUrl?: string;
}

export interface QueryParams {
	operation:
		| "list"
		| "resolve"
		| "paper_studies"
		| "study_papers"
		| "study"
		| "paper_claims"
		| "claim";
	paperId?: string;
	studyId?: string;
	claimId?: string;
	claimType?: string;
	year?: number;
	author?: string;
	venue?: string;
}

export interface ReadParams {
	source: "card" | "study" | "survey";
	paperId?: string;
	studyId?: string;
	artifact?: "analysis" | "note" | "plan" | "result";
	document?: string;
	maxChars?: number;
}

export interface PrimaryParams {
	paperId: string;
	operation?: "status" | "get" | "read";
	role?: string;
	page?: number;
	pageEnd?: number;
	maxChars?: number;
}

/** Build the deterministic CLI argv for a validated query request. */
function queryArgv(params: QueryParams): string[] | undefined {
	const argv = ["literature_query"];
	switch (params.operation) {
		case "list": {
			argv.push("list");
			if (params.year !== undefined) argv.push("--year", String(params.year));
			if (params.author) argv.push("--author", params.author);
			if (params.venue) argv.push("--venue", params.venue);
			return argv;
		}
		case "resolve":
			return params.paperId ? [...argv, "resolve", params.paperId] : undefined;
		case "paper_studies":
			return params.paperId
				? [...argv, "paper-studies", params.paperId]
				: undefined;
		case "study_papers":
		case "study":
			return params.studyId
				? [...argv, "study-papers", params.studyId]
				: undefined;
		case "paper_claims": {
			if (!params.paperId) return undefined;
			const claims = ["literature_claims", "paper", params.paperId];
			if (params.claimType) claims.push("--claim-type", params.claimType);
			return claims;
		}
		case "claim":
			return params.paperId && params.claimId
				? ["literature_claims", "get", params.paperId, params.claimId]
				: undefined;
		default:
			return undefined;
	}
}

/** Build the deterministic CLI argv for a validated bounded-read request. */
function readArgv(params: ReadParams): string[] | undefined {
	const argv = ["literature_read"];
	if (params.source === "card") {
		if (!params.paperId) return undefined;
		argv.push("card", params.paperId);
	} else if (params.source === "survey") {
		if (!params.document) return undefined;
		argv.push("survey", params.document);
	} else {
		if (!params.studyId || !params.artifact) return undefined;
		argv.push("study", params.studyId, params.artifact);
	}
	if (params.maxChars !== undefined) argv.push("--max-chars", String(params.maxChars));
	return argv;
}

const factory: CustomToolFactory = (pi) => {
	const z = pi.zod;
	const startCwd = pi.cwd;

	/** Resolve the repository root once; every invocation revalidates it cheaply. */
	async function repoRoot(): Promise<string | undefined> {
		let current = startCwd;
		for (;;) {
			if (await Bun.file(`${current}/${RESEARCH_MODULE_DIR}/literature_catalog.py`).exists()) {
				return current;
			}
			const parent = current.replace(/\/[^/]+\/?$/, "");
			if (!parent || parent === current) return undefined;
			current = parent;
		}
	}

	async function invoke(argv: string[], rest: unknown[]): Promise<ToolResult> {
		const root = await repoRoot();
		if (!root) {
			const text =
				`wavCSE research modules were not found at or above ${startCwd}; ` +
				`the literature tools only work inside the wavCSE repository`;
			return { content: [{ type: "text", text }], details: { ok: false, error: text } };
		}
		const result = await pi.exec(PYTHON, ["-m", `${MODULE}.${argv[0]}`, ...argv.slice(1)], {
			cwd: root,
			signal: findSignal(rest),
		});
		const payload = result.stdout.trim() || result.stderr.trim();
		let document: unknown = null;
		try {
			document = payload ? JSON.parse(payload) : null;
		} catch {
			document = { unparsed: payload };
		}
		const kind =
			document !== null && typeof document === "object"
				? ((document as Record<string, unknown>).kind as string | undefined)
				: undefined;
		return {
			content: [{ type: "text", text: JSON.stringify(document, null, 2) }],
			details: { ok: result.code === 0, kind, exitCode: result.code },
		};
	}

	return [
		{
			name: "literature_resolve",
			label: "Resolve Literature Identity",
			description:
				"Deduplicate a candidate paper against retained literature. Reports known " +
				"(with the existing paper_id and canonical card), new, or ambiguous. Use it " +
				"before treating a paper as already retained.",
			parameters: z.object({
				paperId: z.string().optional().describe("canonical paper_id (card slug)"),
				title: z.string().optional(),
				doi: z.string().optional(),
				arxiv: z.string().optional(),
				sourceUrl: z.string().optional(),
			}),
			async execute(_toolCallId: string, params: unknown, ...rest: unknown[]) {
				const p = params as ResolveParams;
				const argv = ["literature_query", "identify"];
				if (p.paperId) argv.push("--paper-id", p.paperId);
				if (p.title) argv.push("--title", p.title);
				if (p.doi) argv.push("--doi", p.doi);
				if (p.arxiv) argv.push("--arxiv", p.arxiv);
				if (p.sourceUrl) argv.push("--source-url", p.sourceUrl);
				if (argv.length === 2) {
					return invalid("one of paperId, title, doi, arxiv, sourceUrl", "literature_resolve");
				}
				return invoke(argv, rest);
			},
		},
		{
			name: "literature_query",
			label: "Query Literature",
			description:
				"Enumerate retained papers and read Study-scoped literature state as compact " +
				"records with card and artifact pointers. A Study's decision describes that " +
				"Study's question only; it is never a global paper status.",
			parameters: z.object({
				operation: z
					.enum([
						"list",
						"resolve",
						"paper_studies",
						"study_papers",
						"study",
						"paper_claims",
						"claim",
					])
					.describe(
						"which bounded read-only query to run; paper_claims and claim return " +
							"recorded source-bound literature claims (paper_claims: all claims for a " +
							"paper, optionally filtered by claimType; claim: one exact claim)",
					),
				paperId: z
					.string()
					.optional()
					.describe("required by resolve, paper_studies, paper_claims and claim"),
				studyId: z.string().optional().describe("required by study_papers and study"),
				claimId: z.string().optional().describe("required by claim"),
				claimType: z
					.string()
					.optional()
					.describe("optional paper_claims filter, e.g. relation-object or method-objective"),
				year: z.number().int().optional(),
				author: z.string().optional(),
				venue: z.string().optional(),
			}),
			async execute(_toolCallId: string, params: unknown, ...rest: unknown[]) {
				const p = params as QueryParams;
				const argv = queryArgv(p);
				if (!argv) {
					const needed =
						p.operation === "list"
							? "operation"
							: p.operation === "claim"
								? "paperId and claimId"
								: p.operation === "paper_claims"
									? "paperId"
									: "paperId or studyId";
					return invalid(needed, `literature_query operation=${p.operation}`);
				}
				return invoke(argv, rest);
			},
		},
		{
			name: "literature_read",
			label: "Read Literature Artifact",
			description:
				"Read one bounded retained-literature artifact: a paper's canonical card, a " +
				"registered LT Study artifact (analysis, note, plan, result), or a derived " +
				"literature-survey document. Returns text plus its evidence level " +
				"(card-derived, Study-derived or survey-derived) and a truncation flag. " +
				"Arbitrary filesystem paths are refused by design.",
			parameters: z.object({
				source: z
					.enum(["card", "study", "survey"])
					.describe(
						"card = per-paper derived knowledge; study = registered LT artifact; " +
							"survey = derived literature/theory synthesis document",
					),
				paperId: z.string().optional().describe("required when source is card"),
				studyId: z.string().optional().describe("required when source is study"),
				artifact: z
					.enum(["analysis", "note", "plan", "result"])
					.optional()
					.describe("required when source is study"),
				document: z
					.string()
					.optional()
					.describe("required when source is survey; a direct literature_survey/*.md filename"),
				maxChars: z.number().int().min(1).max(100000).optional(),
			}),
			async execute(_toolCallId: string, params: unknown, ...rest: unknown[]) {
				const p = params as ReadParams;
				const argv = readArgv(p);
				if (!argv) {
					const needed =
						p.source === "study"
							? "studyId and artifact"
							: p.source === "survey"
								? "document"
								: "paperId";
					return invalid(needed, `literature_read source=${p.source}`);
				}
				return invoke(argv, rest);
			},
		},
		{
			name: "literature_primary",
			label: "Primary Paper Evidence",
			description:
				"Work with a retained primary artifact (the original PDF) by paper_id. " +
				"operation=status reports which artifacts are retained (a paper may retain " +
				"several versions, e.g. a preprint and the published version), operation=get " +
				"resolves a checksum-verified local copy, and operation=read returns a bounded " +
				"text view of the verified artifact with page provenance (a 1-based physical " +
				"PDF page index, or an inclusive page range), the artifact's role and its " +
				"SHA-256. Pass role to choose a version: it is required when more than one is " +
				"retained, and never guessed, so a preprint and the published version cannot " +
				"be conflated. Reports precise states such as PRIMARY_NOT_AVAILABLE, " +
				"AMBIGUOUS_ARTIFACT or STORAGE_NOT_CONFIGURED. It never asks for credentials, " +
				"a bucket, an object key or a filesystem path, cannot register an artifact, " +
				"and cannot fetch from the internet. Extracted text is a derived view, not the " +
				"evidence: the PDF bytes remain primary, and imperfectly extracted equations " +
				"must be reported as uncertain, never reconstructed.",
			parameters: z.object({
				paperId: z.string().describe("canonical paper_id"),
				operation: z
					.enum(["status", "get", "read"])
					.optional()
					.describe(
						"status = retention/cache state for every retained version; get = resolve " +
							"the verified local copy; read = bounded, page-provenanced text of the " +
							"verified artifact (default: status)",
					),
				role: z
					.string()
					.optional()
					.describe(
						"artifact version/role, e.g. preprint or published; required when the " +
							"paper retains more than one artifact, never guessed",
					),
				page: z
					.number()
					.int()
					.optional()
					.describe("operation=read only: a 1-based physical PDF page index"),
				pageEnd: z
					.number()
					.int()
					.optional()
					.describe("operation=read only: inclusive end of a page range (with page)"),
				maxChars: z
					.number()
					.int()
					.min(1)
					.max(100000)
					.optional()
					.describe("operation=read only: bound on returned characters"),
			}),
			async execute(_toolCallId: string, params: unknown, ...rest: unknown[]) {
				const p = params as PrimaryParams;
				const operation = p.operation ?? "status";
				const role = p.role ? ["--role", p.role] : [];
				if (operation === "read") {
					const argv = ["literature_primary", "read", p.paperId, ...role];
					if (p.page !== undefined) argv.push("--page", String(p.page));
					if (p.pageEnd !== undefined) argv.push("--page-end", String(p.pageEnd));
					if (p.maxChars !== undefined) argv.push("--max-chars", String(p.maxChars));
					return invoke(argv, rest);
				}
				return invoke(["literature_primary", operation, p.paperId, ...role], rest);
			},
		},
	];
};

export default factory;
