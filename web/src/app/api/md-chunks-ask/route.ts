import { NextResponse } from "next/server";
import { spawnSync } from "node:child_process";
import { Client } from "pg";

interface MdChunkRow {
  doc_id: string | null;
  chunk_index: number | null;
  content: string;
  score: number | null;
}

interface AskRequestBody {
  question?: string;
}

/** Direct TCP to Postgres (fast). Docker fallback is slow (new exec each request). */
const PG_STATEMENT_TIMEOUT_MS = 4000;
const SQL_DOCKER_TIMEOUT_MS = 8000;
const OPENAI_TIMEOUT_MS = 12000;
const TOP_K = 3;
const CHUNK_CHAR_LIMIT = 1500;

function getPostgresConfigFromEnv(): {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
} | null {
  const password = process.env.POSTGRES_PASSWORD;
  if (!password) {
    return null;
  }
  const host = process.env.POSTGRES_HOST || "localhost";
  const port = Number(process.env.POSTGRES_PORT || "5432");
  const user = process.env.POSTGRES_USER || "postgres";
  const database = process.env.POSTGRES_DATABASE || "ragchat";
  return { host, port, user, password, database };
}

async function getTopChunksPg(
  question: string,
  topK: number
): Promise<MdChunkRow[] | null> {
  const cfg = getPostgresConfigFromEnv();
  if (!cfg) {
    return null;
  }

  const client = new Client({
    host: cfg.host,
    port: cfg.port,
    user: cfg.user,
    password: cfg.password,
    database: cfg.database,
    connectionTimeoutMillis: 5000,
  });

  try {
    await client.connect();
    await client.query(`SET statement_timeout = '${PG_STATEMENT_TIMEOUT_MS}ms'`);

    const sql = `
      SELECT coalesce(json_agg(row_to_json(t)), 'null'::json)::text AS payload FROM (
        SELECT doc_id, chunk_index,
               left(content, ${CHUNK_CHAR_LIMIT}) AS content,
               ts_rank_cd(
                 to_tsvector('english', content),
                 plainto_tsquery('english', $1)
               ) AS score
        FROM rag.md_chunks
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
        ORDER BY score DESC
        LIMIT ${topK}
      ) t
    `;

    const result = await client.query<{ payload: string }>(sql, [question]);
    const output = result.rows[0]?.payload?.trim();
    if (!output || output === "null") {
      return [];
    }
    return JSON.parse(output) as MdChunkRow[];
  } catch {
    return null;
  } finally {
    await client.end().catch(() => undefined);
  }
}

function runSqlDocker(sql: string, dbPassword: string): string {
  const result = spawnSync(
    "docker",
    [
      "exec",
      "-e",
      `PGPASSWORD=${dbPassword}`,
      "onyx-relational_db-1",
      "psql",
      "-h",
      "host.docker.internal",
      "-U",
      "postgres",
      "-d",
      "ragchat",
      "-t",
      "-A",
      "-c",
      sql,
    ],
    {
      encoding: "utf8",
      timeout: SQL_DOCKER_TIMEOUT_MS,
      maxBuffer: 8 * 1024 * 1024,
    }
  );

  if (result.error) {
    if ((result.error as NodeJS.ErrnoException).code === "ETIMEDOUT") {
      throw new Error("Postgres query timed out");
    }
    throw new Error(result.error.message);
  }

  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || "Failed to query rag.md_chunks");
  }

  return (result.stdout || "").trim();
}

function getTopChunksDocker(question: string, dbPassword: string, topK: number): MdChunkRow[] {
  const escapedQuestion = question.replaceAll("$q$", " ");
  const sql = `
    SELECT coalesce(json_agg(row_to_json(t)), 'null'::json)::text FROM (
      SELECT doc_id, chunk_index, left(content, ${CHUNK_CHAR_LIMIT}) AS content,
             ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $q$${escapedQuestion}$q$)) AS score
      FROM rag.md_chunks
      WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $q$${escapedQuestion}$q$)
      ORDER BY score DESC
      LIMIT ${topK}
    ) t;
  `.replace(/\s+/g, " ");

  const output = runSqlDocker(sql, dbPassword);
  if (!output || output === "null") {
    return [];
  }

  return JSON.parse(output) as MdChunkRow[];
}

async function getTopChunks(question: string, dbPassword: string, topK: number): Promise<MdChunkRow[]> {
  const direct = await getTopChunksPg(question, topK);
  if (direct !== null) {
    return direct;
  }
  return getTopChunksDocker(question, dbPassword, topK);
}

async function askOpenAI(
  apiKey: string,
  model: string,
  question: string,
  chunks: MdChunkRow[]
): Promise<string> {
  const context = chunks
    .map(
      (chunk) =>
        `[doc_id=${chunk.doc_id ?? ""}, chunk_index=${chunk.chunk_index ?? ""}, score=${Number(
          chunk.score ?? 0
        ).toFixed(4)}]\n${chunk.content}`
    )
    .join("\n\n");

  async function callResponsesApi(targetModel: string): Promise<string> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), OPENAI_TIMEOUT_MS);
    const response = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: targetModel,
        max_output_tokens: 220,
        input: [
          {
            role: "system",
            content:
              "Answer strictly from the provided context. Be concise. If the context is insufficient, say so clearly.",
          },
          {
            role: "user",
            content: `Question:\n${question}\n\nContext:\n${context}`,
          },
        ],
      }),
      signal: controller.signal,
    }).finally(() => clearTimeout(timeout));

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`OpenAI error ${response.status}: ${text}`);
    }

    const data = (await response.json()) as {
      output_text?: string;
      output?: Array<{
        type?: string;
        content?: Array<{ type?: string; text?: string }>;
      }>;
    };
    if (data.output_text?.trim()) {
      return data.output_text.trim();
    }

    const message = data.output?.find((item) => item.type === "message");
    const textParts =
      message?.content
        ?.filter((part) => part.type === "output_text")
        .map((part) => part.text?.trim() || "")
        .filter(Boolean) || [];

    return textParts.join("\n").trim();
  }

  const primary = await callResponsesApi(model);
  if (primary) {
    return primary;
  }

  if (model !== "gpt-4o-mini") {
    const fallback = await callResponsesApi("gpt-4o-mini");
    if (fallback) {
      return fallback;
    }
  }

  return "I found relevant chunks but the model returned an empty response. Please try again.";
}

function sourceFileFromDocId(docId: string | null): string {
  if (!docId) {
    return "unknown";
  }

  const normalized = docId.replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || "unknown";
}

export async function POST(request: Request) {
  const requestStarted = Date.now();
  try {
    const body = (await request.json()) as AskRequestBody;
    const question = (body.question || "").trim();
    if (!question) {
      return NextResponse.json(
        {
          error: "question is required",
          duration_ms: Date.now() - requestStarted,
        },
        { status: 400 }
      );
    }

    const apiKey = process.env.GEN_AI_API_KEY;
    const model =
      process.env.MD_CHUNKS_OPENAI_MODEL ||
      process.env.GEN_AI_MODEL_VERSION ||
      "gpt-4o-mini";
    const dbPassword = process.env.POSTGRES_PASSWORD;
    if (!apiKey) {
      return NextResponse.json(
        {
          error: "GEN_AI_API_KEY missing",
          duration_ms: Date.now() - requestStarted,
        },
        { status: 500 }
      );
    }
    if (!dbPassword) {
      return NextResponse.json(
        {
          error: "POSTGRES_PASSWORD missing",
          duration_ms: Date.now() - requestStarted,
        },
        { status: 500 }
      );
    }

    const chunks = await getTopChunks(question, dbPassword, TOP_K);
    if (!chunks.length) {
      return NextResponse.json({
        answer: "I could not find relevant rows in rag.md_chunks for that question.",
        sources: [],
        duration_ms: Date.now() - requestStarted,
      });
    }

    const answer = await askOpenAI(apiKey, model, question, chunks);
    const seenDocIds = new Set<string>();
    const sources: Array<{ doc_id: string; source_file: string }> = [];
    for (const chunk of chunks) {
      const docId = chunk.doc_id || "";
      if (!docId || seenDocIds.has(docId)) {
        continue;
      }
      seenDocIds.add(docId);
      sources.push({
        doc_id: docId,
        source_file: sourceFileFromDocId(docId),
      });
    }

    return NextResponse.json({
      answer,
      sources,
      duration_ms: Date.now() - requestStarted,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({
      answer: `I could not complete the request: ${message}`,
      sources: [],
      duration_ms: Date.now() - requestStarted,
    });
  }
}
