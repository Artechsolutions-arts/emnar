#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { basename } from "node:path";

function parseEnvFile(path) {
  const raw = readFileSync(path, "utf8");
  const env = {};
  for (const line of raw.split(/\r?\n/)) {
    if (!line || line.trim().startsWith("#")) continue;
    const idx = line.indexOf("=");
    if (idx <= 0) continue;
    const k = line.slice(0, idx).trim();
    const v = line.slice(idx + 1).trim();
    env[k] = v;
  }
  return env;
}

function shellOut(command, args) {
  const result = spawnSync(command, args, { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || `${command} failed`);
  }
  return result.stdout.trim();
}

function fetchTopChunks(question, topK, dbPassword) {
  // Use JSON output for robust parsing.
  const sql = `
    SELECT json_agg(row_to_json(t)) FROM (
      SELECT doc_id, chunk_index, content,
             ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $q$${question}$q$)) AS score
      FROM rag.md_chunks
      WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $q$${question}$q$)
      ORDER BY score DESC
      LIMIT ${topK}
    ) t;
  `.replace(/\s+/g, " ");

  const output = shellOut("docker", [
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
  ]);

  if (!output || output === "null") return [];
  return JSON.parse(output);
}

async function askOpenAI(apiKey, model, question, chunks) {
  const context = chunks
    .map(
      (c) =>
        `[doc_id=${c.doc_id ?? ""}, chunk_index=${c.chunk_index ?? ""}, score=${Number(
          c.score ?? 0
        ).toFixed(4)}]\n${c.content}`
    )
    .join("\n\n");

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: "system",
          content:
            "Answer strictly from the provided context. If the context is insufficient, say so clearly.",
        },
        {
          role: "user",
          content: `Question:\n${question}\n\nContext:\n${context}`,
        },
      ],
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`OpenAI error ${response.status}: ${err}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content ?? "";
}

function extractSourceName(docId) {
  if (!docId) return "unknown";
  const normalized = String(docId).replace(/\\/g, "/");
  return basename(normalized);
}

async function main() {
  const question = process.argv.slice(2).join(" ").trim();
  if (!question) {
    console.error('Usage: node tools/md_chunks_ask.mjs "your question"');
    process.exit(1);
  }

  const envPath = resolve("deployment/docker_compose/.env");
  const env = parseEnvFile(envPath);
  const apiKey = env.GEN_AI_API_KEY;
  const model = env.GEN_AI_MODEL_VERSION || "gpt-5-nano";
  const dbPassword = env.POSTGRES_PASSWORD;

  if (!apiKey) throw new Error("GEN_AI_API_KEY missing in deployment/docker_compose/.env");
  if (!dbPassword) throw new Error("POSTGRES_PASSWORD missing in deployment/docker_compose/.env");

  const chunks = fetchTopChunks(question, 5, dbPassword);
  if (!chunks.length) {
    console.log("No relevant rows found in rag.md_chunks.");
    return;
  }

  const answer = await askOpenAI(apiKey, model, question, chunks);
  const seenDocIds = new Set();
  const sources = [];
  for (const chunk of chunks) {
    const docId = chunk.doc_id ?? "";
    if (seenDocIds.has(docId)) continue;
    seenDocIds.add(docId);
    sources.push({
      doc_id: docId,
      source_file: extractSourceName(docId),
    });
  }

  console.log(`Answer: ${answer}\n`);
  console.log("Sources:");
  for (const source of sources) {
    console.log(`- ${source.source_file} (doc_id: ${source.doc_id})`);
  }
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
