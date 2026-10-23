#!/usr/bin/env node
/**
 * Start the console on port 8080 and the local case brief on 8091.
 * The brief is the first line of an open case. Voice still needs XAI_API_KEY.
 */
import { spawn } from "node:child_process";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// .env supplies XAI_API_KEY for the voice assistant. A value already in the process wins.
export function readDotEnv(text) {
  const env = {};
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (key) env[key] = value;
  }
  return env;
}

function isMain() {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return realpathSync(entry) === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
}

function main() {
  const root = dirname(dirname(fileURLToPath(import.meta.url)));
  const fileEnv = existsSync(join(root, ".env")) ? readDotEnv(readFileSync(join(root, ".env"), "utf8")) : {};
  const env = { ...fileEnv, ...process.env };
  const python =
    process.platform === "win32"
      ? join(root, ".venv", "Scripts", "python.exe")
      : join(root, ".venv", "bin", "python");
  const brief = join(root, "python", "brief_server.py");

  let briefProc = null;
  if (existsSync(python) && existsSync(brief)) {
    briefProc = spawn(python, [brief], { stdio: "ignore", windowsHide: true, env });
    briefProc.on("error", () => {
      briefProc = null;
    });
  }

  const vite = spawn(
    process.execPath,
    [
      join(root, "scripts", "with-app-env.mjs"),
      process.execPath,
      join(root, "node_modules", "vite", "bin", "vite.js"),
      "dev",
      "--host",
      "0.0.0.0",
      "--port",
      "8080",
    ],
    { stdio: "inherit", env },
  );

  function stopBrief() {
    if (briefProc && briefProc.exitCode === null) briefProc.kill();
  }

  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(signal, () => {
      stopBrief();
      vite.kill(signal);
    });
  }

  vite.on("exit", (code, signal) => {
    stopBrief();
    if (signal) process.exit(1);
    process.exit(code ?? 0);
  });
}

if (isMain()) main();
