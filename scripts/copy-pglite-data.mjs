import { copyFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const fromDir = join(root, "node_modules/@electric-sql/pglite/dist");
const toDir = join(root, ".vercel/output/functions/__server.func/_libs");
const files = ["pglite.data", "pglite.wasm", "initdb.wasm"];

if (!existsSync(toDir)) {
  console.warn("[pglite] function bundle not present, skip");
  process.exit(0);
}
for (const name of files) {
  const source = join(fromDir, name);
  if (!existsSync(source)) {
    console.warn(`[pglite] missing ${name}, skip`);
    continue;
  }
  copyFileSync(source, join(toDir, name));
}
console.log("[pglite] copied wasm and data next to the server bundle");
