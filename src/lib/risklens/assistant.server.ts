import { doubtInputSchema } from "./contract";

const SYSTEM = [
  "You are the RiskLens voice assistant.",
  "Answer in one or two short sentences, under 35 words.",
  "The book is synthetic. A high score is a doubt, not a finding of fraud.",
  "Use only the facts given. Do not decide the review.",
  "No markdown.",
].join(" ");

function spoken(content: unknown) {
  const raw = typeof content === "string" ? content : "";
  const cleaned = raw
    .replace(/[*_`#]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned.slice(0, 700);
}

export async function speakDoubt(input: unknown): Promise<{ heard: string; text: string; audioBase64: string }> {
  const facts = doubtInputSchema.parse(input);
  const apiKey = process.env.XAI_API_KEY;
  if (!apiKey) throw new Error("The voice assistant is not available in this environment.");

  const heard = facts.audioBase64 ? await transcribe(apiKey, facts.audioBase64, facts.audioMime) : "";
  const asked = [heard, facts.question ?? ""].filter(Boolean).join(" ").trim();

  const lines = [
    facts.item
      ? `Open case ${facts.item.transactionId}. Amount ${facts.item.amount.toFixed(2)} pounds. Hour ${facts.item.hour}. Velocity ${facts.item.velocity}. Distance ${facts.item.distanceKm.toFixed(1)} km. New device ${facts.item.newDevice ? "yes" : "no"}. Score ${facts.item.score.toFixed(3)}. ${facts.item.flagged ? "Flagged" : "Not flagged"}. Notes: ${facts.item.reasons || "none"}. Prior review: ${facts.item.decision ?? "none"}${facts.item.reasonCode ? ` / ${facts.item.reasonCode}` : ""}.`
      : "No case is open. Speak about where this model is doubtful.",
    facts.item?.injected === true ? "The generator marked this row as an injected scenario. Mention that only as a label." : "",
    facts.model
      ? `Model ${facts.model.label}. Threshold ${facts.model.threshold.toFixed(3)}. Holdout flag rate ${(facts.model.flagRate * 100).toFixed(1)} percent. Precision ${facts.model.precision === null ? "unknown" : (facts.model.precision * 100).toFixed(1) + " percent"}. Recall ${facts.model.recall === null ? "unknown" : (facts.model.recall * 100).toFixed(1) + " percent"}. AUC ${facts.model.auc === null ? "unknown" : facts.model.auc.toFixed(2)}.`
      : "No model is registered.",
    `Drift: ${facts.drift.map((row) => `${row.feature} ${row.status} ${row.smd.toFixed(2)}`).join("; ") || "none"}.`,
    `Slice gaps in percentage points: ${facts.slices.map((row) => `${row.name} ${row.gap >= 0 ? "+" : ""}${(row.gap * 100).toFixed(1)} on ${row.n}`).join("; ") || "none"}.`,
    asked ? "" : "No new question. Brief the open doubt.",
  ]
    .filter(Boolean)
    .join("\n");

  const prior = (facts.history ?? []).map((turn) => ({
    role: turn.role,
    content: turn.text,
  }));
  const messages = [
    { role: "system", content: SYSTEM },
    { role: "user", content: lines },
    ...prior,
    ...(asked ? [{ role: "user" as const, content: asked }] : []),
  ];

  const chat = await fetch("https://api.x.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "grok-4-1-fast-non-reasoning",
      max_tokens: 70,
      temperature: 0.2,
      messages,
    }),
  });
  if (!chat.ok) throw new Error("The assistant could not answer. Try again in a moment.");
  const chatBody = (await chat.json()) as {
    choices?: { message?: { content?: unknown } }[];
  };
  const text = spoken(chatBody.choices?.[0]?.message?.content);
  if (!text) throw new Error("The assistant had nothing to say about this case.");

  const speech = await fetch("https://api.x.ai/v1/tts", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      text,
      voice_id: "eve",
      language: "en",
      speed: 1.2,
    }),
  });
  if (!speech.ok) throw new Error("The answer is ready, but the voice did not answer. Try again.");
  const audio = Buffer.from(await speech.arrayBuffer()).toString("base64");
  return { heard, text, audioBase64: audio };
}

async function transcribe(apiKey: string, audioBase64: string, audioMime?: string) {
  const bytes = Buffer.from(audioBase64, "base64");
  if (bytes.length < 800) throw new Error("That clip was too short to hear.");
  const mime = audioMime?.includes("mp4") ? "audio/mp4" : "audio/webm";
  const name = mime === "audio/mp4" ? "doubt.m4a" : "doubt.webm";
  const form = new FormData();
  form.append("model", "grok-voice-transcribe-2.0");
  form.append("language", "en");
  form.append("file", new Blob([bytes], { type: mime }), name);
  const response = await fetch("https://api.x.ai/v1/stt", {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: form,
  });
  if (!response.ok) throw new Error("The recording could not be heard. Try again, or type the doubt.");
  const body = (await response.json()) as { text?: string };
  const text = (body.text ?? "").replace(/\s+/g, " ").trim().slice(0, 280);
  if (!text) throw new Error("I didn't catch that. Try again a little closer to the microphone.");
  return text;
}
