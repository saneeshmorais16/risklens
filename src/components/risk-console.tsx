import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CONTRACT_FIELDS, CONTRACT_VERSION } from "@/lib/risklens/contract";
import { featureDrift, sliceGaps } from "@/lib/risklens/monitor";
import { getQueue, getWorkspace, recordReview, registerModel, scoreBatch, answerDoubt, quickBrief, speakText } from "@/lib/risklens/api";

type Workspace = Awaited<ReturnType<typeof getWorkspace>>;
type ModelSummary = Workspace["models"][number];
type QueueRow = Workspace["queue"][number];
type AuditEvent = Workspace["events"][number];
type LiveRow = Awaited<ReturnType<typeof scoreBatch>>["rows"][number];

const REASONS = [
  ["amount", "Amount"],
  ["velocity", "Velocity"],
  ["distance", "Distance"],
  ["device", "Device"],
  ["overnight", "Overnight"],
  ["pattern", "Pattern"],
] as const;

const DECISIONS = [
  ["clear", "Clear"],
  ["hold", "Hold"],
  ["escalate", "Escalate"],
] as const;

const gbp = new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP" });

function pct(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(1)}%`;
}

function messageOf(error: unknown) {
  if (error instanceof Error && error.message) return error.message;
  return "The scoring service could not finish that request.";
}

function parseCsv(text: string) {
  const lines = text
    .replace(/^\uFEFF/, "")
    .trim()
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);
  if (lines.length < 2) throw new Error("Use a header row and at least one transaction.");
  const header = lines[0].split(",").map((cell) => cell.trim().toLowerCase());
  const needed = ["amount", "hour", "velocity_24h", "distance_km", "new_device"];
  const missing = needed.filter((name) => !header.includes(name));
  if (missing.length > 0) throw new Error(`Missing column${missing.length > 1 ? "s" : ""}: ${missing.join(", ")}.`);
  const index = new Map(header.map((name, position) => [name, position]));
  const body = lines.slice(1);
  const clipped = body.length > 400;
  const rows = body.slice(0, 400).map((line) => {
    const cells = line.split(",").map((cell) => cell.trim());
    const at = (name: string) => Number(cells[index.get(name) ?? 0]);
    const device = at("new_device");
    return {
      amount: at("amount"),
      hour: Math.trunc(at("hour")),
      velocity_24h: Math.trunc(at("velocity_24h")),
      distance_km: at("distance_km"),
      new_device: device === 1 ? 1 : 0,
    };
  });
  return { rows, clipped };
}

export function RiskConsole() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [queue, setQueue] = useState<QueueRow[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [selectedTxId, setSelectedTxId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<"register" | "score" | "review" | "doubt" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [n, setN] = useState(800);
  const [seed, setSeed] = useState(42);
  const [contamination, setContamination] = useState(0.05);
  const [trees, setTrees] = useState(48);
  const [flaggedOnly, setFlaggedOnly] = useState(true);
  const [holdoutOnly, setHoldoutOnly] = useState(true);
  const [showInjected, setShowInjected] = useState(false);
  const [decision, setDecision] = useState<(typeof DECISIONS)[number][0]>("hold");
  const [reasonCode, setReasonCode] = useState<(typeof REASONS)[number][0]>("pattern");
  const [live, setLive] = useState<{ label: string; rows: LiveRow[] } | null>(null);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [question, setQuestion] = useState("");
  const [briefing, setBriefing] = useState<{ text: string; src: string } | null>(null);
  const [turns, setTurns] = useState<{ role: "user" | "assistant"; text: string }[]>([]);
  const [listening, setListening] = useState(false);
  const [heardLive, setHeardLive] = useState("");
  const [voiceNote, setVoiceNote] = useState<string | null>(null);
  const [liveBrief, setLiveBrief] = useState<string | null>(null);
  const voiceRef = useRef<HTMLAudioElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimer = useRef<number | null>(null);
  const heardSpeech = useRef(false);
  const speakRef = useRef<(clip?: { audioBase64: string; audioMime: string }, spoken?: string) => void>(() => undefined);

  useEffect(() => {
    const node = voiceRef.current;
    if (!node || !briefing) return;
    void node.play().catch(() => undefined);
  }, [briefing]);

  useEffect(() => {
    let cancelled = false;
    getWorkspace()
      .then((workspace) => {
        if (cancelled) return;
        setModels(workspace.models);
        setQueue(workspace.queue);
        setSelectedModelId(workspace.activeId);
        setEvents(workspace.events);
      })
      .catch((caught) => {
        if (!cancelled) setError(messageOf(caught));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedModel = models.find((model) => model.id === selectedModelId) ?? null;
  const selectedTx = queue.find((row) => row.id === selectedTxId) ?? null;

  const visible = useMemo(() => {
    return queue.filter((row) => {
      if (holdoutOnly && row.split !== "test") return false;
      if (flaggedOnly && !row.flagged) return false;
      return true;
    });
  }, [queue, holdoutOnly, flaggedOnly]);

  const plotRows = live
    ? live.rows.map((row, index) => ({
        id: index,
        amount: row.amount,
        score: row.anomalyScore,
        flagged: row.flagged,
        name: `Row ${index + 1}`,
      }))
    : queue
        .filter((row) => (holdoutOnly ? row.split === "test" : true))
        .map((row) => ({
          id: row.id,
          amount: row.amount,
          score: row.anomalyScore,
          flagged: row.flagged,
          name: row.transactionId,
        }));

  const calm = plotRows.filter((row) => !row.flagged && row.amount > 0);
  const hot = plotRows.filter((row) => row.flagged && row.amount > 0);
  const holdoutFlagged = queue.filter((row) => row.split === "test" && row.flagged);
  const reviewed = holdoutFlagged.filter((row) => row.decision);
  const flaggedValue = holdoutFlagged.reduce((sum, row) => sum + row.amount, 0);
  const drift = useMemo(() => featureDrift(queue), [queue]);
  const slices = useMemo(() => sliceGaps(queue), [queue]);
  const trend = [...models].reverse();

  async function onRegister() {
    setBusy("register");
    setError(null);
    setNotice(null);
    try {
      const workspace = await registerModel({
        data: { n, seed, contamination, trees },
      });
      setModels(workspace.models);
      setQueue(workspace.queue);
      setSelectedModelId(workspace.activeId);
      setEvents(workspace.events);
      setSelectedTxId(null);
      setLive(null);
      setNotice("New version is active. It was fit on days 1–21 and scored days 22–30 without refitting.");
    } catch (caught) {
      setError(messageOf(caught));
    } finally {
      setBusy(null);
    }
  }

  async function onSelectModel(modelId: number) {
    setSelectedModelId(modelId);
    setSelectedTxId(null);
    setLive(null);
    setError(null);
    setBusy("score");
    try {
      setQueue(await getQueue({ data: { modelId } }));
    } catch (caught) {
      setError(messageOf(caught));
    } finally {
      setBusy(null);
    }
  }

  async function onFile(file: File | null) {
    if (!file) return;
    setBusy("score");
    setError(null);
    setNotice(null);
    try {
      const text = await file.text();
      const parsed = parseCsv(text);
      const result = await scoreBatch({ data: { rows: parsed.rows } });
      setLive({ label: result.versionLabel, rows: result.rows });
      setEvents((current) => [result.event, ...current].slice(0, 20));
      setSelectedTxId(null);
      setNotice(
        parsed.clipped
          ? `Scored the first 400 rows in memory against ${result.versionLabel}. Nothing from the file was stored.`
          : `Scored ${result.rows.length} rows in memory against ${result.versionLabel}. Nothing from the file was stored.`,
      );
    } catch (caught) {
      setError(messageOf(caught));
    } finally {
      setBusy(null);
    }
  }

  async function onReview() {
    if (!selectedTx) return;
    setBusy("review");
    setError(null);
    try {
      const saved = await recordReview({
        data: {
          scoredTransactionId: selectedTx.id,
          decision,
          reasonCode,
        },
      });
      setQueue((current) =>
        current.map((row) =>
          row.id === saved.scoredTransactionId
            ? { ...row, decision: saved.decision, reasonCode: saved.reasonCode }
            : row,
        ),
      );
      setEvents((current) => [saved.event, ...current].slice(0, 20));
      setNotice("Review stored on this synthetic queue item.");
    } catch (caught) {
      setError(messageOf(caught));
    } finally {
      setBusy(null);
    }
  }

  async function onSpeak(clip?: { audioBase64: string; audioMime: string }, spoken?: string) {
    setBusy("doubt");
    setError(null);
    setVoiceNote(null);
    setLiveBrief(null);
    const asked = (spoken ?? question).trim();
    const payload = {
      question: asked || undefined,
      audioBase64: clip?.audioBase64,
      audioMime: clip?.audioMime,
      history: turns.slice(-4),
      item: selectedTx
        ? {
            transactionId: selectedTx.transactionId,
            amount: selectedTx.amount,
            hour: selectedTx.hour,
            velocity: selectedTx.velocity24h,
            distanceKm: selectedTx.distanceKm,
            newDevice: selectedTx.newDevice === 1,
            score: selectedTx.anomalyScore,
            flagged: selectedTx.flagged,
            reasons: selectedTx.reviewReasons,
            decision: selectedTx.decision,
            reasonCode: selectedTx.reasonCode,
            injected: showInjected ? selectedTx.injectedScenario : undefined,
          }
        : null,
      model: selectedModel
        ? {
            label: selectedModel.versionLabel,
            auc: selectedModel.rocAuc,
            precision: selectedModel.precision,
            recall: selectedModel.recall,
            threshold: selectedModel.threshold,
            flagRate: selectedModel.testFlagRate,
          }
        : null,
      drift: drift.map((row) => ({ feature: row.feature, status: row.status, smd: row.smd })),
      slices: slices.map((row) => ({ name: row.name, n: row.n, gap: row.gap })),
    };
    const briefTask = quickBrief({ data: payload })
      .then((brief) => setLiveBrief(brief.text))
      .catch(() => undefined);
    try {
      const result = await answerDoubt({ data: payload });
      await briefTask;
      const userLine = result.heard || asked;
      setTurns((current) =>
        [...current, ...(userLine ? [{ role: "user" as const, text: userLine }] : []), { role: "assistant" as const, text: result.text }].slice(-6),
      );
      setQuestion("");
      setBusy(null);
      const clipAudio = await speakText({ data: { text: result.text } });
      setBriefing({ text: result.text, src: `data:audio/mpeg;base64,${clipAudio.audioBase64}` });
    } catch (caught) {
      setError(messageOf(caught));
      setVoiceNote(messageOf(caught));
      setBusy(null);
    }
  }

  speakRef.current = onSpeak;

  function stopListening() {
    if (stopTimer.current) window.clearInterval(stopTimer.current);
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
      return;
    }
    setListening(false);
  }

  function onTalk() {
    if (listening) {
      stopListening();
      return;
    }
    setVoiceNote(null);
    setError(null);
    setHeardLive("");
    void recordClip();
  }

  async function recordClip() {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setVoiceNote("This browser has no microphone. Type the doubt instead.");
      return;
    }
    setListening(true);
    heardSpeech.current = false;
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setListening(false);
      const name = err instanceof DOMException ? err.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        setVoiceNote("Allow the microphone for this page, then tap Talk again.");
      } else if (name === "NotFoundError") {
        setVoiceNote("No microphone was found. Type the doubt instead.");
      } else {
        setVoiceNote("The microphone did not start. Type the doubt instead.");
      }
      return;
    }
    const preferred = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((type) => MediaRecorder.isTypeSupported(type));
    const rec = preferred ? new MediaRecorder(stream, { mimeType: preferred }) : new MediaRecorder(stream);
    chunksRef.current = [];
    const audio = new AudioContext();
    await audio.resume();
    const meter = audio.state === "running";
    const analyser = audio.createAnalyser();
    analyser.fftSize = 512;
    audio.createMediaStreamSource(stream).connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);
    let quiet = 0;
    rec.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    rec.onstop = () => {
      if (stopTimer.current) window.clearInterval(stopTimer.current);
      stream.getTracks().forEach((track) => track.stop());
      void audio.close();
      setListening(false);
      if (meter && !heardSpeech.current) {
        setVoiceNote("I didn't hear anything. Tap Talk and speak.");
        return;
      }
      const mime = rec.mimeType || "audio/webm";
      const blob = new Blob(chunksRef.current, { type: mime });
      const reader = new FileReader();
      reader.onload = () => {
        const raw = String(reader.result ?? "");
        const audioBase64 = raw.slice(raw.indexOf(",") + 1);
        void speakRef.current({ audioBase64, audioMime: mime });
      };
      reader.onerror = () => setVoiceNote("The recording could not be read.");
      reader.readAsDataURL(blob);
    };
    rec.start(200);
    recorderRef.current = rec;
    const started = performance.now();
    stopTimer.current = window.setInterval(() => {
      if (rec.state !== "recording") return;
      analyser.getByteTimeDomainData(samples);
      let energy = 0;
      for (let index = 0; index < samples.length; index += 1) {
        const sample = (samples[index] - 128) / 128;
        energy += sample * sample;
      }
      const loud = Math.sqrt(energy / samples.length) > 0.015;
      if (loud) {
        if (!heardSpeech.current) setHeardLive("…");
        heardSpeech.current = true;
        quiet = 0;
      } else if (heardSpeech.current) {
        quiet += 100;
      }
      const elapsed = performance.now() - started;
      if ((heardSpeech.current && quiet >= 700) || elapsed > 8000) rec.stop();
    }, 100);
  }

  return (
    <main className="min-h-screen bg-ink text-paper">
      <header className="border-b border-line">
        <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="flex gap-4">
            <span className="mt-1 hidden w-1 shrink-0 self-stretch bg-brass sm:block" aria-hidden="true" />
            <div>
              <p className="font-mono text-xs tracking-widest text-brass">PHASE 3 · OVERSIGHT</p>
              <h1 className="mt-1 font-display text-4xl font-semibold text-paper sm:text-5xl">RiskLens</h1>
              <p className="mt-2 max-w-xl text-sm leading-relaxed text-mist">
                Holdout scoring, with drift, slice gaps, and an append-only record of registrations, reviews, and file scores.
                Synthetic books only. A high score is a review prompt, not a fraud verdict.
              </p>
            </div>
          </div>
          <div className="rounded-lg border border-line bg-panel px-4 py-3">
            <p className="text-xs tracking-wide text-mist">Active contract</p>
            <p className="mt-1 font-display text-2xl text-paper">{CONTRACT_VERSION}</p>
            <p className="mt-1 text-xs text-mist">
              {selectedModel ? `${selectedModel.versionLabel} · v${selectedModel.id}` : "No model registered"}
            </p>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,1fr)_18rem] lg:items-start">
        <div className="flex min-w-0 flex-col gap-6">
        <section className="grid gap-3 md:grid-cols-3">
          <ServiceCard
            kicker="01"
            title="Scoring"
            copy="Trees are grown on days 1–21. Days 22–30, and any uploaded file, are scored with that frozen artifact."
          />
          <ServiceCard
            kicker="02"
            title="Registry"
            copy="Each run stores estimators, the threshold, holdout precision, and the forest itself. Promoting a version archives the previous one."
          />
          <ServiceCard
            kicker="03"
            title="Review"
            copy="Clear, hold, or escalate a saved synthetic item with a reason code. Uploaded files are scored and then discarded."
          />
        </section>

        <section className="rounded-lg border border-line bg-panel p-4 sm:p-5">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
            <h2 className="font-display text-2xl">Register a model</h2>
            <p className="text-xs text-mist">Train window days 1–21 · holdout days 22–30</p>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Field label="Transactions" value={n} min={200} max={1600} step={100} onChange={setN} />
            <Field label="Seed" value={seed} min={0} max={1000000} step={1} onChange={setSeed} />
            <Field
              label="Contamination"
              value={contamination}
              min={0.01}
              max={0.2}
              step={0.01}
              onChange={setContamination}
            />
            <Field label="Trees" value={trees} min={24} max={80} step={8} onChange={setTrees} />
          </div>
          <div className="mt-4 flex flex-col gap-3 sm:flex-row">
            <button
              type="button"
              onClick={onRegister}
              disabled={busy !== null}
              className="inline-flex min-h-11 items-center justify-center rounded-lg bg-brass px-4 text-sm font-medium text-brass-ink transition-opacity disabled:opacity-50"
            >
              {busy === "register" ? "Fitting holdout model…" : "Fit and register"}
            </button>
            <label className="inline-flex min-h-11 cursor-pointer items-center justify-center rounded-lg border border-line bg-panel-2 px-4 text-sm text-paper">
              Score a CSV
              <input
                type="file"
                accept=".csv,text/csv"
                className="sr-only"
                disabled={busy !== null}
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  event.target.value = "";
                  void onFile(file);
                }}
              />
            </label>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-mist">
            CSV columns: amount, hour, velocity_24h, distance_km, new_device. Files are validated against {CONTRACT_VERSION} and are not written to the registry.
          </p>
        </section>

        {error ? (
          <p role="alert" className="rounded-lg border border-brass bg-panel px-4 py-3 text-sm text-paper">
            {error}
          </p>
        ) : null}
        {notice ? <p className="rounded-lg border border-line bg-panel-2 px-4 py-3 text-sm text-mist">{notice}</p> : null}

        {loading ? (
          <div className="grid gap-3 sm:grid-cols-4" aria-busy="true">
            {Array.from({ length: 4 }, (_, index) => (
              <div key={index} className="h-24 animate-pulse rounded-lg bg-panel" />
            ))}
          </div>
        ) : (
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Metric label="Holdout rows" value={String(selectedModel?.nTest ?? 0)} hint={selectedModel ? `Train ${selectedModel.nTrain}` : "Register a model"} />
            <Metric label="Holdout flagged" value={pct(selectedModel?.testFlagRate)} hint={`${gbp.format(flaggedValue)} flagged value`} />
            <Metric label="Precision" value={pct(selectedModel?.precision)} hint="Against injected scenarios, not verified fraud" />
            <Metric label="ROC AUC" value={selectedModel?.rocAuc === null || selectedModel?.rocAuc === undefined ? "n/a" : selectedModel.rocAuc.toFixed(2)} hint={`${reviewed.length} of ${holdoutFlagged.length} flagged reviewed`} />
          </section>
        )}

        <section className="rounded-lg border border-line bg-panel p-4 sm:p-5">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
            <h2 className="font-display text-2xl">Oversight</h2>
            <p className="text-xs text-mist">Train window versus holdout. Shift means a standardized mean difference of 0.5 or more.</p>
          </div>
          {queue.length === 0 ? (
            <p className="mt-4 text-sm text-mist">Register a model to compare windows and slices.</p>
          ) : (
            <div className="mt-4 grid gap-6 lg:grid-cols-2">
              <div>
                <h3 className="text-sm font-medium text-paper">Feature drift</h3>
                <ul className="mt-3 flex flex-col gap-2">
                  {drift.map((row) => (
                    <li key={row.feature} className="grid grid-cols-2 items-baseline gap-2 text-sm sm:grid-cols-4">
                      <span className="text-paper">{row.feature}</span>
                      <span className="tabular-nums text-mist">{row.trainMean.toFixed(1)}</span>
                      <span className="tabular-nums text-mist">{row.testMean.toFixed(1)}</span>
                      <span className={row.status === "stable" ? "text-mist" : "text-brass"}>
                        {row.status} {row.smd >= 0 ? "+" : ""}
                        {row.smd.toFixed(2)}
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="mt-2 text-xs text-mist">Columns: feature, train mean, holdout mean, shift.</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-paper">Holdout slice gaps</h3>
                <ul className="mt-3 flex flex-col gap-2">
                  {slices.map((row) => (
                    <li key={row.name} className="grid grid-cols-3 items-baseline gap-2 text-sm">
                      <span className="text-paper">
                        {row.name}
                        <span className="mt-0.5 block text-xs text-mist">n {row.n}</span>
                      </span>
                      <span className="tabular-nums text-mist">{pct(row.flagRate)}</span>
                      <span className={Math.abs(row.gap) >= 0.15 && row.n >= 15 ? "text-brass" : "text-mist"}>
                        gap {row.gap >= 0 ? "+" : ""}
                        {(row.gap * 100).toFixed(1)} pp
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="mt-2 text-xs text-mist">Flag rate inside the slice versus the rest of the holdout. A gap is not a fairness verdict.</p>
              </div>
            </div>
          )}
          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <div>
              <h3 className="text-sm font-medium text-paper">Holdout AUC by version</h3>
              {trend.length === 0 ? (
                <p className="mt-3 text-sm text-mist">No versions yet.</p>
              ) : (
                <ul className="mt-3 flex flex-col gap-2">
                  {trend.map((model) => (
                    <li key={model.id} className="flex items-center gap-2 text-sm">
                      <span className="w-12 text-mist">v{model.id}</span>
                      <span className="h-2 flex-1 overflow-hidden rounded bg-ink">
                        <span
                          className="block h-2 bg-brass"
                          style={{ width: `${Math.round((model.rocAuc ?? 0) * 100)}%` }}
                        />
                      </span>
                      <span className="w-10 tabular-nums text-paper">{model.rocAuc === null ? "n/a" : model.rocAuc.toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div>
              <h3 className="text-sm font-medium text-paper">Audit</h3>
              {events.length === 0 ? (
                <p className="mt-3 text-sm text-mist">No events yet. Registrations, reviews, and file scores are appended here and are not deleted.</p>
              ) : (
                <ul className="mt-3 flex flex-col gap-2">
                  {events.map((event) => (
                    <li key={event.id} className="text-sm">
                      <span className="text-mist">{event.createdAt.slice(11, 19)} </span>
                      <span className="text-paper">{event.summary}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          <p className="mt-4 text-xs leading-relaxed text-mist">
            The audit is append-only. Uploaded files are scored and discarded. Reviews store a decision and a reason code, not a free-text note. Nothing here is an automated adverse decision.
          </p>
        </section>

        <div className="grid gap-6 lg:grid-cols-5">
          <section className="rounded-lg border border-line bg-panel p-4 lg:col-span-3">
            <div className="flex items-baseline justify-between gap-3">
              <h2 className="font-display text-2xl">{live ? "Uploaded scores" : "Holdout landscape"}</h2>
              <p className="text-xs text-mist">Higher score, more unusual</p>
            </div>
            <div className="mt-4 h-72 w-full">
              {plotRows.length === 0 ? (
                <p className="flex h-full items-center text-sm text-mist">
                  {loading
                    ? "Loading the registry…"
                    : models.length === 0
                      ? "Register a model to plot the book."
                      : "No scored points in this window."}
                </p>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                    <CartesianGrid stroke="var(--color-line)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="amount"
                      type="number"
                      scale="log"
                      domain={["auto", "auto"]}
                      name="Amount"
                      tick={{ fill: "var(--color-mist)", fontSize: 12 }}
                      axisLine={{ stroke: "var(--color-line)" }}
                      tickLine={false}
                    />
                    <YAxis
                      dataKey="score"
                      type="number"
                      domain={["auto", "auto"]}
                      tick={{ fill: "var(--color-mist)", fontSize: 12 }}
                      axisLine={{ stroke: "var(--color-line)" }}
                      tickLine={false}
                      width={36}
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: "3 3", stroke: "var(--color-mist)" }}
                      content={<PlotTip />}
                    />
                    <Scatter data={calm} fill="var(--color-mist)" />
                    <Scatter data={hot} fill="var(--color-brass)" />
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </div>
            <p className="mt-2 flex gap-4 text-xs text-mist">
              <span className="inline-flex items-center gap-2"><i className="inline-block size-2 rounded-full bg-mist" /> Inside threshold</span>
              <span className="inline-flex items-center gap-2"><i className="inline-block size-2 rounded-full bg-brass" /> Flagged</span>
            </p>
          </section>

          <section className="rounded-lg border border-line bg-panel p-4 lg:col-span-2">
            <h2 className="font-display text-2xl">Registry</h2>
            {models.length === 0 && !loading ? (
              <p className="mt-4 text-sm text-mist">No versions yet. The first fit becomes the active model.</p>
            ) : (
              <ul className="mt-4 flex flex-col gap-2">
                {models.map((model) => {
                  const current = model.id === selectedModelId;
                  return (
                    <li key={model.id}>
                      <button
                        type="button"
                        onClick={() => void onSelectModel(model.id)}
                        className={`flex w-full min-h-11 flex-col gap-1 rounded-lg border px-3 py-2 text-left ${
                          current ? "border-brass bg-panel-2" : "border-line bg-ink"
                        }`}
                      >
                        <span className="flex items-center justify-between gap-2">
                          <span className="text-sm font-medium text-paper">
                            {model.versionLabel} · v{model.id}
                          </span>
                          <span
                            className={`rounded px-2 py-0.5 text-xs ${
                              model.status === "active" ? "bg-brass text-brass-ink" : "text-mist"
                            }`}
                          >
                            {model.status}
                          </span>
                        </span>
                        <span className="text-xs text-mist tabular-nums">
                          AUC {model.rocAuc === null ? "n/a" : model.rocAuc.toFixed(2)} · precision {pct(model.precision)} · recall {pct(model.recall)}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        </div>

        {selectedTx && !live ? (
          <section className="rounded-lg border border-brass bg-panel p-4 sm:p-5">
            <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
              <h2 className="font-display text-2xl">{selectedTx.transactionId}</h2>
              <p className="text-sm tabular-nums text-mist">
                {selectedTx.customerId} · day {selectedTx.eventDay + 1} · {selectedTx.split}
              </p>
            </div>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3 lg:grid-cols-6">
              <Fact label="Amount" value={gbp.format(selectedTx.amount)} />
              <Fact label="Score" value={selectedTx.anomalyScore.toFixed(3)} />
              <Fact label="Hour" value={String(selectedTx.hour)} />
              <Fact label="Velocity" value={String(selectedTx.velocity24h)} />
              <Fact label="Distance" value={`${selectedTx.distanceKm.toFixed(1)} km`} />
              <Fact label="New device" value={selectedTx.newDevice ? "Yes" : "No"} />
            </dl>
            <p className="mt-4 text-sm text-paper">{selectedTx.reviewReasons}</p>
            {showInjected ? (
              <p className="mt-2 text-xs text-mist">
                Synthetic scenario flag: {selectedTx.injectedScenario ? "injected" : "not injected"}. Not used to train the forest.
              </p>
            ) : null}
            <div className="mt-4 flex flex-col gap-3">
              <div className="flex flex-wrap gap-2">
                {DECISIONS.map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setDecision(value)}
                    className={`min-h-11 rounded-lg px-3 text-sm ${
                      decision === value ? "bg-brass text-brass-ink" : "border border-line text-paper"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="flex flex-wrap gap-2">
                {REASONS.map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setReasonCode(value)}
                    className={`min-h-11 rounded-lg px-3 text-sm ${
                      reasonCode === value ? "bg-panel-2 text-paper ring-1 ring-brass" : "border border-line text-mist"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => void onReview()}
                disabled={busy !== null}
                className="inline-flex min-h-11 w-full items-center justify-center rounded-lg bg-brass px-4 text-sm font-medium text-brass-ink disabled:opacity-50 sm:w-auto"
              >
                {busy === "review" ? "Saving…" : "Save review"}
              </button>
            </div>
          </section>
        ) : null}

        <section className="rounded-lg border border-line bg-panel p-4 sm:p-5">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
            <h2 className="font-display text-2xl">Voice assistant</h2>
            <p className="text-xs text-mist">
              {selectedTx ? `Listening about ${selectedTx.transactionId}` : "No case open. It will speak about the model."}
            </p>
          </div>
          <p className="mt-2 max-w-2xl text-sm text-mist">
            A Python brief appears at once, then the spoken answer. It does not decide the review.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={onTalk}
              disabled={busy !== null || loading}
              className={`inline-flex min-h-11 items-center justify-center rounded-lg px-4 text-sm font-medium disabled:opacity-50 ${
                listening ? "bg-brass text-brass-ink" : "border border-brass text-paper"
              }`}
            >
              {listening ? "Listening… tap to send" : "Talk"}
            </button>
            <button
              type="button"
              onClick={() => void onSpeak()}
              disabled={busy !== null || loading || listening}
              className="inline-flex min-h-11 items-center justify-center rounded-lg bg-brass px-4 text-sm font-medium text-brass-ink disabled:opacity-50"
            >
              {busy === "doubt" ? "Answering…" : "Ask in text"}
            </button>
          </div>
          <label className="mt-4 block text-sm text-mist">
            Or type it
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value.slice(0, 280))}
              placeholder="Is the amount the doubt, or the new device?"
              className="mt-1 min-h-11 w-full rounded-lg border border-line bg-ink px-3 text-sm text-paper placeholder:text-mist"
            />
          </label>
          {liveBrief ? <p className="mt-3 text-sm text-paper">Now · {liveBrief}</p> : null}
          {voiceNote ? <p className="mt-3 text-sm text-brass">{voiceNote}</p> : null}
          {heardLive ? <p className="mt-3 text-sm text-paper">Hearing: {heardLive}</p> : null}
          {turns.length > 0 ? (
            <ul className="mt-4 flex flex-col gap-3">
              {turns.map((turn, index) => (
                <li key={`${turn.role}-${index}`} className="text-sm leading-relaxed">
                  <span className="text-mist">{turn.role === "user" ? "You" : "Assistant"} · </span>
                  <span className="text-paper">{turn.text}</span>
                </li>
              ))}
            </ul>
          ) : null}
          {briefing ? <audio ref={voiceRef} className="mt-4 w-full" controls autoPlay src={briefing.src} /> : null}
        </section>

        <section className="rounded-lg border border-line bg-panel p-4 sm:p-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 className="font-display text-2xl">{live ? "Unsaved score" : "Investigation queue"}</h2>
              <p className="mt-1 text-xs text-mist">
                {live
                  ? "These rows exist only in this session."
                  : "Heuristic reasons are threshold context, not an explanation of the forest."}
              </p>
            </div>
            {live ? (
              <button
                type="button"
                onClick={() => setLive(null)}
                className="inline-flex min-h-11 items-center justify-center rounded-lg border border-line px-3 text-sm text-paper"
              >
                Back to saved queue
              </button>
            ) : (
              <div className="flex flex-col gap-2 text-sm text-mist">
                <Toggle label="Flagged only" checked={flaggedOnly} onChange={setFlaggedOnly} />
                <Toggle label="Holdout only" checked={holdoutOnly} onChange={setHoldoutOnly} />
                <Toggle label="Show scenario flags" checked={showInjected} onChange={setShowInjected} />
              </div>
            )}
          </div>

          {live ? (
            <ul className="mt-4 flex flex-col gap-2">
              {live.rows.map((row, index) => (
                <li key={index} className="rounded-lg border border-line bg-ink px-3 py-3 text-sm">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="tabular-nums text-paper">{gbp.format(row.amount)}</span>
                    <span className={row.flagged ? "text-brass" : "text-mist"}>
                      {row.flagged ? "Flagged" : "Clear"} · {row.anomalyScore.toFixed(3)}
                    </span>
                  </div>
                  <p className="mt-1 text-mist">{row.reviewReasons}</p>
                </li>
              ))}
            </ul>
          ) : visible.length === 0 ? (
            <p className="mt-4 text-sm text-mist">
              {loading ? "Loading the queue…" : "Nothing in this view. Register a model, or widen the filters."}
            </p>
          ) : (
            <>
              <ul className="mt-4 flex flex-col gap-2 md:hidden">
                {visible.slice(0, 80).map((row) => (
                  <li key={row.id}>
                    <QueueButton row={row} selected={row.id === selectedTxId} showInjected={showInjected} onSelect={setSelectedTxId} />
                  </li>
                ))}
              </ul>
              <div className="mt-4 hidden overflow-x-auto md:block">
                <table className="w-full text-left text-sm">
                  <thead className="text-xs text-mist">
                    <tr className="border-b border-line">
                      <th className="py-2 pr-3 font-medium">Transaction</th>
                      <th className="py-2 pr-3 font-medium">Amount</th>
                      <th className="py-2 pr-3 font-medium">Score</th>
                      <th className="py-2 pr-3 font-medium">Reasons</th>
                      <th className="py-2 font-medium">Review</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visible.slice(0, 80).map((row) => (
                      <tr key={row.id} className="border-b border-line">
                        <td className="py-2 pr-3">
                          <button type="button" className="min-h-11 text-left text-paper" onClick={() => setSelectedTxId(row.id)}>
                            {row.transactionId}
                            <span className="mt-0.5 block text-xs text-mist">
                              {row.customerId}
                              {showInjected ? ` · ${row.injectedScenario ? "injected" : "ordinary"}` : ""}
                            </span>
                          </button>
                        </td>
                        <td className="py-2 pr-3 tabular-nums">{gbp.format(row.amount)}</td>
                        <td className={`py-2 pr-3 tabular-nums ${row.flagged ? "text-brass" : "text-mist"}`}>
                          {row.anomalyScore.toFixed(3)}
                        </td>
                        <td className="max-w-xs py-2 pr-3 text-mist">{row.reviewReasons}</td>
                        <td className="py-2 text-paper">{row.decision ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {visible.length > 80 ? (
                <p className="mt-3 text-xs text-mist">Showing the first 80 of {visible.length} in this view, highest score first.</p>
              ) : null}
            </>
          )}
        </section>

        <section className="rounded-lg border border-line bg-panel p-4 sm:p-5">
          <h2 className="font-display text-2xl">Feature contract</h2>
          <p className="mt-1 text-sm text-mist">
            {CONTRACT_VERSION} is the only shape the scoring service accepts. Amount and distance are log-transformed inside the forest; the other fields stay as recorded.
          </p>
          <dl className="mt-4 grid gap-3 sm:grid-cols-2">
            {CONTRACT_FIELDS.map((field) => (
              <div key={field.name} className="rounded-lg border border-line bg-ink px-3 py-3">
                <dt className="font-mono text-xs text-brass">{field.name}</dt>
                <dd className="mt-1 text-sm text-paper">{field.rule}</dd>
              </div>
            ))}
          </dl>
        </section>
        </div>
        <DeskGuide
          model={selectedModel}
          transaction={selectedTx}
          drifted={drift.filter((row) => row.status !== "stable").map((row) => row.feature)}
        />
      </div>
      <ScrollScout />
    </main>
  );
}

function ScrollScout() {
  const rider = useRef<HTMLDivElement>(null);
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    if (sessionStorage.getItem("risklens-scout") === "min") setHidden(true);
  }, []);

  useEffect(() => {
    if (hidden) return;
    const el = rider.current;
    if (!el) return;
    let frame = 0;
    const place = () => {
      frame = 0;
      const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      const progress = Math.min(1, Math.max(0, window.scrollY / max));
      const travel = Math.max(0, window.innerWidth - 176);
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      el.style.transform = `translateX(${progress * travel}px) rotate(${reduce ? 0 : progress * 28 - 10}deg)`;
    };
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(place);
    };
    place();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [hidden]);

  function hide() {
    sessionStorage.setItem("risklens-scout", "min");
    setHidden(true);
  }

  function show() {
    sessionStorage.setItem("risklens-scout", "open");
    setHidden(false);
  }

  if (hidden) {
    return (
      <button
        type="button"
        onClick={show}
        className="fixed bottom-4 left-4 z-30 inline-flex min-h-11 min-w-11 items-center justify-center rounded-full border border-brass bg-panel text-sm text-brass"
        aria-label="Show the scroll scout"
      >
        <ScoutMark />
      </button>
    );
  }

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-4 z-30 h-16">
      <div ref={rider} className="pointer-events-auto absolute bottom-0 left-3 flex items-center gap-2">
        <div className="grid h-14 w-14 place-items-center rounded-full border border-brass bg-panel shadow-lg">
          <ScoutMark />
        </div>
        <button
          type="button"
          onClick={hide}
          className="inline-flex min-h-11 items-center rounded-full border border-line bg-ink px-3 text-sm text-mist"
        >
          Minimize
        </button>
      </div>
    </div>
  );
}

function ScoutMark() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
      <circle cx="14" cy="14" r="8" fill="none" stroke="#e0a34a" strokeWidth="1.6" />
      <circle cx="14" cy="14" r="2" fill="#e0a34a" />
      <path d="M14 2.5v4.2M14 21.3v4.2M2.5 14h4.2M21.3 14h4.2" stroke="#9aafa4" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function DeskGuide({
  model,
  transaction,
  drifted,
}: {
  model: ModelSummary | null;
  transaction: QueueRow | null;
  drifted: string[];
}) {
  const which = [
    model
      ? `${model.versionLabel} is the active model, version ${model.id}, threshold ${model.threshold.toFixed(3)}.`
      : "No model is registered yet. Fit one before a score means anything.",
    transaction
      ? `${transaction.transactionId} is the open row, score ${transaction.anomalyScore.toFixed(3)}${transaction.flagged ? ", flagged" : ", not flagged"}${transaction.decision ? `, marked ${transaction.decision}` : ""}.`
      : "No row is open. Choose one from the queue.",
  ].join(" ");
  const when = drifted.length
    ? `Look again now. ${drifted.join(", ")} moved between the train window and the holdout.`
    : "Review a row when it is flagged. Come back to drift when a feature leaves stable.";
  const notes = [
    ["Why", "Someone has to be allowed to doubt a score. This book is synthetic, so that practice never names a real customer. A high score is a prompt to look, not a finding of fraud."],
    ["What", "RiskLens fits an Isolation Forest, freezes it, and scores a later window. Flagged rows wait in a queue. Drift, slice gaps, and an audit line show whether the book or the review moved."],
    ["How", "Days 1–21 grow the trees. Days 22–30 are scored with that frozen forest. A CSV is scored, then discarded. Talk or type a doubt: a short brief appears first, then the voice. You still choose clear, hold, or escalate."],
    ["Which", which],
    ["When", when],
  ] as const;
  return (
    <aside className="order-first lg:sticky lg:top-4 lg:order-none lg:max-h-[calc(100vh-2rem)] lg:overflow-y-auto">
      <div className="rounded-lg border border-line bg-panel p-4">
        <p className="font-mono text-xs tracking-widest text-brass">DESK</p>
        <h2 className="mt-1 font-display text-2xl">Why this is here</h2>
        <ul className="mt-4 flex flex-col gap-4">
          {notes.map(([title, copy]) => (
            <li key={title}>
              <p className="text-xs tracking-wide text-brass">{title}</p>
              <p className="mt-1 text-sm leading-relaxed text-paper">{copy}</p>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}

function ServiceCard({ kicker, title, copy }: { kicker: string; title: string; copy: string }) {
  return (
    <article className="rounded-lg border border-line bg-panel px-4 py-4">
      <p className="font-mono text-xs text-brass">{kicker}</p>
      <h2 className="mt-1 font-display text-xl text-paper">{title}</h2>
      <p className="mt-2 text-sm leading-relaxed text-mist">{copy}</p>
    </article>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <article className="rounded-lg border border-line bg-panel px-4 py-3">
      <p className="text-xs text-mist">{label}</p>
      <p className="mt-1 font-display text-3xl tabular-nums text-paper">{value}</p>
      <p className="mt-1 text-xs text-mist">{hint}</p>
    </article>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-mist">{label}</dt>
      <dd className="mt-1 tabular-nums text-paper">{value}</dd>
    </div>
  );
}

function Field({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="flex min-h-11 flex-col justify-center gap-1 text-xs text-mist">
      <span className="flex items-center justify-between">
        {label}
        <span className="tabular-nums text-paper">{step < 1 ? value.toFixed(2) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="accent-brass"
      />
    </label>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="inline-flex min-h-11 items-center gap-2">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="size-4 accent-brass"
      />
      {label}
    </label>
  );
}

function QueueButton({
  row,
  selected,
  showInjected,
  onSelect,
}: {
  row: QueueRow;
  selected: boolean;
  showInjected: boolean;
  onSelect: (id: number) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(row.id)}
      className={`w-full rounded-lg border px-3 py-3 text-left ${
        selected ? "border-brass bg-panel-2" : "border-line bg-ink"
      }`}
    >
      <span className="flex items-baseline justify-between gap-3">
        <span className="text-sm text-paper">{row.transactionId}</span>
        <span className={`tabular-nums text-sm ${row.flagged ? "text-brass" : "text-mist"}`}>
          {row.anomalyScore.toFixed(3)}
        </span>
      </span>
      <span className="mt-1 block text-sm tabular-nums text-paper">{gbp.format(row.amount)}</span>
      <span className="mt-1 block text-xs text-mist">
        {row.reviewReasons}
        {row.decision ? ` · ${row.decision}` : ""}
        {showInjected ? ` · ${row.injectedScenario ? "injected" : "ordinary"}` : ""}
      </span>
    </button>
  );
}

function PlotTip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: { name?: string; amount?: number; score?: number; flagged?: boolean } }>;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <div className="rounded-lg border border-line bg-ink px-3 py-2 text-xs text-paper">
      <p>{point.name}</p>
      <p className="mt-1 tabular-nums text-mist">
        {gbp.format(point.amount ?? 0)} · score {(point.score ?? 0).toFixed(3)} · {point.flagged ? "flagged" : "inside"}
      </p>
    </div>
  );
}
