import { z } from "zod";

export const CONTRACT_VERSION = "tx.v1";

export const FEATURES = [
  "amount",
  "hour",
  "velocity_24h",
  "distance_km",
  "new_device",
] as const;

export const featureSchema = z.object({
  amount: z.number().finite().nonnegative(),
  hour: z.number().int().min(0).max(23),
  velocity_24h: z.number().int().nonnegative().max(500),
  distance_km: z.number().finite().nonnegative(),
  new_device: z.union([z.literal(0), z.literal(1)]),
});

export type FeatureRow = z.infer<typeof featureSchema>;

export const trainInputSchema = z.object({
  n: z.number().int().min(200).max(1600),
  seed: z.number().int().min(0).max(1_000_000),
  contamination: z.number().gt(0).lte(0.2),
  trees: z.number().int().min(24).max(80),
});

export type TrainInput = z.infer<typeof trainInputSchema>;

export const scoreInputSchema = z.object({
  rows: z.array(featureSchema).min(1).max(400),
});

export const reviewInputSchema = z.object({
  scoredTransactionId: z.number().int().positive(),
  decision: z.enum(["clear", "hold", "escalate"]),
  reasonCode: z.enum([
    "amount",
    "velocity",
    "distance",
    "device",
    "overnight",
    "pattern",
  ]),
});

export const queueInputSchema = z.object({
  modelId: z.number().int().positive(),
});

export const doubtInputSchema = z.object({
  question: z.string().trim().max(280).optional(),
  audioBase64: z.string().max(1_600_000).optional(),
  audioMime: z.string().max(40).optional(),
  history: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        text: z.string().trim().min(1).max(500),
      }),
    )
    .max(6)
    .optional(),
  item: z
    .object({
      transactionId: z.string().min(1).max(40),
      amount: z.number().finite().nonnegative(),
      hour: z.number().int().min(0).max(23),
      velocity: z.number().int().nonnegative().max(500),
      distanceKm: z.number().finite().nonnegative(),
      newDevice: z.boolean(),
      score: z.number().finite().min(0).max(1),
      flagged: z.boolean(),
      reasons: z.string().max(400),
      decision: z.enum(["clear", "hold", "escalate"]).nullable(),
      reasonCode: z.string().max(24).nullable(),
      injected: z.boolean().optional(),
    })
    .nullable(),
  model: z
    .object({
      label: z.string().min(1).max(40),
      auc: z.number().nullable(),
      precision: z.number().nullable(),
      recall: z.number().nullable(),
      threshold: z.number().finite(),
      flagRate: z.number().finite(),
    })
    .nullable(),
  drift: z
    .array(
      z.object({
        feature: z.string().max(24),
        status: z.enum(["stable", "watch", "shift"]),
        smd: z.number().finite(),
      }),
    )
    .max(5),
  slices: z
    .array(
      z.object({
        name: z.string().max(24),
        n: z.number().int().nonnegative(),
        gap: z.number().finite(),
      }),
    )
    .max(3),
});

export const DAY_COUNT = 30;
export const TRAIN_LAST_DAY = 20;

export const CONTRACT_FIELDS = [
  { name: "amount", rule: "Pounds, zero or greater. Logged before scoring." },
  { name: "hour", rule: "Integer 0–23, left as recorded." },
  { name: "velocity_24h", rule: "Integer count of transactions in the prior day." },
  { name: "distance_km", rule: "Kilometres from the usual place. Logged before scoring." },
  { name: "new_device", rule: "1 if the device is new, otherwise 0." },
] as const;

export const investigateInputSchema = z.object({
  transactionId: z.string().min(1).max(40),
  customerId: z.string().min(1).max(40),
  amount: z.number().finite().nonnegative(),
  hour: z.number().int().min(0).max(23),
  velocity: z.number().int().nonnegative().max(500),
  distanceKm: z.number().finite().nonnegative(),
  newDevice: z.boolean(),
  score: z.number().finite(),
  note: z.string().max(240).optional(),
  history: z
    .array(
      z.object({
        transactionId: z.string().max(40),
        customerId: z.string().max(40),
        amount: z.number().finite().nonnegative(),
        hour: z.number().int().min(0).max(23),
      }),
    )
    .max(8),
});
