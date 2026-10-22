import { createServerFn } from "@tanstack/react-start";
import type { ZodType } from "zod";
import { doubtInputSchema, queueInputSchema, reviewInputSchema, scoreInputSchema, trainInputSchema } from "./contract";

function read<T>(schema: ZodType<T>, input: unknown): T {
  const result = schema.safeParse(input);
  if (!result.success) {
    throw new Error(result.error.issues[0]?.message ?? "Check the inputs and try again.");
  }
  return result.data;
}

export const getWorkspace = createServerFn({ method: "GET" }).handler(async () => {
  const { loadWorkspace } = await import("./repo.server");
  return loadWorkspace();
});

export const registerModel = createServerFn({ method: "POST" })
  .validator((input: unknown) => read(trainInputSchema, input))
  .handler(async ({ data }) => {
    const { registerModel: run } = await import("./repo.server");
    return run(data);
  });

export const getQueue = createServerFn({ method: "POST" })
  .validator((input: unknown) => read(queueInputSchema, input))
  .handler(async ({ data }) => {
    const { loadQueue } = await import("./repo.server");
    return loadQueue(data.modelId);
  });

export const scoreBatch = createServerFn({ method: "POST" })
  .validator((input: unknown) => read(scoreInputSchema, input))
  .handler(async ({ data }) => {
    const { scoreUpload } = await import("./repo.server");
    return scoreUpload(data);
  });

export const recordReview = createServerFn({ method: "POST" })
  .validator((input: unknown) => read(reviewInputSchema, input))
  .handler(async ({ data }) => {
    const { saveReview } = await import("./repo.server");
    return saveReview(data);
  });

export const speakDoubt = createServerFn({ method: "POST" })
  .validator((input: unknown) => read(doubtInputSchema, input))
  .handler(async ({ data }) => {
    const { speakDoubt: run } = await import("./assistant.server");
    return run(data);
  });
