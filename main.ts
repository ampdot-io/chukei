import {
    Application,
    Context,
    Next,
    proxy,
    Router,
} from "https://deno.land/x/oak/mod.ts";
import { trace, SpanStatusCode } from "npm:@opentelemetry/api";
import * as z from "npm:zod";
import os from "node:os";
import * as toml from "https://deno.land/std@0.224.0/toml/mod.ts";
import { walk } from "https://deno.land/std@0.224.0/fs/walk.ts";
import { getQuantizationType, QuantInfo, securePath } from "./util.ts";
import merge from "npm:merge-deep";
import { exists as fileExists } from "https://deno.land/std@0.224.0/fs/exists.ts";
import { dirname } from "node:path";
import * as hfHub from "https://esm.sh/@huggingface/hub";

const router = new Router();

// Track running koboldcpp processes so we can manage memory
const runningModels = new Map<
    string,
    { proc: Deno.ChildProcess; lastUsed: number; memory: number; port: number }
>();

async function ensureModelsPath() {
    await Deno.mkdir(os.homedir() + "/models", { recursive: true });
}

function getConfigPath() {
    return os.homedir() + "/chukei.autoconfig";
}

async function ensureModelsDir() {
    await Deno.mkdir(getConfigPath(), { recursive: true });
}

async function waitForPort(port: number) {
    for (let i = 0; i < 60; i++) {
        try {
            const conn = await Deno.connect({ hostname: "127.0.0.1", port });
            conn.close();
            return;
        } catch {
            await new Promise((r) => setTimeout(r, 500));
        }
    }
    throw new Error(`Timeout waiting for port ${port}`);
}

const specialFiles = new Set(["config.toml"]);

router.get("/v1/models", async (ctx) => {
    await ensureModelsDir();
    const modelFiles = [];
    for await (
        const dirEntry of walk(getConfigPath(), {
            exts: ["toml"],
            canonicalize: false,
        })
    ) {
        if (dirEntry.isFile) {
            modelFiles.push(dirEntry.path.slice(getConfigPath().length + 1));
        }
    }
    ctx.response.body = {
        data: modelFiles.filter((fname) => !specialFiles.has(fname)).map(
            (fname) => ({ id: fname.slice(0, -".toml".length) }),
        ),
    };
});

// underscores for configuration
const configSchema = z.looseObject({
    // needs to be changed to support multiple providers for one model
    provider: z.string().optional(),
    api_base: z.string().optional(),
    api_key: z.string().optional(),
    headers: z.record(z.string(), z.string()).default({}),
    body: z.looseObject({}).default({}),
    kobold_path: z.string().optional(),
    model_path: z.string().optional(),
});

const baseProviderSchema = configSchema.omit({ provider: true }).required({
    api_base: true,
});

const KoboldProvider = baseProviderSchema.extend({
    discovery_type: z.literal("koboldcpp"),
    kobold_path: z.string(),
    quantization: z.object({
        precision: z.string().default("Q5_K_M"),
        prefer_correct_precision: z.number().default(10000),
        prefer_imatrix: z.number().default(100),
        prefer_same_owner: z.number().default(10),
        tiebreak_strategy: z.literal(["random", "popular"]),
    }),
});

const OpenAIProvider = baseProviderSchema.extend({
    discovery_type: z.literal("openai_models_list").optional().default(
        "openai_models_list",
    ),
});

const Provider = z.discriminatedUnion("discovery_type", [
    KoboldProvider,
    OpenAIProvider,
]);

const globalConfigSchema = z.object({
    providers: z.looseObject({}).catchall(Provider),
});

async function loadGlobalConfig() {
    const path = getConfigPath() + "/config.toml";
    try {
        await Deno.lstat(path);
    } catch (err) {
        if (!(err instanceof Deno.errors.NotFound)) {
            throw err;
        }
        await Deno.writeTextFile(path, "[providers]\n");
    }
    return globalConfigSchema.parse(toml.parse(await Deno.readTextFile(path)));
}

const requestSchema = z.object({
    model: z.string(),
});

type Headerable = {
    headers?: Record<string, string>;
    api_key?: string;
};

function computeHeaders(obj: Headerable): Record<string, string> {
    const headers = obj.headers ?? {};
    if (obj.api_key) {
        headers.authorization = "Bearer " + obj.api_key;
    }
    return headers;
}

interface HfQuant {
    model: hfHub.ModelEntry;
    files: hfHub.ListFileEntry[];
    path: string;
    preferenceScore: number;
    quantInfo: QuantInfo;
}

function betterQuantization(modelA: HfQuant, modelB: HfQuant) {
    const hasIMatrix = (file: hfHub.ListFileEntry) =>
        file.path.includes("imatrix");
    modelA.files.some(hasIMatrix);
    modelA.files.some(hasIMatrix);
}

async function handleRequest(ctx: Context, next: Next) {
    let globalConfig: z.infer<typeof globalConfigSchema>,
        req: z.infer<typeof requestSchema>,
        config: z.infer<typeof configSchema> | undefined,
        originalBody;
    try {
        globalConfig = await loadGlobalConfig();
        originalBody = await ctx.request.body.json();
        req = requestSchema.parse(originalBody);
        const running = runningModels.get(req.model);
        if (running) running.lastUsed = Date.now();
        const modelFileName = securePath(getConfigPath(), req.model + ".toml");
        if (modelFileName == null) {
            ctx.response.status = 400;
            ctx.response.body = { errors: ["don't fuck with me m8"] };
            return;
        }
        const decoder = new TextDecoder("utf-8");
        if (!await fileExists(modelFileName)) {
            console.log(`Attempting to autoconfigure ${req.model}`);
            // autoconfig
            await trace.getTracer("chukei").startActiveSpan("autoconfig", async (span) => {
                try {
                    span.setAttribute("model", req.model);
                    for (
                        const [providerName, provider] of Object.entries(
                            globalConfig.providers,
                        )
                    ) {
                        let models;
                        if (provider.discovery_type === "openai_models_list") {
                            try {
                                console.log(`Checking /v1/models for: ${providerName}`);
                                const fetchRes = await fetch(
                                    provider.api_base + "/v1/models",
                                    { headers: computeHeaders(provider) },
                                );
                                models = await fetchRes.json();
                                for (const model of models.data) {
                                    if (
                                        model?.id?.toLowerCase() ===
                                            req.model.toLowerCase() ||
                                        // OpenRouter does not consistently use the same casing
                                        model?.hugging_face_id?.toLowerCase() ===
                                            req.model.toLowerCase()
                                    ) {
                                        console.log("Found!");
                                        config = {
                                            provider: providerName,
                                            body: { model: model.id },
                                            headers: {},
                                            api_base: provider.api_base,
                                        };
                                        break;
                                    }
                                }
                            } catch (error) {
                                console.log(
                                    `Encountered error: ${providerName}`,
                                    req.model,
                                    error,
                                );
                            }
                        } else if (provider.discovery_type === "koboldcpp") {
                            const koboldProvider = provider as unknown as z.infer<
                                typeof KoboldProvider
                            >;
                            const quants: HfQuant[] = [];
                            try {
                                const files = await Array.fromAsync(
                                    hfHub.listFiles({ repo: req.model }),
                                );
                                const basePreference = files.some((file) =>
                                        file.path.includes("imatrix")
                                    )
                                    ? koboldProvider.quantization.prefer_imatrix
                                    : 0;
                                for (const fileEntry of files) {
                                    if (fileEntry.path.endsWith(".gguf")) {
                                        let preferenceScore = basePreference;
                                        const quantInfo = await getQuantizationType(
                                            req.model,
                                            fileEntry.path,
                                        );
                                        if (
                                            quantInfo?.quantLevel ===
                                                koboldProvider.quantization.precision
                                        ) {
                                            preferenceScore +=
                                                koboldProvider.quantization
                                                    .prefer_correct_precision;
                                        }
                                        const entry: HfQuant = {
                                            model: { id: req.model } as any,
                                            files,
                                            path: fileEntry.path,
                                            preferenceScore,
                                            quantInfo,
                                        };
                                        console.log("Quantization candidate", entry);
                                        quants.push(entry);
                                    }
                                }
                            } catch (err) {
                                console.log("Error listing files for", req.model, err);
                            }
                            const bestQuantizations = (() => {
                                const maxPref = Math.max(
                                    ...quants.map((quant) => quant.preferenceScore),
                                );
                                return quants.filter((quant) =>
                                    quant.preferenceScore === maxPref
                                );
                            })();
                            let selectedQuantization;
                            if (
                                koboldProvider.quantization.tiebreak_strategy ===
                                    "random"
                            ) {
                                selectedQuantization = bestQuantizations[
                                    Math.floor(
                                        Math.random() * bestQuantizations.length,
                                    )
                                ];
                            } else {
                                selectedQuantization = bestQuantizations[0];
                            }
                            console.log("Selected quantization", selectedQuantization);

                            await ensureModelsPath();
                            const fileEntry = selectedQuantization.files.find((f) =>
                                f.path === selectedQuantization.path
                            );
                            const requiredMem = fileEntry?.size ?? 0;
                            let available = Deno.systemMemoryInfo().available;
                            if (requiredMem > available) {
                                const entries = [...runningModels.entries()].sort((
                                    a,
                                    b,
                                ) => a[1].lastUsed - b[1].lastUsed);
                                for (const [modelName, info] of entries) {
                                    console.log("Killing LRU process", modelName);
                                    info.proc.kill("SIGKILL");
                                    runningModels.delete(modelName);
                                    available = Deno.systemMemoryInfo().available;
                                    if (requiredMem <= available) break;
                                }
                            }
                            const modelsPath = os.homedir() + "/models";
                            const localModelPath = modelsPath + "/" +
                                selectedQuantization.path.split("/").pop();
                            if (!await fileExists(localModelPath)) {
                                const downloadUrl =
                                    `https://huggingface.co/${selectedQuantization.model.id}/resolve/main/${selectedQuantization.path}`;
                                console.log("Downloading model", downloadUrl);
                                const res = await fetch(downloadUrl);
                                if (!res.ok || !res.body) {
                                    throw new Error("Failed to download model");
                                }
                                const file = await Deno.open(localModelPath, {
                                    create: true,
                                    write: true,
                                    truncate: true,
                                });
                                await res.body.pipeTo(file.writable);
                            }

                            const port = 55000 + runningModels.size;
                            const command = new Deno.Command(
                                koboldProvider.kobold_path,
                                {
                                    args: [
                                        "--multiuser",
                                        "--skiplauncher",
                                        "--port",
                                        String(port),
                                        "--model",
                                        localModelPath,
                                    ],
                                },
                            );
                            const proc = command.spawn();
                            runningModels.set(req.model, {
                                proc,
                                lastUsed: Date.now(),
                                memory: requiredMem,
                                port,
                            });
                            await waitForPort(port);
                            config = {
                                api_base: `http://127.0.0.1:${port}`,
                                headers: {},
                                body: {},
                                kobold_path: koboldProvider.kobold_path,
                                model_path: localModelPath,
                            };
                        }
                    }
                } catch (e) {
                    span.recordException(e as Error);
                    span.setStatus({ code: SpanStatusCode.ERROR });
                    throw e;
                } finally {
                    span.end();
                }
            });

            if (config == null) {
                ctx.response.status = 404;
                ctx.response.body = {
                    error: {
                        message: "Could not find a provider that supports " +
                            req.model,
                        providersTried: Object.keys(globalConfig.providers),
                    },
                };
            } else {
                await Deno.mkdir(dirname(modelFileName), { recursive: true });
                await Deno.writeTextFile(modelFileName, toml.stringify(config));
            }
        }
        config = configSchema.parse(
            toml.parse(decoder.decode(await Deno.readFile(modelFileName))),
        );
        if (
            !runningModels.has(req.model) &&
            config.kobold_path &&
            config.model_path
        ) {
            const stat = await Deno.stat(config.model_path).catch(() => null);
            const requiredMem = stat?.size ?? 0;
            let available = Deno.systemMemoryInfo().available;
            if (requiredMem > available) {
                const entries = [...runningModels.entries()].sort((a, b) =>
                    a[1].lastUsed - b[1].lastUsed
                );
                for (const [modelName, info] of entries) {
                    console.log("Killing LRU process", modelName);
                    info.proc.kill("SIGKILL");
                    runningModels.delete(modelName);
                    available = Deno.systemMemoryInfo().available;
                    if (requiredMem <= available) break;
                }
            }
            const port = 55000 + runningModels.size;
            const command = new Deno.Command(config.kobold_path, {
                args: [
                    "--multiuser",
                    "--skiplauncher",
                    "--port",
                    String(port),
                    "--model",
                    config.model_path,
                ],
            });
            const proc = command.spawn();
            runningModels.set(req.model, {
                proc,
                lastUsed: Date.now(),
                memory: requiredMem,
                port,
            });
            await waitForPort(port);
            config.api_base = `http://127.0.0.1:${port}`;
            await Deno.writeTextFile(modelFileName, toml.stringify(config));
        }

        if (config.provider != null) {
            const provider = globalConfig.providers?.[config.provider];
            if (provider != null) {
                config = merge(config, provider);
            }
        }
        if (!config || !config.api_base) {
             throw new Error("Configuration incomplete: missing api_base");
        }
        const modifiedBody = merge(originalBody, config.body);
        await proxy(config.api_base, {
            headers: computeHeaders(config),
            proxyHeaders: false,
            request: (req) => {
                if (modifiedBody) {
                    const newReq = new Request(req, {
                        body: JSON.stringify(modifiedBody),
                    });
                    return newReq;
                } else return req;
            },
        })(ctx, next);

    } catch (error) {
        // Get the current span to record the error
        const span = trace.getActiveSpan();
        if (span) {
            span.recordException(error as Error);
            span.setStatus({ code: SpanStatusCode.ERROR });
        }

        if (error instanceof z.ZodError) {
            ctx.response.status = 400;
            ctx.response.body = {
                error: {
                    message: "Invalid request or config",
                    issues: error.issues,
                    code: 400,
                },
            };
            return;
        }

        // Catch-all for other errors
        console.error("Unhandled error:", error);
        ctx.response.status = 500;
        ctx.response.body = {
            error: {
                message: "Internal Server Error",
                details: error instanceof Error ? error.message : String(error),
            },
        };
    }
}

router.post("/v1/completions", handleRequest);
// this will be supported eventually but isn't a priority
router.post("/v1/chat/completions", handleRequest);

const app = new Application();
app.use(router.routes());
app.use(router.allowedMethods());

// todo: other interfaces to chukei
await app.listen({
    port: 6011,
    // Deno docs are wrong and adding this is required to listen on public interfaces
    hostname: "0.0.0.0",
});
