import {
    Application,
    Context,
    Next,
    proxy,
    Router,
} from "https://deno.land/x/oak/mod.ts";
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

type ModelWithMetadata = hfHub.ModelEntry & {
    author?: string;
    tags?: string[];
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function withExponentialBackoff<T>(
    fn: () => Promise<T>,
    options: { maxAttempts?: number; initialDelayMs?: number } = {},
): Promise<T> {
    const maxAttempts = options.maxAttempts ?? 5;
    const initialDelayMs = options.initialDelayMs ?? 250;
    let attempt = 0;
    let lastError: unknown = null;
    while (attempt < maxAttempts) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            if (
                error instanceof hfHub.HubApiError &&
                [401, 403, 404].includes(error.statusCode)
            ) {
                throw error;
            }
            attempt++;
            if (attempt >= maxAttempts) {
                throw error;
            }
            const delay = initialDelayMs * 2 ** (attempt - 1);
            await sleep(delay);
        }
    }
    throw lastError ?? new Error("Unknown error during backoff operation");
}

async function collectRepoFiles(
    repoId: string,
): Promise<hfHub.ListFileEntry[]> {
    return await withExponentialBackoff(async () => {
        const entries: hfHub.ListFileEntry[] = [];
        for await (const entry of hfHub.listFiles({ repo: repoId })) {
            entries.push(entry);
        }
        return entries;
    });
}

async function mapWithConcurrency<T, R>(
    items: T[],
    concurrency: number,
    mapper: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
    const results: R[] = new Array(items.length);
    let nextIndex = 0;
    const workers = Array.from({
        length: Math.max(1, Math.min(concurrency, items.length)),
    }, () =>
        (async () => {
            while (true) {
                const currentIndex = nextIndex++;
                if (currentIndex >= items.length) break;
                results[currentIndex] = await mapper(
                    items[currentIndex],
                    currentIndex,
                );
            }
        })());
    await Promise.all(workers);
    return results;
}

function computeRepoPreference(
    model: ModelWithMetadata,
    quantPrefs: z.infer<typeof KoboldProvider>["quantization"],
): number {
    if (!quantPrefs.prefer_same_owner) return 0;
    const [owner] = model.name?.split("/") ?? [];
    if (
        owner && model.author &&
        owner.toLowerCase() === model.author.toLowerCase()
    ) {
        return quantPrefs.prefer_same_owner;
    }
    return 0;
}

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
        tiebreak_strategy: z.enum(["random", "popular"]).default("random"),
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
        await Deno.writeTextFile(path, "");
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
    model: ModelWithMetadata;
    file: hfHub.ListFileEntry;
    preferenceScore: number;
    quantInfo: QuantInfo;
}

function chooseQuantization(
    candidates: HfQuant[],
    strategy: "random" | "popular",
): HfQuant {
    if (candidates.length === 1 || strategy === "random") {
        return candidates[Math.floor(Math.random() * candidates.length)];
    }
    const maxDownloads = Math.max(
        ...candidates.map((candidate) => candidate.model.downloads ?? 0),
    );
    const mostPopular = candidates.filter((candidate) =>
        (candidate.model.downloads ?? 0) === maxDownloads
    );
    return mostPopular[Math.floor(Math.random() * mostPopular.length)];
}

async function discoverQuantization(
    modelId: string,
    provider: z.infer<typeof KoboldProvider>,
): Promise<HfQuant | null> {
    const preferences = provider.quantization;
    const seenRepos = new Set<string>();
    const candidates: HfQuant[] = [];
    let bestScore = -Infinity;
    const maxPossibleScore = preferences.prefer_correct_precision +
        preferences.prefer_imatrix +
        preferences.prefer_same_owner;

    const processRepo = async (model: ModelWithMetadata) => {
        if (seenRepos.has(model.name)) return;
        seenRepos.add(model.name);
        let files: hfHub.ListFileEntry[];
        try {
            files = await collectRepoFiles(model.name);
        } catch (error) {
            if (error instanceof hfHub.HubApiError) {
                if ([401, 403].includes(error.statusCode)) {
                    console.log(
                        `Skipping repo ${model.name} due to access restrictions`,
                    );
                    return;
                }
            }
            console.log("Error listing files for", model.name, error);
            return;
        }
        const ggufFiles = files.filter((file) =>
            file.type === "file" && file.path.endsWith(".gguf")
        );
        if (ggufFiles.length === 0) {
            return;
        }
        const repoPreference = computeRepoPreference(model, preferences);
        const quantResults = await mapWithConcurrency(
            ggufFiles,
            4,
            async (file) => {
                let quantInfo: QuantInfo = {};
                try {
                    quantInfo = await withExponentialBackoff(() =>
                        getQuantizationType(model.name, file.path)
                    );
                } catch (error) {
                    console.log(
                        `Failed to get quantization info for ${model.name}/${file.path}`,
                        error,
                    );
                }
                let score = repoPreference;
                if (quantInfo.quantLevel === preferences.precision) {
                    score += preferences.prefer_correct_precision;
                }
                if (file.path.toLowerCase().includes("imatrix")) {
                    score += preferences.prefer_imatrix;
                }
                const candidate: HfQuant = {
                    model,
                    file,
                    preferenceScore: score,
                    quantInfo,
                };
                console.log(
                    `Candidate quant ${model.name}/${file.path} @ ${candidate.quantInfo?.quantLevel} (preference score ${candidate.preferenceScore})`,
                );
                return candidate;
            },
        );
        for (const candidate of quantResults) {
            if (candidate.preferenceScore > bestScore) {
                bestScore = candidate.preferenceScore;
                candidates.splice(0, candidates.length, candidate);
            } else if (candidate.preferenceScore === bestScore) {
                candidates.push(candidate);
            }
        }
    };

    let baseModel: ModelWithMetadata | null = null;
    try {
        baseModel = await withExponentialBackoff(() =>
            hfHub.modelInfo({
                name: modelId,
                additionalFields: ["author", "tags"],
            })
        );
    } catch (error) {
        console.log("Unable to fetch model info for", modelId, error);
    }
    if (baseModel) {
        await processRepo(baseModel);
        if (bestScore >= maxPossibleScore && candidates.length > 0) {
            return chooseQuantization(
                candidates,
                preferences.tiebreak_strategy,
            );
        }
    }

    const quantTag = `base_model:quantized:${modelId}`;
    try {
        for await (
            const repo of hfHub.listModels({
                search: { tags: [quantTag, "gguf"] },
                additionalFields: ["author", "tags"],
                limit: 100,
            })
        ) {
            await processRepo(repo as ModelWithMetadata);
            if (bestScore >= maxPossibleScore && candidates.length > 0) {
                break;
            }
        }
    } catch (error) {
        console.log("Error searching for quantized repos for", modelId, error);
    }

    if (candidates.length === 0) return null;
    return chooseQuantization(candidates, preferences.tiebreak_strategy);
}

async function handleRequest(ctx: Context, next: Next) {
    let globalConfig, req, config, originalBody;
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
            // TODO: Does not take into account possibility of a model provider failing or handle failover
            console.log(`Attempting to autoconfigure ${req.model}`);
            // autoconfig
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
                                console.log(
                                    `Found ${req.model} on ${providerName}!`,
                                );
                                config = {
                                    provider: providerName,
                                    body: { model: model.id },
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
                    let selectedQuantization: HfQuant | null = null;
                    try {
                        selectedQuantization = await discoverQuantization(
                            req.model,
                            koboldProvider,
                        );
                    } catch (error) {
                        console.log(
                            "Error discovering quantization for",
                            req.model,
                            error,
                        );
                    }
                    if (!selectedQuantization) {
                        console.log("No quantizations found for", req.model);
                        continue;
                    }
                    console.log("Selected quantization", {
                        repo: selectedQuantization.model.name,
                        path: selectedQuantization.file.path,
                        preferenceScore: selectedQuantization.preferenceScore,
                        quantLevel: selectedQuantization.quantInfo?.quantLevel,
                    });

                    await ensureModelsPath();
                    const fileEntry = selectedQuantization.file;
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
                    const repoDir = modelsPath + "/" +
                        selectedQuantization.model.name;
                    await Deno.mkdir(repoDir, { recursive: true });
                    const localModelPath = repoDir + "/" +
                        selectedQuantization.file.path.split("/").pop();
                    if (!await fileExists(localModelPath)) {
                        const downloadUrl =
                            `https://huggingface.co/${selectedQuantization.model.name}/resolve/main/${selectedQuantization.file.path}`;
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
                if (config != null) {
                    break;
                }
            }
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
                stderr: "piped",
            });
            const proc = command.spawn();
            const stderrPath =
                `${getConfigPath()}/koboldcpp.${proc.pid}.stderr`;
            const stderrFile = await Deno.open(stderrPath, {
                create: true,
                write: true,
                truncate: true,
            });
            proc.stderr?.pipeTo(stderrFile.writable).catch((err) =>
                console.error("Failed to pipe koboldcpp stderr", err)
            );
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
    } catch (error) {
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
        } else {
            ctx.response.status = 500;
            ctx.response.body = {
                error: {
                    message: "Internal server error",
                    stack: error.stack,
                    code: 500,
                },
            };
        }
    }
    if (config.provider != null) {
        const provider = globalConfig.providers?.[config.provider];
        if (provider != null) {
            config = merge(config, provider);
        }
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
}

router.post("/v1/completions", handleRequest);
// this will be supported eventually but isn't a priority
router.post("/v1/chat/completions", handleRequest);

const app = new Application();
app.use(router.routes());
app.use(router.allowedMethods());

app.addEventListener("listen", ({ hostname, port }) => {
    console.log(`HTTP server listening on ${hostname}:${port}`);
});

// todo: other interfaces to chukei
await app.listen({
    port: 6011,
    // Deno docs are wrong and adding this is required to listen on public interfaces
    hostname: "0.0.0.0",
});
