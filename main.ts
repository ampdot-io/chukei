import {
    Application,
    Context,
    Next,
    proxy,
    Router,
} from "https://deno.land/x/oak/mod.ts";
import * as z from "npm:zod";
import os from "node:os";
import * as toml from "jsr:@std/toml";
import { walk } from "jsr:@std/fs/walk";
import { securePath } from "./util.ts";
import merge from "npm:merge-deep";
import { exists as fileExists } from "jsr:@std/fs/exists";
import { dirname } from "node:path";

const router = new Router();

function getConfigPath() {
    return os.homedir() + "/chukei.autoconfig";
}

async function ensureModelsDir() {
    await Deno.mkdir(getConfigPath(), { recursive: true });
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

const configSchema = z.looseObject({
    provider: z.string().optional(),
    api_base: z.string().optional(),
    api_key: z.string().optional(),
    headers: z.record(z.string(), z.string()).default({}),
    body: z.looseObject({}).default({}),
});

const globalConfigSchema = z.object({
    providers: z.looseObject({}).catchall(
        configSchema.omit({ provider: true }).required({ api_base: true })
            .extend({
                discoveryType: z.enum(["openai_models_list", "koboldcpp"])
                    .default("openai_models_list"),
            }),
    ).default({}),
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

async function handleRequest(ctx: Context, next: Next) {
    let globalConfig, req, config, originalBody;
    try {
        globalConfig = await loadGlobalConfig();
        originalBody = await ctx.request.body.json();
        req = requestSchema.parse(originalBody);
        const modelFileName = securePath(getConfigPath(), req.model + ".toml");
        if (modelFileName == null) {
            ctx.response.status = 400;
            ctx.response.body = { errors: ["don't fuck with me m8"] };
            return;
        }
        const decoder = new TextDecoder("utf-8");
        if (!await fileExists(modelFileName)) {
            // autoconfig
            for (
                const [providerName, provider] of Object.entries(
                    globalConfig.providers,
                )
            ) {
                let models;
                if (provider.discoveryType === "openai_models_list") {
                    try {
                        const fetchRes = await fetch(
                            provider.api_base + "/v1/models",
                            { headers: computeHeaders(provider) },
                        );
                        models = await fetchRes.json();
                        for (const model of models.data) {
                            if (
                                model.id === req.model ||
                                // OpenRouter does not consistently use the same casing
                                model.hugging_face_id.toLowerCase() === req.model.toLowerCase()
                            ) {
                                config = {
                                    provider: providerName,
                                    body: { model: model.id },
                                };
                                break;
                            }
                        }
                    } catch (error) {
                        console.log(providerName + req.model, error);
                    }
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
        }
        throw error;
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

// todo: other interfaces to chukei
await app.listen({
    port: 6011,
    // Deno docs are wrong and adding this is required to listen on public interfaces
    hostname: "0.0.0.0",
});
