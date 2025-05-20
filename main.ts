import * as hfHub from "https://esm.sh/@huggingface/hub";

const globalConfig = {
    defaultQuant: "Q6_K",
};

/* Basic primitives */
const FailedAttempt = Symbol();

function attempt(fn) {
    return function () {
        let res;
        try {
            res = fn.apply(null, arguments);
        } catch (err) {
            // todo: log it
            return FailedAttempt;
        }
        // todo: log the success
        return res;
    };
}

// todo: execution order strategies; random, parallel, intelligent (AI)
async function attemptForEach(iterator, fn) {
    for await (const member of iterator) {
        let res;
        try {
            return fn(member);
        } catch (err) {
            // todo: log it
            continue;
        }
        // todo: log the success
        return res;
    }
}

function attemptWith(methods) {
    return async function () {
        for (const method of methods) {
            let res;
            try {
                res = method.apply(null, arguments);
            } catch (err) {
                // todo: log it
                continue;
            }
            return res;
        }
    };
}

/* Entry points */
function configureModel(modelString, goal) {
    if (goal === "soon") {
    } else if (goal === "efficiently") {
    } else if (goal === "reliable") {
    }
}

/* High-level routines */
function configureLocal(modelString) {
    const parts = modelString.split("/");
    let ogHfRepo; // original
    let quantization;
    if (parts.length >= 2) {
        quantization = parts[parts.length - 1];
        ogHfRepo = parts.slice(0, -1).join("/");
    } else {
        quantization = globalConfig.defaultQuant;
        ogHfRepo = modelString;
    }

    await attemptForEach(getAvailableQuantRepos(ogHfRepo), (hfRepo) => {
        const ggufPath = downloadGGUF(hfRepo, quantization);
        const mlxRes = attempt(loadOnMLX)(ggufPath);
        if (mlxRes === FailedAttempt) {
            const llamaCppRes = attempt(loadOnLlamaCpp)(ggufPath);
            if (llamaCppRes === FailedAttempt) {
                throw new Error();
            } else {
            }
        } else {
        }
    });
}

// returns an async generator
function getAvailableQuantRepos(model) {
    return hfHub.listModels({
        search: {
            tags: ["base_model:quantized:" + model],
        },
    });
}

/* Send a request to Modal serverless */
async function quantizeModel(hfRepo) {
}

/* Local inference */
async function downloadGGUF(hfRepo, quantization) {
}

async function loadOnLlamaCpp(ggufPath) {
}

async function loadOnMLX(ggufPath) {
}

/* Human integration */
async function askToven() {
}

async function askLithros() {
}

async function emailHuman() {
}
