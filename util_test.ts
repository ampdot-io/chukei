import {
    assert,
    assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { securePath } from "./util.ts";

Deno.test("securePath prevents directory traversal", () => {
    const base = "/tmp/base";
    const result = securePath(base, "../etc/passwd");
    assert(result?.startsWith(`${base}/`));
});

Deno.test("securePath joins paths within base", () => {
    const base = "/tmp/base";
    assertEquals(securePath(base, "file.txt"), `${base}/file.txt`);
});

import { downloadFile } from "./util.ts";

Deno.test("downloadFile simple fetch", async () => {
    const dest = await Deno.makeTempFile();
    await downloadFile("http://example.com/", dest);
    const text = await Deno.readTextFile(dest);
    assert(text.includes("Example Domain"));
    await Deno.remove(dest);
});
