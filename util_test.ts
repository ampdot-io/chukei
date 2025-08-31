import { assert, assertEquals } from "@std/assert";
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
