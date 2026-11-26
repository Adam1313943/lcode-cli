"""Offline tests for lcode: the tool functions + the whole tool-calling loop against a fake Ollama server.
No real model required.

Run: python3 -I test_lcode.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lcode  # noqa: E402


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        with open(os.path.join(self.root, "a.py"), "w") as f:
            f.write("def add(a, b):\n    return a - b\n\nx = 1\nx = 1\n")
        os.makedirs(os.path.join(self.root, "pkg"))
        with open(os.path.join(self.root, "pkg", "m.py"), "w") as f:
            f.write("import os\n")
        self.t = lcode.Tools(self.root, approve_all=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_with_line_numbers(self):
        out = self.t.read_file("a.py")
        self.assertIn("    2\t    return a - b", out)

    def test_read_range(self):
        out = self.t.read_file("a.py", start_line=4, max_lines=1)
        self.assertIn("    4\tx = 1", out)
        self.assertIn("1 more line", out)

    def test_path_escape_blocked(self):
        for p in ("../x", "/etc/passwd", "pkg/../../x"):
            with self.assertRaises(lcode.ToolError):
                self.t.read_file(p)

    def test_symlink_escape_blocked(self):
        outside = tempfile.mkdtemp()
        os.symlink(outside, os.path.join(self.root, "link"))
        with self.assertRaises(lcode.ToolError):
            self.t.write_file("link/evil.txt", "x")
        self.assertFalse(os.path.exists(os.path.join(outside, "evil.txt")))

    def test_edit_unique(self):
        self.t.edit_file("a.py", "return a - b", "return a + b")
        self.assertIn("return a + b", open(os.path.join(self.root, "a.py")).read())

    def test_edit_ambiguous(self):
        with self.assertRaises(lcode.ToolError) as cm:
            self.t.edit_file("a.py", "x = 1", "x = 2")
        self.assertIn("appears 2 times", str(cm.exception))

    def test_edit_replace_all(self):
        self.t.edit_file("a.py", "x = 1", "x = 2", replace_all=True)
        self.assertEqual(open(os.path.join(self.root, "a.py")).read().count("x = 2"), 2)

    def test_edit_not_found(self):
        with self.assertRaises(lcode.ToolError):
            self.t.edit_file("a.py", "nope", "x")

    def test_write_new_in_subdir(self):
        self.t.write_file("new/dir/b.txt", "hi\n")
        self.assertEqual(open(os.path.join(self.root, "new/dir/b.txt")).read(), "hi\n")

    def test_list_and_grep(self):
        self.assertIn("pkg/m.py", self.t.list_files("**/*.py"))
        self.assertIn("pkg/m.py:1: import os", self.t.grep(r"^import", file_glob="*.py"))
        self.assertIn("a.py:2:", self.t.grep(r"a - b", path="a.py"))

    def test_shell(self):
        out = self.t.run_shell("echo hello && exit 3")
        self.assertIn("exit code 3", out)
        self.assertIn("hello", out)

    def test_sandbox_blocks_outside_write_and_network(self):
        t = lcode.Tools(self.root, approve_all=True, sandbox=True)
        probe = os.path.join(os.path.expanduser("~"), ".lcode_sandbox_test_probe")
        # write via python to test sandbox permissions (echo redirection is blocked by write detection first)
        self.assertIn("exit code 0", t.run_shell("python3 -c \"open('inside.txt','w').write('ok')\""))
        self.assertNotIn("exit code 0", t.run_shell(f"python3 -c \"open('{probe}','w').write('x')\""))
        self.assertFalse(os.path.exists(probe))
        self.assertNotIn("exit code 0", t.run_shell("curl -s -m 5 https://example.com -o /dev/null"))

    def test_denied_without_approval(self):
        t = lcode.Tools(self.root, approve_all=False)
        import builtins
        orig = builtins.input
        builtins.input = lambda *_: "n"
        try:
            with self.assertRaises(lcode.ToolError):
                t.run_shell("touch should_not_exist")
        finally:
            builtins.input = orig
        self.assertFalse(os.path.exists(os.path.join(self.root, "should_not_exist")))


# ---------- fake Ollama: return pre-scripted replies in order and record the requests received ----------
SCRIPT = []
REQUESTS = []


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(body)
        reply = SCRIPT.pop(0)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        chunks = []
        if reply.get("thinking"):
            chunks.append({"message": {"role": "assistant", "thinking": reply["thinking"]}, "done": False})
        if reply.get("content"):
            chunks.append({"message": {"role": "assistant", "content": reply["content"]}, "done": False})
        if reply.get("tool_calls"):
            chunks.append({"message": {"role": "assistant", "tool_calls": reply["tool_calls"]}, "done": False})
        chunks.append({"message": {"role": "assistant", "content": ""}, "done": True,
                       "done_reason": reply.get("done_reason", "stop"),
                       "prompt_eval_count": 1200, "eval_count": 50, "eval_duration": 2_000_000_000})
        for ch in chunks:
            self.wfile.write((json.dumps(ch) + "\n").encode())


def tc(name, **args):
    return {"function": {"name": name, "arguments": args}}


class LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def run_cli(self, root, prompt, script):
        SCRIPT[:] = script
        REQUESTS.clear()
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "lcode.py"), "-p", prompt, "--yolo",
                            "--host", self.url, "--dir", root, "--think", "low"],
                           capture_output=True, text=True, timeout=60)
        return r

    def test_fix_bug_flow(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "calc.py"), "w") as f:
                f.write("def add(a, b):\n    return a - b\n")
            script = [
                {"thinking": "read the file first", "tool_calls": [tc("read_file", path="calc.py")]},
                {"tool_calls": [tc("edit_file", path="calc.py", old_string="return a - b", new_string="return a + b")]},
                {"tool_calls": [tc("run_shell", command="python3 -c 'import calc; assert calc.add(2,3)==5; print(\"ok\")'")]},
                {"content": "Fixed add; tests pass."},
            ]
            r = self.run_cli(root, "fix add", script)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("return a + b", open(os.path.join(root, "calc.py")).read())
            self.assertIn("Fixed add", r.stdout)
            # every request carries a fixed context, think, and tool list
            self.assertTrue(all(q["options"]["num_ctx"] == 32768 and q["think"] == "low" and q["tools"] for q in REQUESTS))
            # the 4th request must see the shell result, and must not replay thinking
            last = REQUESTS[-1]["messages"]
            self.assertTrue(any(m["role"] == "tool" and "ok" in m["content"] for m in last))
            self.assertFalse(any("thinking" in m for m in last))

    def test_tool_error_reported_to_model(self):
        with tempfile.TemporaryDirectory() as root:
            script = [{"tool_calls": [tc("read_file", path="../../etc/passwd")]}, {"content": "cannot read."}]
            self.run_cli(root, "read a file", script)
            tool_msgs = [m for m in REQUESTS[-1]["messages"] if m["role"] == "tool"]
            self.assertIn("Access denied", tool_msgs[0]["content"])

    def test_repeat_loop_guard(self):
        with tempfile.TemporaryDirectory() as root:
            script = [{"tool_calls": [tc("list_files")]}] * 5
            r = self.run_cli(root, "list files", script)
            self.assertIn("3 times in a row", r.stdout)
            self.assertEqual(len(REQUESTS), 3)

    def test_length_warning(self):
        with tempfile.TemporaryDirectory() as root:
            r = self.run_cli(root, "hard problem", [{"thinking": "thinking hard", "done_reason": "length"}])
            self.assertIn("output limit during the thinking phase", r.stdout)

    def test_unknown_tool(self):
        with tempfile.TemporaryDirectory() as root:
            self.run_cli(root, "x", [{"tool_calls": [tc("delete_everything")]}, {"content": "ok"}])
            tool_msgs = [m for m in REQUESTS[-1]["messages"] if m["role"] == "tool"]
            self.assertIn("no such tool", tool_msgs[0]["content"])


class ShellWriteDetectionTests(unittest.TestCase):
    BLOCK = [
        "cat > solution.py << 'EOF'\ndef f(x):\n    return x > 1\nEOF",          # a heredoc file write seen in the wild
        "cd /tmp/x && cat > task_assembled.s << 'EOF'\n.globl _f\nEOF",
        "cat <<EOF > a.txt\nhi\nEOF",
        "cat << 'EOF' >> notes.md\nhi\nEOF",
        "tee out.py << 'EOF'\nprint(1)\nEOF",
        "echo 'print(1)' > a.py",
        "printf 'x\\n' >> log.txt",
        "cd src && echo hi > b.txt",
        "ls; cat a b > c",
        "echo x | tee -a file.txt",
        "sed -i '' 's/a/b/' calc.py",
        "perl -pi -e 's/a/b/' calc.py",
    ]
    ALLOW = [
        "python3 << 'PYEOF'\nwith open('task.s') as f:\n    print(len(f.read()) > 3)\nPYEOF",  # heredoc to run python
        "python3 -c \"print(1 > 0)\"",
        "python3 -c 'import dis; dis.dis(lambda: 1)' 2>&1",
        "cc -shared -o t.dylib t.s 2>&1",
        "make test > /dev/null 2>&1",
        "echo done",
        "echo error >&2",
        "cat task.s",
        "grep -n '>' calc.py",
        "python3 run.py | tee /dev/null",
        "sed -n '1,5p' calc.py",
        "nm task.dylib",
    ]

    def test_block(self):
        for cmd in self.BLOCK:
            self.assertTrue(lcode.shell_writes_files(cmd), cmd)

    def test_allow(self):
        for cmd in self.ALLOW:
            self.assertEqual(lcode.shell_writes_files(cmd), "", cmd)

    def test_blocked_command_not_executed(self):
        root = tempfile.mkdtemp()
        t = lcode.Tools(root, approve_all=True)
        with self.assertRaises(lcode.ToolError) as cm:
            t.run_shell("cat > x.py << 'EOF'\nprint(1)\nEOF")
        self.assertIn("write_file", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(root, "x.py")))

    def test_change_tracking(self):
        root = tempfile.mkdtemp()
        with open(os.path.join(root, "keep.txt"), "w") as f:
            f.write("a")
        with open(os.path.join(root, "gone.txt"), "w") as f:
            f.write("a")
        t = lcode.Tools(root, approve_all=True)
        out = t.run_shell("python3 -c \"open('made.py','w').write('x'); open('keep.txt','a').write('b')\" && rm gone.txt")
        self.assertIn("[file changes]", out)
        self.assertIn("added made.py", out)
        self.assertIn("modified keep.txt", out)
        self.assertIn("deleted gone.txt", out)

    def test_no_change_no_report(self):
        t = lcode.Tools(tempfile.mkdtemp(), approve_all=True)
        self.assertNotIn("[file changes]", t.run_shell("echo hi"))


FIXTURES = os.path.join(HERE, "fixtures")


class WebParseTests(unittest.TestCase):
    def test_real_ddg_page(self):
        page = open(os.path.join(FIXTURES, "ddg_argparse.html"), encoding="utf-8").read()
        rs = lcode.parse_ddg_html(page, 10)
        self.assertEqual(len(rs), 10)
        self.assertEqual(rs[0]["url"], "https://docs.python.org/3/library/argparse.html")
        self.assertIn("argparse", rs[0]["title"])
        self.assertIn("argparse", rs[0]["snippet"].lower())
        self.assertEqual(len(lcode.parse_ddg_html(page, 3)), 3)

    def test_ads_skipped_and_redirect_decoded(self):
        page = """
        <div class="result result--ad"><a class="result__a" href="https://ads.example/x">Ad</a>
          <a class="result__snippet" href="#">buy</a></div>
        <div class="result results_links"><a class="result__a"
          href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fdoc%3Fa%3D1&amp;rut=x">Real <b>Doc</b></a>
          <a class="result__snippet" href="#">Some <b>useful</b>   text</a></div>"""
        rs = lcode.parse_ddg_html(page)
        self.assertEqual(len(rs), 1)
        self.assertEqual(rs[0]["url"], "https://example.org/doc?a=1")
        self.assertEqual(rs[0]["title"], "Real Doc")
        self.assertEqual(rs[0]["snippet"], "Some useful text")

    def test_html_to_text(self):
        ex = lcode._TextExtractor()
        ex.feed("<html><head><title>T</title><style>.x{}</style></head><body><script>evil()</script>"
                "<h1>Head</h1><p>para one</p><ul><li>a</li><li>b</li></ul></body></html>")
        self.assertEqual(ex.title, "T")
        t = ex.text()
        self.assertNotIn("evil", t)
        self.assertNotIn(".x{}", t)
        self.assertIn("Head", t)
        self.assertIn("para one", t)

    def test_check_public_url(self):
        orig = lcode.socket.getaddrinfo
        table = {"public.test": "93.184.216.34", "lan.test": "192.168.1.1", "meta.test": "169.254.169.254",
                 "localhost": "127.0.0.1"}
        lcode.socket.getaddrinfo = lambda host, port, *a, **k: [(2, 1, 6, "", (table.get(host, host), port))]
        try:
            lcode.check_public_url("https://public.test/x")
            for bad in ("http://127.0.0.1:11434/api", "http://localhost/", "http://lan.test/", "http://meta.test/",
                        "file:///etc/passwd", "ftp://public.test/", "http://10.0.0.5/"):
                with self.assertRaises(lcode.ToolError, msg=bad):
                    lcode.check_public_url(bad)
        finally:
            lcode.socket.getaddrinfo = orig


class FakeWeb(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):  # fake DuckDuckGo
        q = self.rfile.read(int(self.headers["Content-Length"])).decode()
        FakeWeb.last_query, FakeWeb.last_ua = q, self.headers.get("User-Agent")
        body = open(os.path.join(FIXTURES, "ddg_argparse.html"), "rb").read()
        if "blockme" in q:
            body = b"<html>If this error persists, please let us know. anomaly detected</html>"
        self._send(200, "text/html; charset=utf-8", body)

    def do_GET(self):
        if self.path == "/page":
            self._send(200, "text/html", "<html><title>Page</title><body><p>" + "word " * 2000 +
                       "</p><script>ignore()</script></body></html>")
        elif self.path == "/plain":
            self._send(200, "text/plain", "hello plain")
        elif self.path == "/bin":
            self._send(200, "application/octet-stream", b"\x00\x01")
        else:
            self._send(404, "text/plain", "nope")

    def _send(self, code, ctype, body):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class WebToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeWeb)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        cls.orig_ddg, cls.orig_interval = lcode.DDG_URL, lcode.MIN_SEARCH_INTERVAL
        lcode.DDG_URL, lcode.MIN_SEARCH_INTERVAL = cls.base + "/html/", 0

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        lcode.DDG_URL, lcode.MIN_SEARCH_INTERVAL = cls.orig_ddg, cls.orig_interval

    def web(self, allow=True, approve=True):
        return lcode.WebTools(lambda q: approve, allow_private=allow)

    def test_search(self):
        out = self.web().web_search("python argparse", max_results=3)
        self.assertTrue(out.startswith(lcode.WEB_NOTICE))
        self.assertIn("1. argparse", out)
        self.assertIn("https://docs.python.org/3/library/argparse.html", out)
        self.assertNotIn("4. ", out)
        self.assertIn("q=python+argparse", FakeWeb.last_query)
        self.assertTrue(FakeWeb.last_ua.startswith("lcode/"))

    def test_search_blocked_page(self):
        with self.assertRaises(lcode.ToolError) as cm:
            self.web().web_search("blockme")
        self.assertIn("blocked", str(cm.exception))

    def test_search_denied(self):
        with self.assertRaises(lcode.ToolError):
            self.web(approve=False).web_search("x")

    def test_fetch_html_paging(self):
        out = self.web().web_fetch(self.base + "/page", max_chars=1000)
        self.assertIn("Page", out)
        self.assertNotIn("ignore()", out)
        self.assertIn("start_char=1000 to continue reading", out)
        out2 = self.web().web_fetch(self.base + "/page", start_char=8000, max_chars=12000)
        self.assertNotIn("to continue reading", out2)

    def test_fetch_plain_and_errors(self):
        self.assertIn("hello plain", self.web().web_fetch(self.base + "/plain"))
        with self.assertRaises(lcode.ToolError):
            self.web().web_fetch(self.base + "/bin")
        with self.assertRaises(lcode.ToolError):
            self.web().web_fetch(self.base + "/missing")

    def test_fetch_private_blocked_by_default(self):
        with self.assertRaises(lcode.ToolError) as cm:
            self.web(allow=False).web_fetch(self.base + "/plain")
        self.assertIn("local or private", str(cm.exception))

    def test_web_tools_only_with_flag(self):
        names = lambda web: {t["function"]["name"] for t in lcode.tool_schemas(web)}
        self.assertNotIn("web_search", names(False))
        self.assertTrue({"web_search", "web_fetch"} <= names(True))
        self.assertNotIn("Web tools", lcode.system_prompt(tempfile.mkdtemp()))
        self.assertIn("untrusted", lcode.system_prompt(tempfile.mkdtemp(), web=True))


class PromptTests(unittest.TestCase):
    def test_general_rules_present(self):
        p = lcode.system_prompt(tempfile.mkdtemp())
        self.assertIn("in your head", p)
        self.assertIn("wrap up", p)
        self.assertNotIn("Project notes", p)

    def test_project_notes_loaded(self):
        root = tempfile.mkdtemp()
        with open(os.path.join(root, "LCODE.md"), "w") as f:
            f.write("RE task: arm64 can be assembled and compared via ctypes.")
        p = lcode.system_prompt(root)
        self.assertIn("# Project notes (LCODE.md)", p)
        self.assertIn("ctypes", p)

    def test_project_notes_truncated(self):
        root = tempfile.mkdtemp()
        with open(os.path.join(root, "LCODE.md"), "w") as f:
            f.write("x" * 20000)
        p = lcode.system_prompt(root)
        self.assertIn("truncated", p)
        self.assertLess(len(p), 20000)

    def test_notes_sent_to_model(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "LCODE.md"), "w") as f:
                f.write("project rule ABC123")
            LoopTests.setUpClass()
            try:
                LoopTests.run_cli(LoopTests, root, "hi", [{"content": "ok"}])
            finally:
                LoopTests.tearDownClass()
            self.assertIn("project rule ABC123", REQUESTS[0]["messages"][0]["content"])


class CompactTests(unittest.TestCase):
    def test_compact_old_tool_outputs(self):
        args = type("A", (), {"dir": tempfile.mkdtemp(), "yolo": True, "think": "low", "ctx": 1000,
                              "model": "m", "max_out": 100, "show_thinking": False, "host": "", "sandbox": False})()
        s = lcode.Session(args)
        for i in range(4):
            s.messages += [{"role": "user", "content": f"q{i}"},
                           {"role": "assistant", "content": "", "tool_calls": [tc("read_file", path="x")]},
                           {"role": "tool", "tool_name": "read_file", "content": "X" * 3000}]
        s.last_prompt_tokens = 900
        s.compact()
        tools = [m["content"] for m in s.messages if m["role"] == "tool"]
        self.assertTrue(tools[0].startswith("[earlier tool output omitted")
                        and tools[1].startswith("[earlier tool output omitted"))
        self.assertEqual(tools[-1], "X" * 3000)


if __name__ == "__main__":
    unittest.main(verbosity=1)
