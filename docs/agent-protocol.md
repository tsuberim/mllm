# Agent Protocol

The contract between Merlin and its sandbox — identical at trace generation time and inference time.

## Token format

Tool calls, results, and thinking use special tokens. The model emits tool calls and thinking; the harness injects results.

```
<|think|>
I need to find all TODO comments. grep -rn is the right call.
<|/think|>
<|tool_call|>bash grep -rn 'TODO' src/<|/tool_call|>
<|tool_result|>src/utils.py:14: # TODO: fix this<|/tool_result|>
<|think|>
Found one result. Task complete.
<|/think|>

<|spawn|>find all TODOs in src/ -- src/utils.py src/main.py<|/spawn|>
<|agent_id|>12345<|/agent_id|>

<|wait|>12345 12346<|/wait|>
<|wait_result|>
12345 completed
12346 completed
<|/wait_result|>
```

`spawn` format: `task -- context_file1 context_file2` (`--` separates task from context files).
`wait_result` format: one line per agent, `agent_id status`.

### Special tokens

```
<|task|>         <|/task|>
<|think|>        <|/think|>
<|tool_call|>    <|/tool_call|>
<|tool_result|>  <|/tool_result|>
<|spawn|>        <|/spawn|>
<|agent_id|>     <|/agent_id|>
<|wait|>         <|/wait|>
<|wait_result|>  <|/wait_result|>
<|done|>         <|/done|>
```

## Full trace example

```
<|task|>Find all TODO comments in src/ and count them.<|/task|>
<|think|>
I'll use grep to find TODO comments recursively.
<|/think|>
<|tool_call|>bash grep -rn 'TODO' src/<|/tool_call|>
<|tool_result|>src/utils.py:14: # TODO: fix this
src/main.py:42: # TODO: handle edge case
<|/tool_result|>
<|done|>Found 2 TODO comments: src/utils.py:14 and src/main.py:42.<|/done|>
```

Empty answer:
```
<|task|>Add a newline at the end of src/utils.py.<|/task|>
<|tool_call|>bash echo "" >> src/utils.py<|/tool_call|>
<|tool_result|><|/tool_result|>
<|done|>
```

## Tools

```
bash(cmd)                                      → {stdout, stderr, exit_code}
spawn(task [-- context_file ...])              → agent_id
wait(agent_id ...)                             → {agent_id: status, ...}
```

`bash` covers all filesystem and execution operations. The model uses standard shell commands — no custom read/write/grep/find tools. Pretraining distribution applies directly.

`spawn` is non-blocking. Returns immediately with an `agent_id`. Use `wait` to block until agents complete.

## Bash conventions

The model uses standard Unix commands:

```bash
cat file.py                       # read
cat file.py | head -100           # read with limit
echo "content" > file.py         # write
cat >> file.py                    # append
sed -i 's/old/new/g' file.py     # replace
grep -rn "pattern" src/           # search
rg "pattern" src/                 # fast search
find . -name "*.py"               # find files
fd -e py                          # fast find
python script.py                  # run script
curl url -o file.html             # fetch
```

## Harness behavior

- **Thin and deterministic** — no semantic processing, no summarization
- **Truncation** via familiar shell conventions: `# ... 234 more lines`
- **Output format** faithful to real bash — what grep returns is what the model sees
- Stateful within a trace — same container across all bash calls

## Spawn / wait pattern

```
<|spawn|>find all TODO comments in src/ -- src/<|/spawn|>
<|agent_id|>12345<|/agent_id|>
<|spawn|>count test coverage in tests/ -- tests/<|/spawn|>
<|agent_id|>12346<|/agent_id|>
<|wait|>12345 12346<|/wait|>
<|wait_result|>
12345 completed
12346 completed
<|/wait_result|>
<|tool_call|>bash cat results/12345.txt<|/tool_call|>
<|tool_result|>...</|tool_result|>
```

Workers write output to `results/<agent_id>.txt`. Orchestrator reads after `wait`.

---

# Sandbox

## Base image

`python:3.12-slim`

## CLI tools

```
git, curl, wget, bash
rg, fd, tree
diff, patch
zip, unzip, tar
make
sqlite3
jq, yq, xmllint, csvkit
```

## Python packages

```
numpy, pandas, scipy, matplotlib, scikit-learn
requests, httpx, beautifulsoup4
pytest, ruff, black
pyyaml, sqlalchemy
rich, tqdm
```

## Constraints

- Python 3.12 — fixed, no other runtimes
- No network access except via explicit `curl`/`wget` in traces
- Stateful within a trace, fresh container per trace
- Pre-pull image on each generation node before run starts
