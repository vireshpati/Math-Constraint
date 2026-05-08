# MathConstraint

MathConstraint is a solver-verified benchmark for evaluating language models
on combinatorial constraint-satisfaction problems. Each instance is generated
from a formal PyCSP3 or SAT-modulo-symmetries backend, and submitted answers
are checked by the same solver stack used during generation.

## Datasets

Our datasets are available on Hugging Face:

- [MathConstraint](https://huggingface.co/datasets/MathConstraint/MathConstraint)
- [MathConstraint-Easy](https://huggingface.co/datasets/MathConstraint/MathConstraint-Easy)

This repository also includes frozen local copies:

- `mathconstraint/`: harder benchmark split
- `mathconstraint_easy/`: smaller/easier split with hinted and unhinted instances

Each dataset contains `instances/*.json`, `problems.jsonl`, and `manifest.json`.

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:vireshpati/Math-Constraint.git
cd Math-Constraint
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Create an environment with Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -e "third_party/sat-modulo-symmetries"
```

PySMS graph problems also require the `smsg` binary:

```bash
cd third_party/sat-modulo-symmetries
./build-and-install.sh --local
export PATH="$HOME/.local/bin:$PATH"
cd ../..
```

The `smsg` build requires CMake, a C++ compiler, and Boost.

## Evaluate A Model

Set an API key for your provider:

```bash
export OPENROUTER_API_KEY=...
```

Run a no-tool evaluation:

```bash
python -m math_constraint eval \
  --provider openrouter \
  --model openai/gpt-oss-120b:free \
  --dataset mathconstraint_easy \
  --output results/gpt-oss-120b/easy/no_tools.json \
  --workers 4 \
  --max-tokens 4096 \
  --temperature 0
```

Run with tool use:

```bash
python -m math_constraint eval \
  --provider openrouter \
  --model openai/gpt-oss-120b:free \
  --dataset mathconstraint \
  --output results/gpt-oss-120b/hard/tools.json \
  --enable-tools \
  --max-tool-rounds 8 \
  --workers 4 \
  --max-tokens 4096 \
  --temperature 0
```

Evaluation writes an aggregate JSON file plus per-instance traces under the
output directory.

## Python API

```python
from math_constraint import load_dataset
from math_constraint.eval import EvalConfig, Evaluator, compute_metrics
from math_constraint.llm import create_client

problems = load_dataset("mathconstraint_easy")[:10]

config = EvalConfig.from_provider(
    provider="openrouter",
    model="openai/gpt-oss-120b:free",
    temperature=0.0,
    max_tokens=4096,
    enable_tools=False,
)

client = create_client(config.provider, config.model, base_url=config.base_url)
evaluator = Evaluator(config, client)

results = [evaluator.evaluate(problem) for problem in problems]
metrics = compute_metrics(results)

print(metrics.accuracy)
```

## Answer Format

Models should answer with JSON:

```json
{
  "satisfiable": true,
  "solution": [],
  "reasoning": "brief explanation"
}
```

For UNSAT instances, use `null` for the solution:

```json
{
  "satisfiable": false,
  "solution": null,
  "reasoning": "brief explanation"
}
```

When `--enable-tools` is set, the evaluator exposes two tools:

- `execute_python(code, timeout_seconds)`: run short helper Python in an isolated subprocess.
- `submit_answer(satisfiable, solution, reasoning)`: submit the final structured answer.

## Tests

Lightweight tests:

```bash
MCNST_SKIP_SOLVER_TESTS=1 python -m pytest -q
```

Full tests require PyCSP3, PySMS, and `smsg` on `PATH`:

```bash
python -m pytest -q
```

## License

This repository and the MathConstraint datasets are released under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

MathConstraint builds on [PyCSP3](https://github.com/xcsp3team/pycsp3),
[pycsp3-models](https://github.com/xcsp3team/pycsp3-models), and
[sat-modulo-symmetries](https://github.com/markirch/sat-modulo-symmetries).
