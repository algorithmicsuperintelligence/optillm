#!/usr/bin/env python3
"""
Tests for the static AST gate on z3_solver (issue #303).

The z3 approach exec()s LLM-generated code; the system prompt promises the
code only uses Z3/SymPy/math. These tests verify the AST gate enforces that
contract: dangerous imports/calls/dunder-attribute traversals are rejected
before exec(), and rejected code degrades to standard inference instead of
entering the fix-up loop.

All tests are offline except the allow-path cases that run the real solver in
the local venv (z3 is a core dependency); no network or LLM is contacted.
"""

import ast
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.z3_solver import (
    Z3SymPySolverSystem,
    _validate_solver_ast,
    execute_code_in_process,
)


def _guard(code: str):
    """Run the gate the way extract_and_validate_expressions() does."""
    _validate_solver_ast(ast.parse(code))


def _response(content, completion_tokens=10):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage.completion_tokens = completion_tokens
    return resp


# ---------------------------------------------------------------------------
# Refusal cases: code that leaves the declared solver subset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,violation", [
    ("import os", "import of 'os'"),
    ("import subprocess\nsubprocess.run(['id'])", "import of 'subprocess'"),
    ("from subprocess import run", "import from 'subprocess'"),
    ("import pathlib", "import of 'pathlib'"),
    ("__import__('os').system('id')", "__import__"),
    ("open('/etc/passwd')", "open()"),
    ("eval('1+1')", "eval()"),
    ("exec('x = 1')", "exec()"),
    ("compile('1', '<s>', 'eval')", "compile()"),
    ("name = globals()", "globals()"),
    ("().__class__.__bases__[0].__subclasses__()", "__class__"),
    ("z3.__loader__", "__loader__"),
    ("sympy.__spec__", "__spec__"),
    ("f = z3.Bool.__init__", "__init__"),
    ("import z3\nz3.__builtins__", "__builtins__"),
    # String-based evasions of the attribute denylist (QA finding): getattr
    # reaches the same objects via a constant argument, and __dict__ /
    # __getattribute__ turn every dunder into a string lookup. A module's
    # __dict__ also exposes its real __builtins__.
    ("getattr(z3, '__loader__')", "getattr()"),
    ("setattr(z3, '__loader__', None)", "setattr()"),
    ("delattr(z3, 'Int')", "delattr()"),
    ("z3.__dict__['__builtins__']['__import__']('os')", "__dict__"),
    ("z3.__getattribute__('__spec__')", "__getattribute__"),
    ("sympy.__getattr__('_load')", "__getattr__"),
])
def test_guard_rejects(code, violation):
    with pytest.raises(ValueError) as excinfo:
        _guard(code)
    assert "static safety check" in str(excinfo.value)
    assert violation in str(excinfo.value)


# ---------------------------------------------------------------------------
# Allow cases: legitimate solver code passes the gate and really executes
# ---------------------------------------------------------------------------

SOLVER_OK_CASES = [
    "x = Int('x')\ns = Solver()\ns.add(x > 2)\nprint(s.check())",
    "from sympy import Symbol, solve\nx = Symbol('x')\nprint(solve(x**2 - 4, x))",
    "from fractions import Fraction\nprint(Fraction(1, 3))",
    "import math\nprint(math.sqrt(16))",
    "from itertools import permutations\nprint(list(permutations([1, 2])))",
]


@pytest.mark.parametrize("code", SOLVER_OK_CASES)
def test_guard_allows_solver_code(code):
    _guard(code)  # must not raise


@pytest.mark.parametrize("code", SOLVER_OK_CASES)
def test_allowed_solver_code_executes(code):
    status, output = execute_code_in_process(code)
    assert status == "success", output
    assert output.strip(), "expected some printed output"


# ---------------------------------------------------------------------------
# Wiring: extract_and_validate_expressions enforces the gate
# ---------------------------------------------------------------------------

ANALYSIS_TEMPLATE = """SOLVER_CAN_BE_APPLIED: True

SOLVER_FORMULATION:
```python
{code}
```

Analysis: some analysis
"""


def test_extract_raises_on_malicious_formulation():
    system = Z3SymPySolverSystem("system prompt", client=MagicMock(), model="m")
    analysis = ANALYSIS_TEMPLATE.format(code="import os\nos.system('id')")
    with pytest.raises(ValueError) as excinfo:
        system.extract_and_validate_expressions(analysis)
    assert "import of 'os'" in str(excinfo.value)


def test_extract_returns_clean_formulation():
    system = Z3SymPySolverSystem("system prompt", client=MagicMock(), model="m")
    analysis = ANALYSIS_TEMPLATE.format(code="x = Int('x')\nprint(x)")
    assert system.extract_and_validate_expressions(analysis) == "x = Int('x')\nprint(x)"


# ---------------------------------------------------------------------------
# Behavior: rejection degrades to standard inference, bypassing the fix loop
# ---------------------------------------------------------------------------

def test_process_query_degrades_on_malicious_code():
    """Malicious formulation -> straight to standard inference, exec() never runs."""
    malicious = "import os\nos.system('id')"
    analyze_resp = _response(ANALYSIS_TEMPLATE.format(code=malicious))
    standard_resp = _response("Direct answer from the model.")

    client = MagicMock()
    client.chat.completions.create.side_effect = [analyze_resp, standard_resp]

    with patch("optillm.z3_solver.execute_code_in_process") as exec_mock:
        system = Z3SymPySolverSystem("system prompt", client=client, model="m")
        result, tokens = system.process_query("solve x > 2")

    assert result == "Direct answer from the model."
    exec_mock.assert_not_called()
    # Exactly two provider calls: analyze + standard inference. No fix-up calls.
    assert client.chat.completions.create.call_count == 2
    for call in client.chat.completions.create.call_args_list[1:]:
        sent = str(call)
        assert "Fix the Z3 or SymPy code" not in sent, \
            "rejected code must not enter the error-fix loop"


def test_process_query_solves_legitimate_formulation():
    """Legitimate formulation still takes the full solver path."""
    code = "x = Int('x')\ns = Solver()\ns.add(x > 2)\nprint(s.check())"
    analyze_resp = _response(ANALYSIS_TEMPLATE.format(code=code))
    answer_resp = _response("The constraint is satisfiable: sat.")

    client = MagicMock()
    client.chat.completions.create.side_effect = [analyze_resp, answer_resp]

    system = Z3SymPySolverSystem("system prompt", client=client, model="m")
    result, tokens = system.process_query("solve x > 2")

    assert "sat" in result
    assert client.chat.completions.create.call_count == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
