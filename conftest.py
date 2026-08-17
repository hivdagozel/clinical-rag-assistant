import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TEST_VECTOR_DIR = Path(tempfile.mkdtemp(prefix="medical-rag-pytest-"))
os.environ["TEST_MODE"] = "true"
os.environ["TEST_VECTORSTORE_DIR"] = str(_TEST_VECTOR_DIR)
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["USE_MEDICINE_API"] = "false"
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"


@pytest.fixture(scope="session", autouse=True)
def isolated_vectorstore():
    yield _TEST_VECTOR_DIR
    shutil.rmtree(_TEST_VECTOR_DIR, ignore_errors=True)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.write_sep("=", "MEDICAL RAG TEST SUMMARY")
    terminalreporter.write_line(f"Exit status: {exitstatus}")
    terminalreporter.write_line(f"Isolated vectorstore: {_TEST_VECTOR_DIR}")
