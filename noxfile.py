import nox

@nox.session(python=False)
def format_and_test(session):
    session.run("poetry", "install", "--with", "dev")
    #session.run("poetry", "run", "ruff", "format", "src", "tests")
    #session.run("poetry", "run", "ruff", "check", "--fix", "src", "tests")
    session.run("poetry", "run", "pytest", "tests/")

@nox.session(python=False)
def make_docs(session):
    session.run("poetry", "install", "--with", "dev")
    session.run("poetry", "run", "pdoc",
                "-d", "restructuredtext",
                "-t", "pdoc-theme-gv",
                "-o", "doc",
                "--math",
                "--mermaid",
                "src/tad")