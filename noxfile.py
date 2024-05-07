import nox


@nox.session(python=False)
def format_and_test(session):
	session.run("poetry", "install", "--with", "dev")
	session.run("poetry", "run", "ruff", "format", "src", "notebooks")
	session.run("poetry", "run", "ruff", "check", "--fix", "src", "notebooks")


@nox.session(python=False)
def make_docs(session):
	session.run("poetry", "install", "--with", "dev")
	session.run(
		"poetry",
		"run",
		"pdoc",
		"-d",
		"restructuredtext",
		"-o",
		"doc",
		"--math",
		"--mermaid",
		"src",
	)
