import nox


@nox.session(python=False)
def format_and_test(session):
	session.run("poetry", "install", "--with", "dev")
	session.run("poetry", "run", "ruff", "format", "quovadis_tad", "notebooks")
	session.run("poetry", "run", "ruff", "check", "--fix", "quovadis_tad", "notebooks")


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
		"quovadis_tad",
	)
