"""Griffe extension: render ROME's reST-flavoured docstrings as Markdown.

ROME's docstrings are written in the Sphinx dialect — ``:meth:`start```,
``:class:`~rome.manager.Manager```, ``.. warning::`` — because that is what the
source has always used and what an editor's tooling understands. MkDocs renders
Markdown, where those constructs are meaningless and leak into the page as
literal text (``:meth: DataManager.add``).

Rather than rewrite sixty-odd docstrings into a second dialect, this rewrites
them on the way into the documentation build. The source stays as it is; the
rendered reference reads correctly.

**Roles become inline code, not links.** ``[x][target]`` would be nicer, but
mkdocs-autorefs fails the ``--strict`` build on any target it cannot resolve, and
several roles in the source name things that are not documented API (a private
helper, a method whose class was since renamed) or are not Python at all
(``docs/dragon.md``). Inline code is always correct. Explicit Markdown links in
the docstring, if anyone adds them, pass through untouched.

Wired up in ``mkdocs.yml`` under the mkdocstrings python handler's
``extensions:``.
"""

from __future__ import annotations

import re

import griffe

#: Sphinx roles that name a Python object. ``:ref:`` and ``:doc:`` are here too
#: because the source uses them for prose cross-references.
_ROLE = re.compile(
    r":(?:py:)?(?:meth|class|func|mod|attr|data|obj|exc|ref|doc|term):"
    r"`(?P<body>[^`]+)`"
)

#: reST directives that map onto a Material admonition of the same name.
_ADMONITION = re.compile(
    r"^(?P<indent>[ \t]*)\.\.[ \t]+(?P<kind>note|warning|danger|tip|important|"
    r"caution|attention|hint|seealso|admonition|deprecated|versionadded|"
    r"versionchanged)::[ \t]*(?P<title>.*)$"
)

#: reST admonition name -> the Material admonition type that reads the same.
_KIND_MAP = {
    "seealso": "info",
    "attention": "warning",
    "caution": "warning",
    "important": "info",
    "hint": "tip",
    "admonition": "note",
    "deprecated": "warning",
    "versionadded": "info",
    "versionchanged": "info",
}


def _role_to_code(match: re.Match) -> str:
    """``:class:`~rome.manager.Manager``` -> `` `Manager` ``."""
    body = match.group("body").strip()

    # ``:role:`text <target>``` — the display text is what the author wanted.
    link = re.match(r"^(?P<text>.+?)\s*<(?P<target>[^>]+)>$", body)
    if link:
        body = link.group("text").strip()

    # A leading ``~`` means "show only the last dotted component".
    if body.startswith("~"):
        body = body.lstrip("~").rsplit(".", 1)[-1]

    return f"`{body}`"


def _convert_admonitions(text: str) -> str:
    """Turn ``.. warning::`` blocks into ``!!! warning`` ones.

    reST indents a directive's body relative to the marker; Material wants it
    indented exactly four spaces past the ``!!!``. So the body is dedented to its
    own common margin and re-indented, which also normalises the blank line the
    two dialects disagree about.
    """
    lines = text.split("\n")
    out: list[str] = []
    index = 0

    while index < len(lines):
        match = _ADMONITION.match(lines[index])
        if match is None:
            out.append(lines[index])
            index += 1
            continue

        indent = match.group("indent")
        kind = match.group("kind")
        title = match.group("title").strip()

        # Collect the body: every following line that is blank or indented
        # further than the marker.
        index += 1
        body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() and not line.startswith(indent + " "):
                break
            body.append(line)
            index += 1

        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()

        margin = min(
            (len(ln) - len(ln.lstrip()) for ln in body if ln.strip()),
            default=0,
        )

        header = f"{indent}!!! {_KIND_MAP.get(kind, kind)}"
        if title:
            header += f' "{title}"'
        out.append(header)
        out.append("")
        out.extend(f"{indent}    {ln[margin:]}" if ln.strip() else "" for ln in body)
        out.append("")

    return "\n".join(out)


def _convert(text: str) -> str:
    return _convert_admonitions(_ROLE.sub(_role_to_code, text))


class SphinxRolesToMarkdown(griffe.Extension):
    """Rewrite reST roles and directives in every docstring griffe loads."""

    def on_instance(self, *, obj: griffe.Object, **kwargs) -> None:  # noqa: D102
        docstring = getattr(obj, "docstring", None)
        if docstring is None or not docstring.value:
            return
        converted = _convert(docstring.value)
        if converted != docstring.value:
            docstring.value = converted
