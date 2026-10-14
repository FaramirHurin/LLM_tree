"""Python source <-> line-level syntax tree.

Each statement is one node. Simple statements keep their full text; compound
statements (def, if, for, while, with, class, try) keep only their header line.
else / except / finally become children of the block they belong to.
A virtual root (empty content) holds the module's top-level statements.
"""
import ast
import copy
import io
import tokenize


class Node:
    def __init__(self, content, children=None):
        self.content = content            # list of str tokens; "\n" separates decorator lines
        self.children = children or []


# ---- lexing -----------------------------------------------------------------

_FSTART = getattr(tokenize, "FSTRING_START", None)
_FEND = getattr(tokenize, "FSTRING_END", None)


def lex(text):
    """Split one header or statement into Python tokens. f-strings stay one token."""
    lines = text.split("\n")
    offsets = [0]
    for l in lines:
        offsets.append(offsets[-1] + len(l) + 1)
    at = lambda pos: offsets[pos[0] - 1] + pos[1]

    toks, depth, start = [], 0, None
    for tok in tokenize.generate_tokens(io.StringIO(text + "\n").readline):
        if _FSTART is not None and tok.type == _FSTART:
            if depth == 0:
                start = tok.start
            depth += 1
            continue
        if depth:
            if tok.type == _FEND:
                depth -= 1
                if depth == 0:
                    toks.append(text[at(start):at(tok.end)])
            continue
        if tok.type in (tokenize.NEWLINE, tokenize.NL):
            toks.append("\n")
        elif tok.type in (tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
            continue
        else:
            toks.append(tok.string)
    while toks and toks[-1] == "\n":
        toks.pop()
    return toks


# ---- code -> tree -----------------------------------------------------------

def _header(stmt):
    """Header line(s) of a compound statement: unparse it with an empty body."""
    stub = copy.copy(stmt)
    for field in ("orelse", "handlers", "finalbody"):
        if hasattr(stub, field):
            setattr(stub, field, [])
    stub.body = [ast.Pass()]
    text = ast.unparse(stub)
    return "\n".join(l for l in text.split("\n") if l and not l[0].isspace())


def _block(stmts):
    return [stmt_to_node(s) for s in stmts]


def stmt_to_node(s):
    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.AsyncWith)):
        return Node(lex(_header(s)), _block(s.body))
    if isinstance(s, (ast.If, ast.For, ast.AsyncFor, ast.While)):
        n = Node(lex(_header(s)), _block(s.body))
        if s.orelse:
            n.children.append(Node(["else", ":"], _block(s.orelse)))
        return n
    if isinstance(s, ast.Try):
        n = Node(["try", ":"], _block(s.body))
        for h in s.handlers:
            n.children.append(Node(lex(_header(h)), _block(h.body)))
        if s.orelse:
            n.children.append(Node(["else", ":"], _block(s.orelse)))
        if s.finalbody:
            n.children.append(Node(["finally", ":"], _block(s.finalbody)))
        return n
    if isinstance(s, ast.Match):
        header = lex(f"match {ast.unparse(s.subject)}:")
        cases = []
        for c in s.cases:
            guard = f" if {ast.unparse(c.guard)}" if c.guard is not None else ""
            cases.append(Node(lex(f"case {ast.unparse(c.pattern)}{guard}:"), _block(c.body)))
        return Node(header, cases)
    if hasattr(s, "body") or hasattr(s, "cases"):   # try* and anything newer: not handled
        raise ValueError(f"unsupported statement: {type(s).__name__}")
    return Node(lex(ast.unparse(s)))


def code_to_tree(code):
    return Node([], _block(ast.parse(code).body))


# ---- tree -> code -----------------------------------------------------------

_TRAILERS = {"except": 0, "else": 1, "finally": 2}   # printed after the body, at the father's indent


def _split_lines(tokens):
    out, cur = [], []
    for t in tokens:
        if t == "\n":
            out.append(cur); cur = []
        else:
            cur.append(t)
    out.append(cur)
    return out


def tree_to_code(root):
    lines = []

    def emit(node, indent):
        for part in _split_lines(node.content):
            lines.append("    " * indent + " ".join(part))
        body = [c for c in node.children if not (c.content and c.content[0] in _TRAILERS)]
        trailers = sorted((c for c in node.children if c.content and c.content[0] in _TRAILERS),
                          key=lambda c: _TRAILERS[c.content[0]])
        for c in body:
            emit(c, indent + 1)
        if not body and node.content and node.content[-1] == ":":
            lines.append("    " * (indent + 1) + "pass")
        for c in trailers:
            emit(c, indent)

    for c in root.children:
        emit(c, 0)
    return "\n".join(lines)


# ---- JSON-friendly form -----------------------------------------------------

def tree_to_nodes(root):
    """Node tree -> flat list [{"id", "father", "content"}], parents before children.

    The root is id 0 with father None. These stored IDs are only for saving: training
    draws fresh random IDs for every example, so the model never sees them.
    """
    nodes = []

    def visit(n, father):
        i = len(nodes)
        nodes.append({"id": i, "father": father, "content": n.content})
        for c in n.children:
            visit(c, i)

    visit(root, None)
    return nodes


def nodes_to_tree(nodes):
    """Inverse of tree_to_nodes. Accepts nodes in any order and any unique ids."""
    built = {n["id"]: Node(list(n["content"])) for n in nodes}
    root = None
    for n in nodes:
        if n["father"] is None:
            root = built[n["id"]]
        else:
            built[n["father"]].children.append(built[n["id"]])
    if root is None:
        raise ValueError("tree has no root (a node with father None)")
    return root


# ---- helpers ----------------------------------------------------------------

def canonical(node):
    """Order-free form of a tree, for comparing trees while ignoring sibling order."""
    return (tuple(node.content), tuple(sorted(canonical(c) for c in node.children)))


def print_tree(node, indent=0):
    if indent == 0:
        print("<root>")
    for c in node.children:
        print("  " * (indent + 1) + " ".join(t if t != "\n" else "\\n" for t in c.content))
        print_tree(c, indent + 1)
