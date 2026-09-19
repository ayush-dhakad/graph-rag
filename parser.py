"""Tree-sitter based Python parser that extracts AST nodes and call edges."""

import tree_sitter
import tree_sitter_python


class PythonParser:
    """Parse Python source files using tree-sitter and extract symbols + edges.

    Extracted node types:
        - ``function_definition``
        - ``class_definition``

    Extracted edge types:
        - ``CONTAINS`` — a class/function contains a nested definition
        - ``CALLS``    — a function calls another symbol
    """

    def __init__(self) -> None:
        self.language = tree_sitter.Language(tree_sitter_python.language())
        self.parser = tree_sitter.Parser(self.language)

    def parse(self, source_code: bytes):
        """Return the tree-sitter parse tree for *source_code*."""
        return self.parser.parse(source_code)

    def extract_nodes_and_edges(
        self, source_code: bytes, filepath: str
    ) -> tuple[list[dict], list[dict]]:
        """Extract symbol nodes and relationship edges from a Python file.

        Parameters
        ----------
        source_code : bytes
            Raw file contents.
        filepath : str
            Relative path used as the namespace prefix for node IDs.

        Returns
        -------
        (nodes, edges)
            *nodes* — list of dicts with keys: id, type, name, filepath,
            start_line, end_line, source.
            *edges* — list of dicts with keys: source, target, type.
        """
        tree = self.parse(source_code)
        nodes: list[dict] = []
        edges: list[dict] = []

        def _walk(cursor, context: str | None = None) -> None:
            node = cursor.node
            new_context = context

            if node.type in ("function_definition", "class_definition"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode("utf8")
                    node_id = (
                        f"{context}.{name}" if context else f"{filepath}::{name}"
                    )

                    nodes.append({
                        "id": node_id,
                        "type": node.type,
                        "name": name,
                        "filepath": filepath,
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                        "source": source_code[node.start_byte:node.end_byte].decode("utf8"),
                    })

                    if context:
                        edges.append({
                            "source": context,
                            "target": node_id,
                            "type": "CONTAINS",
                        })

                    new_context = node_id

            elif node.type == "call":
                func_node = node.child_by_field_name("function")
                if func_node and context:
                    # Raw name stored here; resolution happens in CodeGraph.
                    edges.append({
                        "source": context,
                        "target": func_node.text.decode("utf8"),
                        "type": "CALLS",
                    })

            if cursor.goto_first_child():
                _walk(cursor, new_context)
                while cursor.goto_next_sibling():
                    _walk(cursor, new_context)
                cursor.goto_parent()

        _walk(tree.walk())
        return nodes, edges
