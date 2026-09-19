# Resume Project Section: Graph RAG for Codebases

**Codebase Graph RAG System** | *Python, Tree-sitter, NetworkX, HuggingFace*

* Built a Graph Retrieval-Augmented Generation (Graph RAG) system designed to give AI agents precise context of large Python repositories without overwhelming their context windows.
* Used Tree-sitter to parse entire codebases into an Abstract Syntax Tree (AST) graph, mapping functions and classes as interconnected nodes rather than flat text chunks.
* Implemented a neural retrieval pipeline to search the graph and extract only the top 5 most highly-relevant functions needed for a given prompt.
* Reduced LLM token consumption by over 90% (saving costs and reducing latency) by feeding the AI targeted AST nodes and file skeletons instead of injecting massive 1000+ line files. ok

***
