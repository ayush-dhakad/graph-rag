# Resume Project Section: Graph RAG for Codebases

**Codebase Graph RAG System** | *Python, Tree-sitter, NetworkX, HuggingFace*

* Built a Graph Retrieval-Augmented Generation (Graph RAG) system designed to give AI agents precise context of large Python repositories without overwhelming their context windows.
* Used Tree-sitter to parse entire codebases into an Abstract Syntax Tree (AST) graph, mapping functions and classes as interconnected nodes rather than flat text chunks.
* Implemented a neural retrieval pipeline to search the graph and extract only the top 5 most highly-relevant functions needed for a given prompt.
* Reduced LLM token consumption by over 90% (saving costs and reducing latency) by feeding the AI targeted AST nodes and file skeletons instead of injecting massive 1000+ line files.

***

### 🗣️ How to talk about it in an interview:
If they ask, *"Tell me about this Graph RAG project?"* you can give them this simple 30-second pitch:

> *"Standard RAG for code is really inefficient because it just chunks text blindly. If an AI agent needs to fix a bug, standard RAG might pass it an entire 1,500-line file, which wastes thousands of tokens and confuses the model.*
> 
> *I built a Graph RAG system that acts like a compiler. It parses the code into an AST graph. When the AI needs context, my system searches the graph and returns only the 3 or 4 specific functions it actually needs to read. It reduced token consumption by over 90% while actually improving the AI's ability to fix bugs because it removed all the noisy, irrelevant code."*
