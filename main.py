import os
from parser import PythonParser
from graph import CodeGraph

def main():
    print("Initializing Graph RAG Codebase engine...")
    parser = PythonParser()
    graph = CodeGraph()

    # Define the directory to parse (e.g. current directory)
    workspace_dir = "."
    extractions = []
    
    print(f"Parsing python files in {workspace_dir}...")
    for root, _, files in os.walk(workspace_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'rb') as f:
                    source = f.read()
                rel_path = os.path.relpath(filepath, workspace_dir)
                try:
                    nodes, edges = parser.extract_nodes_and_edges(source, rel_path)
                    extractions.append((nodes, edges))
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")

    print("Building AST Graph and dense embeddings...")
    graph.build_from_extractions(extractions)

    print("\n--- Graph Statistics ---")
    print(f"Total Nodes (Functions/Classes): {graph.graph.number_of_nodes()}")
    print(f"Total Edges (Calls/Relationships): {graph.graph.number_of_edges()}")

    print("\n--- Querying Graph ---")
    query = "extract nodes and edges"
    print(f"Query: '{query}'")
    
    results = graph.search_codebase(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n[Result {i}] (Score: {result.get('rerank_score')})")
        print(f"Type: {result.get('type')} | Name: {result.get('name')}")
        print(f"Snippet: \n{result.get('snippet')}")

if __name__ == "__main__":
    main()
