import logging
import onnx

def load_onnx_model(file_path:str):
    try:
        model = onnx.load(file_path)
        graph = model.graph

        nodes = []
        edges = []
        node_map = {}

        for i, node in enumerate(graph.node):
            label = node.name if node.name else node.op_type
            nodes.append({
                "id": i,
                "label": label,
                "title": str(node),
                "group": "operation",
            })
            if node.output:
                node_map[node.output[0]] = i

        for i, node in enumerate(graph.node):
            for input_name in node.input:
                if input_name in node_map:
                    edges.append({"from": node_map[input_name], "to": i})

        node_summary = []
        for node in graph.node:
            node_summary.append({
                "name": node.name if node.name else "Unnamed",
                "op_type": node.op_type
            })

        summary = {
            "ir_version": model.ir_version,
            "producer": model.producer_name,
            "inputs": [{"name": inp.name} for inp in graph.input],
            "outputs": [{"name": out.name} for out in graph.output],
            "nodes": node_summary,
            "node_count": len(graph.node)
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": summary,
        }
    
    except Exception as e:
        logging.exception("Failed to load ONNX model")
        return {"error": str(e)}