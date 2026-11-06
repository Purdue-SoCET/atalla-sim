from torch.fx import GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer

'''
Render FX graphs with metadata annotations. 
'''


'''
gm: GraphModule
dot_graph: graphviz.Digraph object representing the FX graph
meta_keys: the keys in node.meta to extract and display
'''
def attach_metadata(gm: GraphModule, dot_graph, meta_keys: tuple[str, ...]) -> None:

    name_to_node = {node.name: node for node in gm.graph.nodes}

    for dot_node in dot_graph.get_nodes():
        name = dot_node.get_name().strip('"')
        if name in ("graph", "node", "edge"):
            continue

        node = name_to_node.get(name)
        if node is None:
            continue

        meta_values = []
        for key in meta_keys:
            value = node.meta.get(key)
            if value is not None:
                meta_values.append(f"{key}={value}")

        if not meta_values:
            continue

        base_label = dot_node.get_label()
        if base_label:
            base_label = base_label.strip('"')
            if base_label.startswith("{") and base_label.endswith("}"):
                suffix = "|".join(meta_values)
                merged_label = f"{base_label[:-1]}|{suffix}}}"
            else:
                suffix = "\\n".join(meta_values)
                merged_label = f"{base_label}\\n{suffix}"
        else:
            merged_label = "\\n".join(meta_values)

        dot_node.set_label(merged_label)


def render_graph(gm: GraphModule, path: str, meta_keys: tuple[str, ...]) -> None:
    drawer = FxGraphDrawer(gm, "graph")
    dot_graph = drawer.get_dot_graph()
    attach_metadata(gm, dot_graph, meta_keys=meta_keys)

    dot_graph.write_svg(path)
