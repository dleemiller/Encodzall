from graphviz import Digraph


def create_optimized_model_diagram(output_path="img/model_architecture"):
    dot = Digraph(comment="Encodzall Model Structure", format="png")
    dot.attr(rankdir="TB", splines="curved", nodesep="0.3", ranksep="0.5")

    # General Style
    dot.attr(
        "node",
        shape="box",
        style="rounded, filled",
        color="#66c2a5",
        fontname="Arial",
        fontsize="12",
        fontcolor="black",
        fillcolor="lightyellow",
        width="2.5",
        height="0.8",
    )
    dot.attr("edge", color="#5e81ac", arrowsize="0.8", penwidth="1.2")

    # Input and Embedding
    dot.node(
        "Input",
        "Input Tokens",
        shape="ellipse",
        style="filled",
        fillcolor="lightblue",
        fontsize="14",
        width="2.0",
        height="0.6",
    )
    dot.node("Embedding", "Embedding Layer")
    dot.edge("Input", "Embedding")

    # Encoder 1
    dot.node(
        "Encoder1",
        "Transformer Encoder 1\n(RoPE)",
        fillcolor="lightpink",
        fontsize="12",
    )
    dot.edge("Embedding", "Encoder1")

    # Word Pooling
    dot.node("WordPooling", "Word Pooling")
    dot.edge("Encoder1", "WordPooling")

    # Unpadding
    dot.node("Unpad", "Unpad Sequences")
    dot.edge("WordPooling", "Unpad")

    # Encoder 2
    dot.node(
        "Encoder2",
        "Transformer Encoder 2\n(RoPE)",
        fillcolor="lightpink",
        fontsize="12",
    )
    dot.edge("Unpad", "Encoder2")

    # Average Pooling
    dot.node(
        "AveragePooling", "Average Pooling\n(Packed Words)", width="2.0", height="0.6"
    )
    dot.edge("Encoder2", "AveragePooling")

    # Decoders
    dot.node("WordDecoder", "Word Decoder", fillcolor="lightgreen", fontsize="12")
    dot.edge("AveragePooling", "WordDecoder")

    dot.node(
        "SequenceDecoder", "Sequence Decoder", fillcolor="lightgreen", fontsize="12"
    )
    dot.edge("Encoder2", "SequenceDecoder")

    # Output Layers
    dot.node("OutputWord", "Word Output Layer", fillcolor="lightgray", fontsize="12")
    dot.edge(
        "WordDecoder",
        "OutputWord",
        label="Predicted Chars",
        fontsize="10",
        labeldistance="1.5",
        labelangle="30",
    )

    dot.node("OutputSeq", "Sequence Output Layer", fillcolor="lightgray", fontsize="12")
    dot.edge(
        "SequenceDecoder",
        "OutputSeq",
        label="Predicted Chars",
        fontsize="10",
        labeldistance="1.5",
        labelangle="-30",
    )

    # Cross-Entropy Loss
    dot.node(
        "WordLoss",
        "Cross-Entropy Loss\n(Word)",
        shape="box",
        style="dashed, rounded",
        fillcolor="white",
        fontsize="12",
        width="2.0",
        height="0.6",
    )
    dot.edge("OutputWord", "WordLoss", label="Logits", fontsize="10")

    dot.node(
        "SeqLoss",
        "Cross-Entropy Loss\n(Sequence)",
        shape="box",
        style="dashed, rounded",
        fillcolor="white",
        fontsize="12",
        width="2.0",
        height="0.6",
    )
    dot.edge("OutputSeq", "SeqLoss", label="Logits", fontsize="10")

    # Group Labels
    dot.node("EncodersLabel", "Encoders", shape="plaintext", fontsize="14")
    dot.edge("EncodersLabel", "Encoder1", style="invis")
    dot.edge("EncodersLabel", "Encoder2", style="invis")

    dot.node("DecodersLabel", "Decoders", shape="plaintext", fontsize="14")
    dot.edge("DecodersLabel", "SequenceDecoder", style="invis")
    dot.edge("DecodersLabel", "WordDecoder", style="invis")

    # Render and save
    dot.render(output_path, cleanup=True)
    print(f"Optimized model diagram saved to {output_path}.png")


create_optimized_model_diagram()
