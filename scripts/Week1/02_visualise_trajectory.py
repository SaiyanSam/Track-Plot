from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "results/trajectory_raw.csv"
TOPDOWN_OUT = PROJECT_ROOT / "results/trajectory_topdown.png"
HTML_OUT = PROJECT_ROOT / "results/trajectory_3d.html"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} trajectory points")
    print(df.head())

    # Top-down plot: X-Z plane, because Y is ground height
    plt.figure(figsize=(8, 6))
    plt.plot(df["world_x"], df["world_z"], marker="o", markersize=2, linewidth=1)

    plt.scatter(df["world_x"].iloc[0], df["world_z"].iloc[0], s=80, label="Start")
    plt.scatter(df["world_x"].iloc[-1], df["world_z"].iloc[-1], s=80, label="End")

    plt.xlabel("World X")
    plt.ylabel("World Z")
    plt.title("Top-Down Vehicle Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    TOPDOWN_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(TOPDOWN_OUT, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved top-down plot: {TOPDOWN_OUT}")

    # Interactive 3D plot
    fig = px.scatter_3d(
        df,
        x="world_x",
        y="world_y",
        z="world_z",
        color="frame_index",
        title="Interactive 3D Vehicle Trajectory",
        labels={
            "world_x": "World X",
            "world_y": "World Y",
            "world_z": "World Z",
            "frame_index": "Frame",
        },
    )

    fig.update_traces(marker=dict(size=3))
    fig.data[0].update(mode="markers+lines")

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="World X",
            yaxis_title="World Y",
            zaxis_title="World Z",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.write_html(str(HTML_OUT))

    print(f"Saved 3D HTML plot: {HTML_OUT}")


if __name__ == "__main__":
    main()
