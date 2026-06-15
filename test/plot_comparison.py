import openpyxl
import matplotlib.pyplot as plt
from pathlib import Path

here = Path(__file__).parent
wb = openpyxl.load_workbook(here / "Milvus-Chroma对比实验.xlsx")
ws = wb.active

rows = {row[0]: list(row[1:]) for row in ws.iter_rows(values_only=True) if row[0]}
turns = list(range(1, 12))

def plot(title, series: dict, filename):
    plt.figure()
    for label, values in series.items():
        plt.plot(turns, values, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Turn")
    plt.ylabel("Time (s)")
    plt.xticks(turns)
    plt.legend()
    plt.tight_layout()
    plt.savefig(here / filename)
    plt.close()

plot(
    "Memory Retrieval Time: Milvus vs Chromadb",
    {
        "Milvus": rows["Memory（Milvus）"],
        "Chromadb": rows["Memory（Chromadb）"],
    },
    "memory_comparison.png",
)

plot(
    "First Token Time: Milvus vs Chromadb",
    {
        "Milvus": rows["FirstToken（Milvus）"],
        "Chromadb": rows["FirstToken（Chromadb）"],
    },
    "firsttoken_comparison.png",
)

print("saved: memory_comparison.png, firsttoken_comparison.png")
