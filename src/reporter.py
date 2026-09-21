import json
import os
import argparse
from collections import Counter

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)


# ---------------------------------------------------------
# Read JSON
# ---------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# Analyze results
# ---------------------------------------------------------
def analyze(data):
    """
    Status priority

    Failure:
        - verif_failure != None
        - followup_outputs == None

    Otherwise:
        Correct:
            all relations == True

        Incorrect:
            otherwise
    """

    counter = Counter()

    for sample in data["data"]:

        # Failure
        if sample.get("verif_failure") is not None:
            counter["Failure"] += 1
            continue

        if sample.get("followup_outputs") is None:
            counter["Failure"] += 1
            continue

        relations = sample.get("relations")

        if relations is None:
            counter["Failure"] += 1
            continue

        if all(relations):
            counter["Correct"] += 1
        else:
            counter["Incorrect"] += 1

    total = sum(counter.values())

    stats = {
        "Correct": counter["Correct"],
        "Incorrect": counter["Incorrect"],
        "Failure": counter["Failure"],
        "Total": total,
    }

    return stats


# ---------------------------------------------------------
# Pie chart
# ---------------------------------------------------------
def create_pie_chart(stats, output):
    labels = ["Correct", "Incorrect", "Failure"]

    values = [
        stats["Correct"],
        stats["Incorrect"],
        stats["Failure"],
    ]

    plt.figure(figsize=(6, 6))

    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
    )

    plt.title("MR Testing Result Distribution")
    plt.tight_layout()

    plt.savefig(output, dpi=300)
    plt.close()


# ---------------------------------------------------------
# Bar chart
# ---------------------------------------------------------
def create_bar_chart(stats, output):
    labels = ["Correct", "Incorrect", "Failure"]

    values = [
        stats["Correct"],
        stats["Incorrect"],
        stats["Failure"],
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(labels, values)

    plt.ylabel("Count")
    plt.title("MR Testing Summary")

    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha="center")

    plt.tight_layout()

    plt.savefig(output, dpi=300)
    plt.close()


# ---------------------------------------------------------
# PDF Default Report Name
# ---------------------------------------------------------
def default_report_name(json_path):
    filename = os.path.basename(json_path)

    if filename.startswith("log__"):
        filename = filename.replace("log__", "report__", 1)
    else:
        filename = f"report__{filename}"

    filename = os.path.splitext(filename)[0] + ".pdf"

    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)

    return os.path.join(report_dir, filename)


# ---------------------------------------------------------
# PDF Report
# ---------------------------------------------------------
def create_pdf(report_name, relation_name, stats, pie_path, bar_path):
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(report_name)

    story = []

    story.append(Paragraph("<b>LLM Metamorphic Testing Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))

    story.append(
        Paragraph(f"<b>Relation ID:</b> {relation_name}", styles["Normal"])
    )

    story.append(
        Paragraph(f"<b>Total Samples:</b> {stats['Total']}", styles["Normal"])
    )

    story.append(Spacer(1, 0.25 * inch))

    total = max(stats["Total"], 1)

    table = Table([
        ["Category", "Count", "Percentage"],
        [
            "Correct",
            stats["Correct"],
            f"{100*stats['Correct']/total:.2f}%"
        ],
        [
            "Incorrect",
            stats["Incorrect"],
            f"{100*stats['Incorrect']/total:.2f}%"
        ],
        [
            "Failure",
            stats["Failure"],
            f"{100*stats['Failure']/total:.2f}%"
        ],
    ])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),

        ("GRID",(0,0),(-1,-1),1,colors.black),

        ("BACKGROUND",(0,1),(-1,-1),colors.beige),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("BOTTOMPADDING",(0,0),(-1,0),10),
    ]))

    story.append(table)
    story.append(Spacer(1, 0.4 * inch))

    story.append(Paragraph("<b>Pie Chart</b>", styles["Heading2"]))
    story.append(Image(pie_path, width=5*inch, height=5*inch))

    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("<b>Bar Chart</b>", styles["Heading2"]))
    story.append(Image(bar_path, width=6*inch, height=4*inch))

    doc.build(story)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "json_file",
        help="Path to MR result json"
    )

    parser.add_argument(
        "--output",
        help="Output PDF filename (optional)"
    )

    args = parser.parse_args()

    if args.output is None: # Get default report name
        args.output = default_report_name(args.json_file)

    data = load_json(args.json_file)

    relation = data.get("relation_name", "Unknown")

    stats = analyze(data)

    pie = "pie_chart.png"
    bar = "bar_chart.png"

    create_pie_chart(stats, pie)
    create_bar_chart(stats, bar)

    create_pdf(
        args.output,
        relation,
        stats,
        pie,
        bar,
    )

    os.remove(pie)
    os.remove(bar)

    print("=" * 40)
    print("Relation :", relation)
    print("Total    :", stats["Total"])
    print("Correct  :", stats["Correct"])
    print("Incorrect:", stats["Incorrect"])
    print("Failure  :", stats["Failure"])
    print("=" * 40)

    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
