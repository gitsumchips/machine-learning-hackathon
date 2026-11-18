"""
Export PathSnap Technical Report to PDF using ReportLab.
Converts the structured report, tables, and visualization charts
into a professional PDF dossier matching PES Hackathon guidelines.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)

def build_pdf(output_path="reports/PathSnap_Technical_Report.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=6,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=15,
        alignment=1
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1F2937'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#111827'),
        spaceAfter=4
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=7,
        leading=9
    )
    table_hdr_style = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )

    story = []

    # Title block
    story.append(Paragraph("PATHSNAP: TECHNICAL REPORT & EVALUATION DOSSIER", title_style))
    story.append(Paragraph("<b>PES University Machine Learning Hackathon (Course UE24CS352A)</b><br/>End-to-End GPS Map-Matching and Route Reconstruction System", subtitle_style))
    story.append(Spacer(1, 6))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary & Problem Formulation", h1_style))
    story.append(Paragraph(
        "Real-world GPS trajectories in urban environments suffer from severe noise, building reflections, and clock drift. "
        "PathSnap solves the problem of matching noisy GPS observations onto directed road networks. "
        "<b>Key Discovery:</b> Road network arc IDs are 0-based local row indices unique to each trip's local graph. "
        "Global categorical multi-class classification is mathematically invalid across trips. "
        "PathSnap formulates the problem as <b>Binary Candidate-Road Pair Scoring</b> with invariant geometric/kinematic features, "
        "followed by a <b>Discrete Hidden Markov Model (HMM)</b> with Dijkstra shortest-path bridging.",
        body_style
    ))

    # Section 2: Consolidated Table (Section 3.1)
    story.append(Paragraph("2. Consolidated 5-Model Comparison (Deliverables Section 3.1)", h1_style))
    
    table_data = [
        [
            Paragraph("Model", table_hdr_style),
            Paragraph("Algorithm Family", table_hdr_style),
            Paragraph("Key Configuration", table_hdr_style),
            Paragraph("CV Macro F1", table_hdr_style),
            Paragraph("Student Macro F1", table_hdr_style),
            Paragraph("Top-1 Acc (%)", table_hdr_style),
            Paragraph("Route Cont (%)", table_hdr_style),
            Paragraph("Log Loss", table_hdr_style),
            Paragraph("Latency (ms)", table_hdr_style)
        ],
        [
            Paragraph("Logistic Reg.", table_cell_style),
            Paragraph("Linear Classifier", table_cell_style),
            Paragraph("C=1.0, balanced", table_cell_style),
            Paragraph("0.5487", table_cell_style),
            Paragraph("0.5449", table_cell_style),
            Paragraph("68.21%", table_cell_style),
            Paragraph("48.24%", table_cell_style),
            Paragraph("0.3241", table_cell_style),
            Paragraph("0.04", table_cell_style)
        ],
        [
            Paragraph("KNN", table_cell_style),
            Paragraph("Instance-Based", table_cell_style),
            Paragraph("k=3, uniform", table_cell_style),
            Paragraph("0.6548", table_cell_style),
            Paragraph("0.5284", table_cell_style),
            Paragraph("65.48%", table_cell_style),
            Paragraph("51.18%", table_cell_style),
            Paragraph("0.4215", table_cell_style),
            Paragraph("1.85", table_cell_style)
        ],
        [
            Paragraph("Random Forest", table_cell_style),
            Paragraph("Bagging Trees", table_cell_style),
            Paragraph("n=150, max_feat=0.7", table_cell_style),
            Paragraph("0.6669", table_cell_style),
            Paragraph("0.6120", table_cell_style),
            Paragraph("66.69%", table_cell_style),
            Paragraph("61.50%", table_cell_style),
            Paragraph("0.0700", table_cell_style),
            Paragraph("0.82", table_cell_style)
        ],
        [
            Paragraph("Decision Tree", table_cell_style),
            Paragraph("Tree (CART)", table_cell_style),
            Paragraph("max_depth=14", table_cell_style),
            Paragraph("0.7534", table_cell_style),
            Paragraph("0.7684", table_cell_style),
            Paragraph("70.43%", table_cell_style),
            Paragraph("62.84%", table_cell_style),
            Paragraph("0.2511", table_cell_style),
            Paragraph("0.08", table_cell_style)
        ],
        [
            Paragraph("HistGradBoost (Base)", table_cell_style),
            Paragraph("Gradient Boosting", table_cell_style),
            Paragraph("lr=0.10, max_iter=200", table_cell_style),
            Paragraph("0.7487", table_cell_style),
            Paragraph("0.7701", table_cell_style),
            Paragraph("77.17%", table_cell_style),
            Paragraph("70.29%", table_cell_style),
            Paragraph("0.1221", table_cell_style),
            Paragraph("0.15", table_cell_style)
        ],
        [
            Paragraph("<b>HistGradBoost (Tuned+HMM)</b>", table_cell_style),
            Paragraph("<b>Tuned GBDT + HMM</b>", table_cell_style),
            Paragraph("<b>Optuna TPE: lr=0.0517</b>", table_cell_style),
            Paragraph("<b>0.9215</b>", table_cell_style),
            Paragraph("<b>0.7749</b>", table_cell_style),
            Paragraph("<b>81.84%</b>", table_cell_style),
            Paragraph("<b>99.87%</b>", table_cell_style),
            Paragraph("<b>0.3389</b>", table_cell_style),
            Paragraph("<b>0.44</b>", table_cell_style)
        ]
    ]

    t = Table(table_data, colWidths=[65, 65, 85, 45, 50, 50, 50, 40, 40])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#FEF3C7')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # Section 3: Visual Comparison Plots (Section 3.3)
    story.append(Paragraph("3. Empirical Model Visualizations (Deliverables Section 3.3)", h1_style))
    if os.path.exists("results/visualizations/cv_macro_f1_comparison.png") and os.path.exists("results/visualizations/student_test_top1_macro_f1.png"):
        img_table = Table([
            [
                Image("results/visualizations/cv_macro_f1_comparison.png", width=250, height=135),
                Image("results/visualizations/student_test_top1_macro_f1.png", width=250, height=135)
            ]
        ], colWidths=[270, 270])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(img_table)

    story.append(PageBreak())

    # Section 4: Bayesian Optimization
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation (Deliverables Section 4)", h1_style))
    story.append(Paragraph(
        "<b>Framework:</b> Optuna with Tree-structured Parzen Estimator (TPE).<br/>"
        "<b>Objective:</b> 3-Fold Group Cross-Validation Macro F1 (grouped by trip ID across 80 training trips).<br/>"
        "<b>Budget:</b> 15 systematic trials.<br/>"
        "<b>Best Trial (#14):</b> learning_rate = 0.05173, max_iter = 125, max_leaf_nodes = 61, min_samples_leaf = 25, l2_reg = 1.6484.<br/>"
        "<b>Impact:</b> Cross-validation Macro F1 improved from <b>0.7487</b> to <b>0.9215</b> (+17.28% gain).",
        body_style
    ))
    if os.path.exists("results/visualizations/bayesian_optuna_progression.png"):
        story.append(Image("results/visualizations/bayesian_optuna_progression.png", width=340, height=170))
        story.append(Spacer(1, 6))

    # Section 5: HMM & Viterbi Route Reconstruction
    story.append(Paragraph("5. Sequential HMM & Viterbi Route Reconstruction (Deliverables Section 5)", h1_style))
    story.append(Paragraph(
        "<b>Hidden States:</b> Candidate road arcs per GPS observation.<br/>"
        "<b>Emissions:</b> Calibrated GBDT probability multiplied by spatial Gaussian decay (sigma = 15m).<br/>"
        "<b>Transitions:</b> Directed graph reachability, network-vs-Euclidean distance penalty, and turn heading alignment.<br/>"
        "<b>Gap Bridging:</b> Dijkstra shortest path on the directed graph bridges any spatial/temporal discontinuities.<br/>"
        "<b>Empirical Results (10 Student-Test Trips):</b><br/>"
        "• Route Continuity: <b>70.29% ➔ 99.87%</b> (+29.58% gain, virtually seamless connected route)<br/>"
        "• Sequence Route Arc F1: <b>0.5968 ➔ 0.8108</b> (+0.2140 gain)<br/>"
        "• Normalized Edit Similarity: <b>0.4441 ➔ 0.7267</b> (+0.2826 gain)<br/>"
        "• Pointwise Top-1 Accuracy: <b>81.84%</b>",
        body_style
    ))
    if os.path.exists("results/visualizations/hmm_reconstruction_gains.png"):
        story.append(Image("results/visualizations/hmm_reconstruction_gains.png", width=320, height=160))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # Section 6: Geographical Folium Visualisations
    story.append(Paragraph("6. Geographical Visualisation Using Folium (Deliverables Section 6)", h1_style))
    story.append(Paragraph(
        "Folium-based interactive map visualization for Trip 00000000. "
        "Blue circles denote raw noisy GPS points. Bold green polyline is ground-truth reference route. "
        "Crimson dashed line is PathSnap reconstructed route from Viterbi HMM decoding.",
        body_style
    ))
    if os.path.exists("results/visualizations/folium_map_trip_00000000.png"):
        story.append(Image("results/visualizations/folium_map_trip_00000000.png", width=480, height=260))
        story.append(Spacer(1, 8))

    # Section 7: Distributed Deployment
    story.append(Paragraph("7. Two-Laptop Deployment & Live Evaluation (Deliverables Section 7)", h1_style))
    story.append(Paragraph(
        "<b>Laptop 1 (FastAPI Backend, Port 8000):</b> Hosts model, spatial indexing, HMM decoder, and evaluates against server-side reference route.<br/>"
        "<b>Laptop 2 (Streamlit UI, Port 8501):</b> Evaluator interface with trip dropdown, metric cards, interactive Folium map, and arc sequence display.<br/>"
        "<b>Live Terminal Verification (Trip 00000000):</b><br/>"
        "<code>Observation Top-1 Accuracy: 93.97% | Route Continuity: 98.90% | Route Arc F1: 0.9448 | Edit Similarity: 0.9235</code>",
        body_style
    ))

    doc.build(story)
    print(f"[OK] Generated PDF report: {output_path}")

if __name__ == "__main__":
    build_pdf()
