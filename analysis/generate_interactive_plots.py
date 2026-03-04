import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Create output directory for HTML files
output_dir = "/Users/pittawat/projects/2026/assets/html/2026-04-27-wait-do-we-need-to-wait"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("data/results.csv")

MODEL_NAME_MAPPING = {
    "RFT": "RFT",
    "simplescaling/s1.1-7B": "s1.1-7B",
    "open-thoughts/OpenThinker3-7B": "OpenThinker3-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-7B",
    "Qwen2.5 7B Instruct": "Qwen2.5 7B Instruct",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral 8B Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1 8B Instruct",
    "google/gemma-3-4b-it": "Gemma 3 4B Instruct",
}

DEFAULT_BENCHMARKS = ["AIME 2025", "MATH500", "MMLU Pro-1K", "SuperGPQA-1K"]

def get_plot_df(df, mapping):
    plot_df = df.copy()
    plot_df["Model_Display"] = plot_df["Model"].map(lambda x: mapping.get(x, x))
    plot_df["Approach"] = plot_df["Prompting"].map(
        lambda x: "Zero-shot" if x in ["Zero-shot", "CoT"] else "CoT+BF"
    )
    return plot_df

def write_responsive_html(fig, path):
    fig.write_html(path, include_plotlyjs="cdn", full_html=False, default_width="100%", config={"responsive": True})

def create_performance_comparison(df):
    plot_df = get_plot_df(df, MODEL_NAME_MAPPING)
    # Filter to only the reasoning models we focused on for clarity
    models_to_plot = ["RFT", "s1.1-7B", "OpenThinker3-7B", "DeepSeek-R1-7B"]
    plot_df = plot_df[plot_df["Model_Display"].isin(models_to_plot)]
    
    # Calculate means
    mean_df = plot_df.groupby(["Model_Display", "Approach"], as_index=False)[DEFAULT_BENCHMARKS].mean()
    
    # Create subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=models_to_plot)
    
    colors = {"Zero-shot": "#3498db", "CoT+BF": "#e74c3c"}
    
    for i, model in enumerate(models_to_plot):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        model_data = mean_df[mean_df["Model_Display"] == model]
        
        for approach in ["Zero-shot", "CoT+BF"]:
            app_data = model_data[model_data["Approach"] == approach]
            if not app_data.empty:
                y_values = [app_data[bench].values[0] for bench in DEFAULT_BENCHMARKS]
                fig.add_trace(go.Bar(
                    name=approach,
                    x=DEFAULT_BENCHMARKS,
                    y=y_values,
                    marker_color=colors[approach],
                    legendgroup=approach,
                    showlegend=(i == 0),
                    text=[f"{v:.1f}" for v in y_values],
                    textposition='auto',
                ), row=row, col=col)
                
    fig.update_layout(
        height=800, autosize=True,
        barmode='group',
        template="plotly_white",
        font=dict(size=14)
    )
    
    fig.update_yaxes(range=[0, 100], title_text="Score (%)", row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    fig.update_yaxes(range=[0, 100], title_text="Score (%)", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=2)
    write_responsive_html(fig, f"{output_dir}/fig_1_performance_comparison.html")

def create_average_performance(df):
    plot_df = get_plot_df(df, MODEL_NAME_MAPPING)
    models_to_plot = ["RFT", "s1.1-7B", "OpenThinker3-7B", "DeepSeek-R1-7B"]
    plot_df = plot_df[plot_df["Model_Display"].isin(models_to_plot)]
    plot_df["Average"] = plot_df[DEFAULT_BENCHMARKS].mean(axis=1)
    
    mean_df = plot_df.groupby(["Model_Display", "Approach"], as_index=False)["Average"].mean()
    models = models_to_plot
    
    fig = go.Figure()
    colors = {"Zero-shot": "#3498db", "CoT+BF": "#e74c3c"}
    
    for approach in ["Zero-shot", "CoT+BF"]:
        y_values = []
        for model in models:
            val = mean_df[(mean_df["Model_Display"] == model) & (mean_df["Approach"] == approach)]["Average"]
            y_values.append(val.values[0] if not val.empty else 0)
            
        fig.add_trace(go.Bar(
            name=approach,
            x=models,
            y=y_values,
            marker_color=colors[approach],
            text=[f"{v:.1f}" for v in y_values],
            textposition='auto',
        ))
        
    fig.update_layout(
        height=600, autosize=True,
        barmode='group',
        template="plotly_white",
        font=dict(size=14),
        xaxis_title="Model",
        yaxis_title="Average Score (%)"
    )
    
    write_responsive_html(fig, f"{output_dir}/fig_2_average_performance.html")

def create_qwen_non_reasoning_performance(df):
    plot_df = get_plot_df(df, MODEL_NAME_MAPPING)
    models = ["Qwen2.5 7B Instruct"]
    plot_df = plot_df[plot_df["Model_Display"].isin(models)]
    mean_df = plot_df.groupby(["Model_Display", "Approach"], as_index=False)[DEFAULT_BENCHMARKS].mean()
    
    fig = go.Figure()
    colors = {"Zero-shot": "#3498db", "CoT+BF": "#e74c3c"}
    for approach in ["Zero-shot", "CoT+BF"]:
        app_data = mean_df[mean_df["Approach"] == approach]
        if not app_data.empty:
            y_values = [app_data[bench].values[0] for bench in DEFAULT_BENCHMARKS]
            fig.add_trace(go.Bar(
                name=approach, x=DEFAULT_BENCHMARKS, y=y_values, marker_color=colors[approach],
                text=[f"{v:.1f}" for v in y_values], textposition='auto'
            ))
            
    fig.update_layout(
        height=400, autosize=True, barmode='group', template="plotly_white", font=dict(size=14)
    )
    fig.update_yaxes(range=[0, 100], title_text="Score (%)")
    write_responsive_html(fig, f"{output_dir}/fig_4_qwen_performance.html")

def create_qwen_non_reasoning_average(df):
    plot_df = get_plot_df(df, MODEL_NAME_MAPPING)
    models = ["Qwen2.5 7B Instruct"]
    plot_df = plot_df[plot_df["Model_Display"].isin(models)]
    plot_df["Average"] = plot_df[DEFAULT_BENCHMARKS].mean(axis=1)
    mean_df = plot_df.groupby(["Model_Display", "Approach"], as_index=False)["Average"].mean()
    
    fig = go.Figure()
    colors = {"Zero-shot": "#3498db", "CoT+BF": "#e74c3c"}
    for approach in ["Zero-shot", "CoT+BF"]:
        app_data = mean_df[mean_df["Approach"] == approach]
        if not app_data.empty:
            y_val = app_data["Average"].values[0]
            fig.add_trace(go.Bar(
                name=approach, x=["Qwen2.5 7B Instruct"], y=[y_val], marker_color=colors[approach],
                text=[f"{y_val:.1f}"], textposition='auto'
            ))
            
    fig.update_layout(
        height=400, autosize=True, barmode='group', template="plotly_white", font=dict(size=14), yaxis_title="Average Score (%)"
    )
    write_responsive_html(fig, f"{output_dir}/fig_5_qwen_average.html")

def create_other_non_reasoning_performance(df):
    plot_df = get_plot_df(df, MODEL_NAME_MAPPING)
    models_to_plot = ["Ministral 8B Instruct", "Llama-3.1 8B Instruct", "Gemma 3 4B Instruct"]
    plot_df = plot_df[plot_df["Model_Display"].isin(models_to_plot)]
    
    mean_df = plot_df.groupby(["Model_Display", "Approach"], as_index=False)[DEFAULT_BENCHMARKS].mean()
    fig = make_subplots(rows=1, cols=3, subplot_titles=models_to_plot)
    colors = {"Zero-shot": "#3498db", "CoT+BF": "#e74c3c"}
    
    for i, model in enumerate(models_to_plot):
        col = i + 1
        model_data = mean_df[mean_df["Model_Display"] == model]
        for approach in ["Zero-shot", "CoT+BF"]:
            app_data = model_data[model_data["Approach"] == approach]
            if not app_data.empty:
                y_values = [app_data[bench].values[0] for bench in DEFAULT_BENCHMARKS]
                fig.add_trace(go.Bar(
                    name=approach, x=DEFAULT_BENCHMARKS, y=y_values, marker_color=colors[approach],
                    showlegend=(i == 0), text=[f"{v:.1f}" for v in y_values], textposition='auto'
                ), row=1, col=col)
                
    fig.update_layout(
        height=450, autosize=True, barmode='group', template="plotly_white", font=dict(size=14)
    )
    for col in range(1, 4):
        fig.update_yaxes(range=[0, 100], title_text="Score (%)" if col == 1 else "", row=1, col=col)
        
    write_responsive_html(fig, f"{output_dir}/fig_6_other_performance.html")

def create_other_non_reasoning_average(df):
    plot_df = get_plot_df(df, MODEL_NAME_MAPPING)
    models_to_plot = ["Ministral 8B Instruct", "Llama-3.1 8B Instruct", "Gemma 3 4B Instruct"]
    plot_df = plot_df[plot_df["Model_Display"].isin(models_to_plot)]
    plot_df["Average"] = plot_df[DEFAULT_BENCHMARKS].mean(axis=1)
    
    mean_df = plot_df.groupby(["Model_Display", "Approach"], as_index=False)["Average"].mean()
    
    fig = go.Figure()
    colors = {"Zero-shot": "#3498db", "CoT+BF": "#e74c3c"}
    for approach in ["Zero-shot", "CoT+BF"]:
        y_values = []
        for model in models_to_plot:
            val = mean_df[(mean_df["Model_Display"] == model) & (mean_df["Approach"] == approach)]["Average"]
            y_values.append(val.values[0] if not val.empty else 0)
            
        fig.add_trace(go.Bar(
            name=approach, x=models_to_plot, y=y_values, marker_color=colors[approach],
            text=[f"{v:.1f}" for v in y_values], textposition='auto'
        ))
        
    fig.update_layout(
        height=450, autosize=True, barmode='group', template="plotly_white", font=dict(size=14), yaxis_title="Average Score (%)"
    )
    write_responsive_html(fig, f"{output_dir}/fig_7_other_average.html")

def create_keyword_comparison(df):
    plot_df = get_plot_df(df, MODEL_NAME_MAPPING)
    # Filter to only rows with keywords
    keyword_df = plot_df[plot_df["Keyword"].notna()].copy()
    keyword_df["Average"] = keyword_df[DEFAULT_BENCHMARKS].mean(axis=1)
    
    models_to_plot = [
        "Qwen2.5 7B Instruct",
        "RFT",
        "s1.1-7B",
        "DeepSeek-R1-7B",
        "Ministral 8B Instruct",
        "Llama-3.1 8B Instruct"
    ]
    
    # Filter to only rows with keywords and the target models
    keyword_df = plot_df[(plot_df["Keyword"].notna()) & (plot_df["Model_Display"].isin(models_to_plot))].copy()
    keyword_df["Average"] = keyword_df[DEFAULT_BENCHMARKS].mean(axis=1)
    
    # Calculate standard error across the 4 benchmarks for each row
    keyword_df["StdError"] = keyword_df[DEFAULT_BENCHMARKS].std(axis=1) / np.sqrt(len(DEFAULT_BENCHMARKS))
    
    # Use the specific model order requested
    models = [m for m in models_to_plot if m in keyword_df["Model_Display"].unique()]
    
    keywords = ["Wait", "Perhaps", "Let"]
    colors = {"Wait": "#3498db", "Perhaps": "#e74c3c", "Let": "#2ecc71"}
    
    # 2x3 grid for the 6 models
    rows = 2
    cols = 3
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=models)
    
    for i, model in enumerate(models):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        model_data = keyword_df[keyword_df["Model_Display"] == model]
        
        for kw in keywords:
            kw_data = model_data[model_data["Keyword"] == kw]
            if not kw_data.empty:
                y_val = kw_data["Average"].values[0]
                err_val = kw_data["StdError"].values[0]
                
                fig.add_trace(go.Bar(
                    name=kw,
                    x=[kw],
                    y=[y_val],
                    error_y=dict(type='data', array=[err_val], visible=True),
                    marker_color=colors[kw],
                    legendgroup=kw,
                    showlegend=(i == 0),
                    text=[f"{y_val:.1f}"],
                    textposition='auto',
                ), row=row, col=col)
                
    fig.update_layout(
        height=400 * rows, autosize=True,
        barmode='group',
        template="plotly_white",
        font=dict(size=14)
    )
    
    for row_idx in range(1, rows + 1):
        fig.update_yaxes(title_text="Average Score (%)", row=row_idx, col=1)
    
    write_responsive_html(fig, f"{output_dir}/fig_keyword_comparison.html")

def create_budget_scaling(df):
    plot_df = get_plot_df(df, MODEL_NAME_MAPPING)
    models_to_plot = ["RFT", "s1.1-7B", "OpenThinker3-7B", "DeepSeek-R1-7B"]
    scaling_df = plot_df[(plot_df["Budget"].notna()) & (plot_df["Model_Display"].isin(models_to_plot))].copy()
    
    # Average across duplicates if any, per benchmark
    mean_scale = scaling_df.groupby(["Model_Display", "Budget"], as_index=False)[DEFAULT_BENCHMARKS].mean(numeric_only=True)
    models = models_to_plot
    
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=DEFAULT_BENCHMARKS)
    
    for i, model in enumerate(models):
        model_data = mean_scale[mean_scale["Model_Display"] == model].sort_values("Budget")
        
        for j, bench in enumerate(DEFAULT_BENCHMARKS):
            row = (j // 2) + 1
            col = (j % 2) + 1
            
            show_legend = (j == 0)
            
            fig.add_trace(go.Scatter(
                x=model_data["Budget"],
                y=model_data[bench],
                mode='lines+markers',
                name=model,
                legendgroup=model,
                showlegend=show_legend,
                marker=dict(color=colors[i % len(colors)], size=10),
                line=dict(width=3)
            ), row=row, col=col)
            
    fig.update_layout(
        height=700, autosize=True,
        template="plotly_white",
        font=dict(size=14)
    )
    
    for row in range(1, 3):
        for col in range(1, 3):
            fig.update_xaxes(
                title_text="Budget (tokens)" if row == 2 else "",
                type="log", 
                tickmode="array", 
                tickvals=[256, 512, 1024, 2048, 4096, 8192],
                row=row, col=col
            )
            fig.update_yaxes(
                title_text="Score (%)" if col == 1 else "",
                row=row, col=col
            )
    
    write_responsive_html(fig, f"{output_dir}/fig_3_avg_score_linechart.html")

def create_qwen_budget_scaling(df):
    plot_df = get_plot_df(df, MODEL_NAME_MAPPING)
    models = ["Qwen2.5 7B Instruct"]
    scaling_df = plot_df[(plot_df["Model_Display"].isin(models)) & (plot_df["Budget"].notna())].copy()
    
    mean_scale = scaling_df.groupby(["Model_Display", "Budget"], as_index=False)[DEFAULT_BENCHMARKS].mean(numeric_only=True)
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=DEFAULT_BENCHMARKS)
    
    for i, model in enumerate(models):
        model_data = mean_scale[mean_scale["Model_Display"] == model].sort_values("Budget")
        
        for j, bench in enumerate(DEFAULT_BENCHMARKS):
            row = (j // 2) + 1
            col = (j % 2) + 1
            
            show_legend = (j == 0)
            
            if not model_data.empty:
                fig.add_trace(go.Scatter(
                    x=model_data["Budget"],
                    y=model_data[bench],
                    mode='lines+markers',
                    name=model,
                    legendgroup=model,
                    showlegend=False,
                    marker=dict(color="#3498db", size=10),
                    line=dict(width=3)
                ), row=row, col=col)
                
    fig.update_layout(
        height=700, autosize=True,
        template="plotly_white",
        font=dict(size=14)
    )
    
    for row in range(1, 3):
        for col in range(1, 3):
            fig.update_xaxes(
                title_text="Budget (tokens)" if row == 2 else "",
                type="log", 
                tickmode="array", 
                tickvals=[256, 512, 1024, 2048, 4096, 8192],
                row=row, col=col
            )
            fig.update_yaxes(
                title_text="Score (%)" if col == 1 else "",
                row=row, col=col
            )
            
    write_responsive_html(fig, f"{output_dir}/fig_8_qwen_budget_scaling.html")

if __name__ == "__main__":
    create_performance_comparison(df)
    create_qwen_non_reasoning_performance(df)
    create_average_performance(df)
    create_qwen_non_reasoning_average(df)
    create_other_non_reasoning_performance(df)
    create_other_non_reasoning_average(df)
    create_keyword_comparison(df)
    create_budget_scaling(df)
    create_qwen_budget_scaling(df)
    print("Successfully generated interactive HTML plots!")
