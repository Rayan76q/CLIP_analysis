import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def import_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)



def plot_comparison(df: pd.DataFrame,
                    base_column: str,
                    pred_suffix: str = '_pred',
                    true_suffix: str = '_true',
                    title: str = None) -> go.Figure:
    
    pred_col = f"{base_column}{pred_suffix}"
    true_col = f"{base_column}{true_suffix}"

    
    if pred_col not in df.columns or true_col not in df.columns:
        raise ValueError(f"Columns '{pred_col}' and/or '{true_col}' not found in DataFrame.")

    pred_counts = df[pred_col].value_counts().sort_index()
    true_counts = df[true_col].value_counts().sort_index()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"{base_column.capitalize()} Predicted", f"{base_column.capitalize()} True"))

    fig.add_trace(
        go.Bar(x=pred_counts.index.astype(str), y=pred_counts.values, name='Predicted'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=true_counts.index.astype(str), y=true_counts.values, name='True'),
        row=1, col=2
    )

    fig.update_layout(
        title_text=title or f"Comparison of '{pred_col}' vs '{true_col}' distributions",
        showlegend=False,
        width=800,
        height=400
    )

    fig.update_xaxes(title_text=base_column.capitalize(), row=1, col=1)
    fig.update_xaxes(title_text=base_column.capitalize(), row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=2)

    return fig


def calculate_error_across_classes(df: pd.DataFrame,
                                      base_column: str,
                                      pred_suffix: str = '_pred',
                                      true_suffix: str = '_true') -> None:
    
    pred_col = f"{base_column}{pred_suffix}"
    true_col = f"{base_column}{true_suffix}"

    if pred_col not in df.columns or true_col not in df.columns:
        raise ValueError(f"Columns '{pred_col}' and/or '{true_col}' not found in DataFrame.")

    accuracy_per_class = (df[pred_col] != df[true_col]).groupby(df[true_col]).mean()
    print("Global Error Rate: ", accuracy_per_class.mean().round(2))
    fig = go.Figure(
        data=[
            go.Bar(
                x=accuracy_per_class.index.astype(str),
                y=accuracy_per_class.values,
                marker=dict(color='skyblue'),
                text=accuracy_per_class.values.round(2),
                textposition='auto'
            )
        ]
    )

    fig.update_layout(
        title=f'Error rate per Class for {base_column.capitalize()}',
        xaxis_title=f'{base_column.capitalize()} Class',
        yaxis_title='Error Rate',
        xaxis=dict(tickmode='linear'),
        height=400,
        width=600
    )
    return fig

def confusion_matrix(df: pd.DataFrame,
                     base_column: str,
                     pred_suffix: str = '_pred',
                     true_suffix: str = '_true') -> None:
    pred_col = f"{base_column}{pred_suffix}"
    true_col = f"{base_column}{true_suffix}"

    if pred_col not in df.columns or true_col not in df.columns:
        raise ValueError(f"Columns '{pred_col}' and/or '{true_col}' not found in DataFrame.")

    # Create the confusion matrix
    confusion = pd.crosstab(df[true_col], df[pred_col], rownames=['True'], colnames=['Predicted'], margins=False)

    # Add annotations (numbers) to each square
    annotations = [[f"{value}" for value in row] for row in confusion.values]

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=confusion.values,
        x=confusion.columns.astype(str),
        y=confusion.index.astype(str),
        colorscale='Blues',  # Change the color mapping here
        text=annotations,  # Add annotations
        texttemplate="%{text}",  # Display annotations
        hoverinfo="z"  # Show only the value on hover
    ))

    # Update layout
    fig.update_layout(
        title=f'Confusion Matrix for {base_column.capitalize()}',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        height=600,
        width=600
    )

    return fig
