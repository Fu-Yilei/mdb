from __future__ import annotations

import logging
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOT_STYLE_PRESETS: dict[str, dict[str, object]] = {
    "studio": {
        "paper_bg": "#f5f8fc",
        "plot_bg": "#ffffff",
        "font_color": "#0f2f4f",
        "grid": "#d7e2ee",
        "axis": "#8ea2b6",
        "menu_bg": "#ffffff",
        "menu_border": "#b8c6d6",
        "marker_line": "#ffffff",
        "marker_size": 10,
        "marker_opacity": 0.86,
        "accent_a": "rgba(27, 164, 176, 0.14)",
        "accent_b": "rgba(244, 133, 42, 0.10)",
        "colorway": [
            "#1f77b4",
            "#17a398",
            "#f28e2b",
            "#e15759",
            "#59a14f",
            "#edc949",
            "#4e79a7",
            "#76b7b2",
            "#ff9d52",
            "#9c755f",
        ],
    },
    "sunrise": {
        "paper_bg": "#fff8f1",
        "plot_bg": "#fffdf8",
        "font_color": "#4b2a1f",
        "grid": "#f0ddcc",
        "axis": "#c09c7f",
        "menu_bg": "#fffdf8",
        "menu_border": "#dfc0a6",
        "marker_line": "#fff8f1",
        "marker_size": 10,
        "marker_opacity": 0.84,
        "accent_a": "rgba(255, 167, 38, 0.16)",
        "accent_b": "rgba(0, 148, 136, 0.10)",
        "colorway": [
            "#e76f51",
            "#2a9d8f",
            "#f4a261",
            "#264653",
            "#8ab17d",
            "#e9c46a",
            "#4c956c",
            "#f08a5d",
            "#3d5a80",
            "#bc6c25",
        ],
    },
    "paper": {
        "paper_bg": "#fcfcfb",
        "plot_bg": "#ffffff",
        "font_color": "#222222",
        "grid": "#e1e3e5",
        "axis": "#a7adb3",
        "menu_bg": "#ffffff",
        "menu_border": "#c7ccd1",
        "marker_line": "#ffffff",
        "marker_size": 9,
        "marker_opacity": 0.82,
        "accent_a": "rgba(52, 152, 219, 0.10)",
        "accent_b": "rgba(39, 174, 96, 0.08)",
        "colorway": [
            "#1b6ca8",
            "#1f9d8a",
            "#e67e22",
            "#c0392b",
            "#2e7d32",
            "#c8a200",
            "#5d6d7e",
            "#2f855a",
            "#d35400",
            "#8d6e63",
        ],
    },
}


def maybe_merge_metadata(out: pd.DataFrame, metadata_path: str | None, logger=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not metadata_path or not os.path.isfile(metadata_path):
        return out, None

    meta = pd.read_csv(metadata_path, sep=None, engine="python")

    if "id" in meta.columns:
        out2 = out.copy()
        out2["id"] = out2["id"].astype(str)
        meta2 = meta.copy()
        meta2["id"] = meta2["id"].astype(str)
        out2 = out2.merge(meta2, on="id", how="left")
        return out2, meta

    if len(meta) == len(out):
        meta2 = meta.copy()
        meta2.index = out.index
        return pd.concat([out, meta2], axis=1), meta

    if logger:
        logger.info(f"Metadata not alignable; ignoring (meta_rows={len(meta)} vs n_rows={len(out)})")
    return out, None


def maybe_use_concise_sample_ids(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty or "id" not in out.columns:
        return out

    id_vals = out["id"].astype(str)
    candidate_cols: list[str] = []

    preferred = ["sample_id", "sample_id_y", "sample_name", "sample"]
    for col in preferred:
        if col in out.columns and col not in candidate_cols:
            candidate_cols.append(col)
    for col in out.columns:
        if col.startswith("sample_id") and col not in candidate_cols:
            candidate_cols.append(col)

    best_col = None
    best_mean_len = None
    for col in candidate_cols:
        raw = out[col]
        if raw.isna().any():
            continue
        vals = raw.astype(str).str.strip()
        if (vals == "").any():
            continue
        if vals.nunique(dropna=False) != len(out):
            continue
        if vals.equals(id_vals):
            continue
        mean_len = float(vals.str.len().mean())
        if best_col is None or mean_len < best_mean_len:
            best_col = col
            best_mean_len = mean_len

    if best_col is None:
        return out

    labels = out[best_col].astype(str).str.strip()
    out2 = out.copy()
    out2["id_original"] = out2["id"].astype(str)
    out2["id"] = labels
    if "sample_id" not in out2.columns:
        out2["sample_id"] = labels
    return out2


def normalize_color_value(color: object) -> object:
    if not isinstance(color, str):
        return color
    if color.startswith("rgb(") and color.endswith(")"):
        parts = [p.strip() for p in color[4:-1].split(",")]
        if len(parts) == 3:
            try:
                r, g, b = (max(0, min(255, int(float(part)))) for part in parts)
                return f"#{r:02x}{g:02x}{b:02x}"
            except ValueError:
                return color
    return color


def build_color_styles(df: pd.DataFrame, color_cols: list[str]) -> dict[str, dict[str, object]]:
    palette = [
        normalize_color_value(color)
        for color in (
            px.colors.qualitative.Safe
            + px.colors.qualitative.Set3
            + px.colors.qualitative.Plotly
            + px.colors.qualitative.Dark24
        )
    ]
    styles: dict[str, dict[str, object]] = {}
    for col in color_cols:
        if col not in df.columns:
            continue
        vals = pd.Series(df[col], dtype="string").fillna("NA").astype(str)
        ordered = sorted(vals.unique().tolist())
        cmap = {v: palette[i % len(palette)] for i, v in enumerate(ordered)}
        styles[col] = {"ordered": ordered, "cmap": cmap}
    return styles


def resolve_plot_styles(args) -> list[str]:
    primary = str(getattr(args, "plot_style", "studio"))
    if primary not in PLOT_STYLE_PRESETS:
        primary = "studio"
    styles = [primary]
    if bool(getattr(args, "plot_style_variants", False)):
        for name in PLOT_STYLE_PRESETS:
            if name != primary:
                styles.append(name)
    return styles


def apply_style_to_figure(fig: go.Figure, style_name: str, with_dropdown: bool) -> None:
    style = PLOT_STYLE_PRESETS.get(style_name, PLOT_STYLE_PRESETS["studio"])
    right_margin = 285 if with_dropdown else 95
    top_margin = 136 if with_dropdown else 96
    fig.update_layout(
        template="none",
        width=1120,
        height=860,
        paper_bgcolor=style["paper_bg"],
        plot_bgcolor=style["plot_bg"],
        colorway=style["colorway"],
        font=dict(family="Source Sans Pro, Arial, sans-serif", size=14, color=style["font_color"]),
        title=dict(x=0.01, y=0.98, xanchor="left", yanchor="top", font=dict(size=22, color=style["font_color"])),
        margin=dict(l=72, r=right_margin, t=top_margin, b=70),
        legend=dict(
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor=style["menu_border"],
            borderwidth=1,
            x=1.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=11, color=style["font_color"]),
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.97)",
            bordercolor=style["menu_border"],
            font=dict(size=12, color=style["font_color"]),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=style["grid"],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=style["axis"],
        linewidth=1.2,
        ticks="outside",
        tickcolor=style["axis"],
        title=dict(font=dict(size=14, color=style["font_color"])),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=style["grid"],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=style["axis"],
        linewidth=1.2,
        ticks="outside",
        tickcolor=style["axis"],
        title=dict(font=dict(size=14, color=style["font_color"])),
    )
    fig.update_traces(
        selector=dict(type="scatter"),
        marker=dict(
            size=style["marker_size"],
            opacity=style["marker_opacity"],
            line=dict(color=style["marker_line"], width=0.7),
        ),
    )
    fig.update_traces(
        selector=dict(type="splom"),
        marker=dict(
            size=max(5, int(style["marker_size"]) - 4),
            opacity=min(0.78, float(style["marker_opacity"])),
            line=dict(color=style["marker_line"], width=0.45),
        ),
    )
    if fig.layout.updatemenus:
        styled_menus = []
        for menu in fig.layout.updatemenus:
            menu2 = menu.to_plotly_json()
            menu2["bgcolor"] = style["menu_bg"]
            menu2["bordercolor"] = style["menu_border"]
            menu2["borderwidth"] = 1
            menu2["font"] = dict(size=12, color=style["font_color"])
            menu2["pad"] = dict(l=6, r=6, t=6, b=6)
            if with_dropdown:
                menu2["x"] = 0.01
                menu2["xanchor"] = "left"
                menu2["y"] = 1.15
                menu2["yanchor"] = "top"
                menu2["direction"] = "down"
            styled_menus.append(menu2)
        fig.update_layout(updatemenus=styled_menus)
        if with_dropdown:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.17,
                text="<b>Color By</b>",
                showarrow=False,
                font=dict(size=12, color=style["font_color"]),
                align="left",
            )


def make_dropdown_scatter(
    df,
    x,
    y,
    color_cols,
    hover_cols,
    title,
    color_styles: dict[str, dict[str, object]] | None = None,
    symbol_col: str | None = None,
    symbol_map: dict[str, str] | None = None,
    style_name: str = "studio",
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
):
    scatter_kwargs: dict[str, object] = {}
    if symbol_col and symbol_col in df.columns:
        scatter_kwargs["symbol"] = symbol_col
        if symbol_map:
            scatter_kwargs["symbol_map"] = symbol_map

    if not color_cols:
        fig = px.scatter(df, x=x, y=y, hover_data=hover_cols, title=title, **scatter_kwargs)
        apply_style_to_figure(fig, style_name=style_name, with_dropdown=False)
        fig.update_xaxes(title_text=x_axis_label or x)
        fig.update_yaxes(title_text=y_axis_label or y)
        return fig

    master = go.Figure()
    groups = []
    for i, col in enumerate(color_cols):
        if col not in df.columns:
            continue
        tmp_df = df.copy()
        tmp_df[col] = pd.Series(tmp_df[col], dtype="string").fillna("NA").astype(str)
        scatter_args: dict[str, object] = dict(scatter_kwargs)
        style = (color_styles or {}).get(col, {})
        if "ordered" in style:
            scatter_args["category_orders"] = {col: style["ordered"]}
        if "cmap" in style:
            scatter_args["color_discrete_map"] = style["cmap"]
        tmp = px.scatter(
            tmp_df,
            x=x,
            y=y,
            color=col,
            hover_data=hover_cols,
            title=f"{title} (color_by={col})",
            **scatter_args,
        )
        start = len(master.data)
        for tr in tmp.data:
            tr.visible = (i == 0)
            master.add_trace(tr)
        end = len(master.data)
        groups.append((col, start, end))

    if not groups:
        fig = px.scatter(df, x=x, y=y, hover_data=hover_cols, title=title, **scatter_kwargs)
        apply_style_to_figure(fig, style_name=style_name, with_dropdown=False)
        fig.update_xaxes(title_text=x_axis_label or x)
        fig.update_yaxes(title_text=y_axis_label or y)
        return fig

    buttons = []
    n_tr = len(master.data)
    for col, start, end in groups:
        vis = [False] * n_tr
        for idx in range(start, end):
            vis[idx] = True
        buttons.append(dict(label=col, method="update", args=[{"visible": vis}, {"title": f"{title} (color_by={col})"}]))

    master.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.02, xanchor="left", y=1.0, yanchor="top")]
    )
    apply_style_to_figure(master, style_name=style_name, with_dropdown=True)
    master.update_xaxes(title_text=x_axis_label or x)
    master.update_yaxes(title_text=y_axis_label or y)
    return master


def plotly_png_ok() -> bool:
    try:
        import kaleido  # noqa: F401

        return True
    except Exception:
        return False


def write_plotly_image_safe(fig, path: str, logger: logging.Logger) -> bool:
    try:
        fig.write_image(path)
        return True
    except Exception as exc:
        logger.warning(f"Skipping plotly PNG export for {path}: {exc}")
        return False


__all__ = [
    "PLOT_STYLE_PRESETS",
    "apply_style_to_figure",
    "build_color_styles",
    "make_dropdown_scatter",
    "maybe_merge_metadata",
    "maybe_use_concise_sample_ids",
    "plotly_png_ok",
    "resolve_plot_styles",
    "write_plotly_image_safe",
]
