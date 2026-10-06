"""Shared presentation helpers for compact fixed live-sim plots."""

from __future__ import annotations

import re

PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 6
PAPER_LEGEND_SIZE = 6
PAPER_VALUE_SIZE = 6


def paper_title_case(text: str) -> str:
    """Title-case prose without damaging model names or acronyms (Qwen, FSM, SFT)."""
    minor = {'of', 'per', 'and', 'with', 'for', 'the', 'in', 'to'}
    def convert(match):
        word = match.group()
        if word in minor and match.start() > 0 and text[match.end():match.end() + 1] != '-':
            return word
        return word[0].upper() + word[1:]
    return re.sub(r'[A-Za-z][A-Za-z0-9]*', convert, text)


def style_paper_figure(fig) -> None:
    """Apply the same type hierarchy to every standalone paper figure."""
    for axis in fig.axes:
        axis.set_xlabel(paper_title_case(axis.get_xlabel()), fontsize=PAPER_LABEL_SIZE)
        axis.set_ylabel(paper_title_case(axis.get_ylabel()), fontsize=PAPER_LABEL_SIZE)
        for which in ('x', 'y'):
            labels = getattr(axis, f'get_{which}ticklabels')()
            positions = getattr(axis, f'get_{which}ticks')()
            getattr(axis, f'set_{which}ticks')(
                positions, [paper_title_case(label.get_text()) for label in labels],
                fontsize=PAPER_TICK_SIZE,
            )
        legend = axis.get_legend()
        if legend:
            # Rebuild at the final font size: changing Text alone leaves handle
            # geometry and spacing based on the original per-chart font size.
            anchor = legend.get_bbox_to_anchor().transformed(axis.transAxes.inverted())
            legend = axis.legend(
                handles=legend.legend_handles,
                labels=[label.get_text() for label in legend.get_texts()],
                loc=legend._loc, bbox_to_anchor=anchor,
                bbox_transform=axis.transAxes, ncol=legend._ncols,
                fontsize=PAPER_LEGEND_SIZE, frameon=False,
                borderaxespad=legend.borderaxespad, borderpad=legend.borderpad,
                columnspacing=legend.columnspacing, handletextpad=legend.handletextpad,
                labelspacing=legend.labelspacing,
            )
            for label in legend.get_texts():
                value = paper_title_case(label.get_text())
                # Wrap dense legends rather than using a smaller font in one figure.
                value = value.replace('Inset: Exactly One Physical-Work Agent',
                                      'Inset: Exactly One\nPhysical-Work Agent')
                value = value.replace('Qwen3-VL-8B-Thinking + Rationale SFT',
                                      'Qwen3-VL-8B-Thinking\n+ Rationale SFT')
                value = value.replace(' · ', '\n')
                label.set_text(value)
                label.set_fontsize(PAPER_LEGEND_SIZE)
            # Anchor the swatch to the first line, not the center of a wrapped
            # label. Disable TextArea's multiline baseline adjustment as well.
            columns = legend._legend_handle_box.get_children()
            # Match row heights across independently packed legend columns.
            for row in range(max(len(column.get_children()) for column in columns)):
                texts = [column.get_children()[row].get_children()[1].get_text()
                         for column in columns if row < len(column.get_children())]
                lines = max(text.count('\n') + 1 for text in texts)
                for column in columns:
                    if row < len(column.get_children()):
                        area = column.get_children()[row].get_children()[1]
                        text = area.get_text()
                        area.set_text(text + '\n' * (lines - text.count('\n') - 1))
            for column in columns:
                for entry in column.get_children():
                    entry.align = 'top'
                    for child in entry.get_children():
                        if hasattr(child, 'set_multilinebaseline'):
                            child.set_multilinebaseline(False)
        for label in axis.texts:
            label.set_fontsize(PAPER_VALUE_SIZE)
    # Align each patch's visible center to the first line's visible glyph center.
    # TextArea includes descent/line spacing, so box alignment alone is imprecise.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in fig.axes:
        legend = axis.get_legend()
        if legend is None:
            continue
        for handle, label in zip(legend.legend_handles, legend.get_texts()):
            if not hasattr(handle, 'set_y'):
                continue
            _, lines, _ = label._get_layout(renderer)
            _, _, (_, baseline_y) = lines[0]
            _, height, descent = renderer.get_text_width_height_descent(
                'H', label.get_fontproperties(), ismath=False,
            )
            origin_y = label.get_transform().transform(label.get_position())[1]
            text_center = origin_y + baseline_y + height / 2 - descent
            bounds = handle.get_window_extent(renderer)
            offset_points = (text_center - (bounds.y0 + bounds.y1) / 2) / renderer.points_to_pixels(1)
            handle.set_y(handle.get_y() + offset_points)


def label_bars_above_whiskers(
    axis,
    bars,
    values,
    upper_errors,
    *,
    gap: float = 0.012,
    fontsize: float = 5.6,
    minimum_value: float = 0.0,
) -> None:
    """Place vertical percentage labels a fixed data-space gap above CI caps."""

    for bar, value, upper_error in zip(bars, values, upper_errors):
        if value < minimum_value:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper_error + gap,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fontsize,
            clip_on=False,
        )
