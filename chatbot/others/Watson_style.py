# Watson_Style.py

from collections.abc import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

class WatsonDarkTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.indigo,   # Dark blue
        secondary_hue: colors.Color | str = colors.gray,   # Black/gray
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # Dark background
            body_background_fill="#000000",
            body_background_fill_dark="#0a0a0a",
            # Buttons: blue gradient
            button_primary_background_fill="linear-gradient(90deg, *primary_400, *primary_700)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_300, *primary_600)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *primary_900)",
            # Slider and block styling
            slider_color="*primary_300",
            slider_color_dark="*primary_600",
            block_title_text_weight="600",
            block_border_width="2px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="28px",
            # General text color
            text_color="white",
            text_color_dark="white",
        )
