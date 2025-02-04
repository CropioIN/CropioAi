"""Utility for colored console output."""

from typing import Optional


class Printer:
    """Handles colored console output formatting."""
    
    class RichColorPrinter:
        RESET = "\033[0m"
        ESC = "\033["

        # Standard 16 Colors (by name)
        COLORS = {
            "black": "30", "red": "31", "green": "32", "yellow": "33",
            "blue": "34", "magenta": "35", "cyan": "36", "white": "37",
            "bright_black": "90", "bright_red": "91", "bright_green": "92",
            "bright_yellow": "93", "bright_blue": "94", "bright_magenta": "95",
            "bright_cyan": "96", "bright_white": "97"
        }

        # Background Colors for Standard 16 Colors
        BG_COLORS = {k: v.replace("3", "4", 1) for k, v in COLORS.items()}

        # Text formatting styles
        FORMATS = {
            "bold": "1",
            "dim": "2",
            "italic": "3",
            "underline": "4",
            "blink": "5",
            "inverse": "7",
            "strikethrough": "9"
        }

        def _print(self, codes, content):
            """Constructs and prints the ANSI formatted text."""
            code_str = ";".join(codes)
            print(f"{self.ESC}{code_str}m{content}{self.RESET}")

        def print_color(self, content, color="white", style=None):
            """Prints text in the standard 16-color mode with an optional single style."""
            codes = [self.COLORS.get(color, self.COLORS["white"])]
            if style and style in self.FORMATS:
                codes.append(self.FORMATS[style])
            self._print(codes, content)

        def print_bg(self, content, fg_color="white", bg_color="blue"):
            """Prints text with a colored background."""
            fg = self.COLORS.get(fg_color, self.COLORS["white"])
            bg = self.BG_COLORS.get(bg_color, self.BG_COLORS["blue"])
            self._print([fg, bg], content)

        def print_256_fg(self, content, color_number):
            """Prints text with a 256-color foreground."""
            self._print([f"38;5;{color_number}"], content)

        def print_256_bg(self, content, color_number):
            """Prints text with a 256-color background."""
            self._print([f"48;5;{color_number}"], content)

        def print_256_fg_bg(self, content, fg_color_number, bg_color_number):
            """Prints text with 256-color foreground and background."""
            self._print([f"38;5;{fg_color_number}", f"48;5;{bg_color_number}"], content)

        def print_rgb_fg(self, content, r, g, b):
            """Prints text with an RGB foreground color."""
            self._print([f"38;2;{r};{g};{b}"], content)

        def print_rgb_bg(self, content, r, g, b):
            """Prints text with an RGB background color."""
            self._print([f"48;2;{r};{g};{b}"], content)

        def print_rgb_fg_bg(self, content, fg_rgb, bg_rgb):
            """Prints text with RGB foreground and background colors."""
            fg_r, fg_g, fg_b = fg_rgb
            bg_r, bg_g, bg_b = bg_rgb
            self._print([f"38;2;{fg_r};{fg_g};{fg_b}", f"48;2;{bg_r};{bg_g};{bg_b}"], content)

        def print_with_styles(self, content, color="white", styles=[]):
            """Prints text with multiple text styles (bold, italic, etc.)."""
            codes = [self.COLORS.get(color, self.COLORS["white"])]
            codes.extend([self.FORMATS[style] for style in styles if style in self.FORMATS])
            self._print(codes, content)

        def print(self, content, fg_color="white", bg_color=None, style=None, styles=None,
                  mode="standard"):
            """
            A simplified print method.

            Parameters:
                content (str): The text to print.
                fg_color (str or tuple): The foreground color (a standard color name or an RGB tuple).
                bg_color (str, tuple, or None): If provided, prints text with this background.
                style (str or None): A single text style (e.g., "bold").
                styles (list or None): A list of text styles.
                mode (str): The color mode ("standard", "256", or "rgb").
            """
            if mode == "standard":
                if bg_color:
                    self.print_bg(content, fg_color=fg_color, bg_color=bg_color)
                elif styles:
                    self.print_with_styles(content, color=fg_color, styles=styles)
                elif style:
                    self.print_color(content, color=fg_color, style=style)
                else:
                    self.print_color(content, color=fg_color)
            elif mode == "256":
                # For 256-color mode, we assume fg_color/bg_color are numbers.
                if bg_color is not None:
                    self.print_256_fg_bg(content, fg_color, bg_color)
                else:
                    self.print_256_fg(content, fg_color)
            elif mode == "rgb":
                # For RGB mode, we expect fg_color and bg_color to be 3-tuples.
                if bg_color is not None:
                    self.print_rgb_fg_bg(content, fg_color, bg_color)
                else:
                    self.print_rgb_fg(content, *fg_color)
            else:
                # Fallback to standard
                self.print_color(content, color=fg_color)

    def __init__(self):
        # Instantiate the nested RichColorPrinter for use.
        self.rcp = self.RichColorPrinter()
        # Define base color data.
        # For standard colors, the "value" is a string that corresponds to a key in RichColorPrinter.COLORS.
        # For additional colors, the "value" is an RGB tuple and the mode is set to "rgb".
        base_mapping = {
            "purple":  {"value": "magenta", "mode": "standard"},
            "red":     {"value": "red", "mode": "standard"},
            "green":   {"value": "green", "mode": "standard"},
            "blue":    {"value": "blue", "mode": "standard"},
            "yellow":  {"value": "yellow", "mode": "standard"},
            "cyan":    {"value": "cyan", "mode": "standard"},
            "magenta": {"value": "magenta", "mode": "standard"},
            "orange":  {"value": (255, 165, 0), "mode": "rgb"},
            "pink":    {"value": (255, 192, 203), "mode": "rgb"},
            "lime":    {"value": (0, 255, 0), "mode": "rgb"},
            "teal":    {"value": (0, 128, 128), "mode": "rgb"},
            "violet":  {"value": (238, 130, 238), "mode": "rgb"}
        }
        # Dynamically generate mapping entries for each base color and its variants.
        self.mapping = {}
        for name, data in base_mapping.items():
            value = data["value"]
            mode = data["mode"]
            # Plain color.
            self.mapping[name] = (lambda c, v=value, m=mode: self.rcp.print(c, fg_color=v, mode=m))
            # Underlined variant.
            self.mapping[name + "_underline"] = (lambda c, v=value, m=mode: self.rcp.print(c, fg_color=v, style="underline", mode=m))
            # Bold variant.
            self.mapping["bold_" + name] = (lambda c, v=value, m=mode: self.rcp.print(c, fg_color=v, style="bold", mode=m))
            # Bold + Underlined variant.
            self.mapping["bold_" + name + "_underline"] = (lambda c, v=value, m=mode: self.rcp.print(c, fg_color=v, styles=["bold", "underline"], mode=m))
    
    def print(self, content: str, color: Optional[str] = None):
        """
        Prints the provided content using preset formatting based on the given color name.
        Supported color names include base names and variants such as:
          - "red", "red_underline", "bold_red", "bold_red_underline"
          - "orange", "orange_underline", "bold_orange", "bold_orange_underline"
          - "violet", "violet_underline", "bold_violet", "bold_violet_underline"
          - ...and similarly for purple, green, blue, yellow, cyan, magenta, pink, lime, and teal.
        """
        if color and color in self.mapping:
            self.mapping[color](content)
        else:
            print(content)