"""Utility for colored console output."""

from typing import Optional
import re
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

        def format_ansi(self, codes, content) -> str:
            """Returns the ANSI formatted string (instead of printing it)."""
            code_str = ";".join(codes)
            return f"{self.ESC}{code_str}m{content}{self.RESET}"

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
        # For standard colors, the "value" is a string corresponding to a key in RichColorPrinter.COLORS.
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
        Prints the provided content using either inline markup or a preset color mapping.
        You can pass a default color via the `color` parameter. For example:
          - p.print("Hello World", color="bold_red")
          - p.print("[bold_lime]Agent:[/bold_lime] says hi", color="blue")
          
        When markup is detected in the content, any text outside of tags is formatted using the provided default color.
        """
        # If the content contains markup, process it with a default color for untagged segments.
        if "[" in content and "]" in content:
            formatted = self.format_markup(content, default_color=color)
            print(formatted)
        # Otherwise, if a color is provided and known, use that mapping.
        elif color and color in self.mapping:
            self.mapping[color](content)
        else:
            print(content)
    
    def print_markup(self, content: str):
        """Parses content containing tags like [bold_orange]…[/bold_orange] and prints it."""
        formatted = self.format_markup(content)
        print(formatted)
    
    def format_markup(self, content: str, default_color: Optional[str] = None) -> str:
        """
        Replaces any [tag]...[/tag] occurrences in the content with the appropriate
        ANSI escape sequences so that the text is printed with the intended style.
        Any text not wrapped in a tag is wrapped with the ANSI code corresponding
        to `default_color` (if provided and valid).
        """
        result = ""
        last_end = 0
        # Regex to capture markup tags like [tag]...[/tag]
        pattern = re.compile(r'\[([\w_]+)\](.*?)\[/\1\]', flags=re.DOTALL)
        for match in pattern.finditer(content):
            start, end = match.span()
            # Process text before the tag.
            pre_text = content[last_end:start]
            if pre_text:
                if default_color and default_color in self.mapping:
                    pre_text = self.rcp.format_ansi(
                        [self.rcp.COLORS.get(default_color, self.rcp.COLORS["white"])],
                        pre_text
                    )
                result += pre_text
            # Process the tagged segment.
            tag = match.group(1)
            inner_text = match.group(2)
            # _parse_tag returns (color, style, extra_styles, mode)
            color_parsed, style, extra_styles, mode = self._parse_tag(tag)
            if mode == "standard":
                codes = [self.rcp.COLORS.get(color_parsed, self.rcp.COLORS["white"])]
                if style:
                    codes.append(self.rcp.FORMATS.get(style))
                for s in extra_styles:
                    if s != style:  # avoid duplicate if already added
                        codes.append(self.rcp.FORMATS.get(s))
                result += self.rcp.format_ansi(codes, inner_text)
            elif mode == "rgb":
                if isinstance(color_parsed, tuple):
                    codes = [f"38;2;{color_parsed[0]};{color_parsed[1]};{color_parsed[2]}"]
                    result += self.rcp.format_ansi(codes, inner_text)
                else:
                    result += inner_text
            else:
                result += inner_text
            last_end = end
        # Process any trailing text after the last tag.
        if last_end < len(content):
            tail = content[last_end:]
            if default_color and default_color in self.mapping:
                tail = self.rcp.format_ansi(
                    [self.rcp.COLORS.get(default_color, self.rcp.COLORS["white"])],
                    tail
                )
            result += tail
        return result

    def _parse_tag(self, tag: str):
        """
        Given a tag string (for example "bold_orange_underline" or "bold"),
        determine the color, the primary style (if any), extra styles, and the mode.
        For a bare [bold] tag we default to white.
        """
        # Defaults
        mode = "standard"
        color = "white"
        style = None
        extra_styles = []
        # Special case: if the tag is exactly "bold"
        if tag == "bold":
            style = "bold"
            return color, style, extra_styles, mode
        # Check for a "bold_" prefix.
        if tag.startswith("bold_"):
            extra_styles.append("bold")
            tag = tag[len("bold_"):]
        # Check for an "_underline" suffix.
        if tag.endswith("_underline"):
            extra_styles.append("underline")
            tag = tag[:-len("_underline")]
        # Now decide on the color and mode.
        if tag in self.rcp.COLORS:
            color = tag
            mode = "standard"
        elif tag in {"purple", "red", "green", "blue", "yellow", "cyan", "magenta"}:
            color = tag
            mode = "standard"
        elif tag in {"orange", "pink", "lime", "teal", "violet"}:
            rgb_mapping = {
                "orange": (255, 165, 0),
                "pink": (255, 192, 203),
                "lime": (0, 255, 0),
                "teal": (0, 128, 128),
                "violet": (238, 130, 238)
            }
            color = rgb_mapping[tag]
            mode = "rgb"
        else:
            color = "white"
            mode = "standard"
        return color, style, extra_styles, mode
