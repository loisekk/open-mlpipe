"""Generate the OPENML PIPE ASCII art banner as a PNG image — polished AIDEN style."""

from PIL import Image, ImageDraw, ImageFont
import os

# ASCII art lines (matching the terminal screenshot)
ascii_art = [
    r' ██████╗ ██████╗ ███████╗███╗   ██╗███╗   ███╗██╗',
    r'██╔═══██╗██╔══██╗██╔════╝████╗  ██║████╗ ████║██║',
    r'██║   ██║██████╔╝█████╗  ██╔██╗ ██║██╔████╔██║██║',
    r'██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║╚██╔╝██║██║',
    r'╚██████╔╝██║     ███████╗██║ ╚████║██║ ╚═╝ ██║███████╗',
    r' ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝',
]

subtitle_lines = [
    'Production ML Pipeline | 14+ Models | One Line',
]

# Brand colors (match AIDEN style exactly)
BG_COLOR = (13, 17, 23)         # #0D1117 (near-black like AIDEN)
BANNER_COLOR = (255, 107, 53)   # #FF6B35 (orange — AIDEN exact)
BANNER_SHADOW = (200, 80, 35)   # Darker orange for shadow/depth
BORDER_COLOR = (48, 54, 61)     # #30363D (subtle dark gray border)
TEXT_DIM = (148, 163, 184)      # #94A3B8 (muted gray for status lines)
TEXT_WHITE = (248, 250, 252)    # #F8FAFC (white for subtitle)
TEXT_ACCENT = (255, 107, 53)    # #FF6B35 (orange for labels)
TEXT_GREEN = (34, 197, 94)      # #22C55E (green for "ACTIVE" status)

def generate_banner():
    width = 1200
    height = 420
    padding = 40

    img = Image.new('RGB', (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Try to get a monospace font
    try:
        font_art = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 24)
        font_sub = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
        font_label = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
        font_status = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    except OSError:
        try:
            font_art = ImageFont.truetype("C:/Windows/Fonts/cour.ttf", 24)
            font_sub = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
            font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
            font_status = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
        except OSError:
            font_art = ImageFont.load_default()
            font_sub = ImageFont.load_default()
            font_label = ImageFont.load_default()
            font_status = ImageFont.load_default()

    # Draw border (subtle dark gray like AIDEN)
    draw.rectangle([3, 3, width - 4, height - 4], outline=BORDER_COLOR, width=1)

    # Top-left status lines (like AIDEN)
    status_y = padding
    draw.text((padding, status_y), "// INITIATING OPENML PIPE...", fill=TEXT_DIM, font=font_label)
    status_y += 20
    draw.text((padding, status_y), "> SYSTEM ONLINE", fill=TEXT_DIM, font=font_status)
    status_y += 18
    draw.text((padding, status_y), "> CORE MODULES LOADED", fill=TEXT_DIM, font=font_status)
    status_y += 18
    draw.text((padding, status_y), "> PIPELINE MODE: ", fill=TEXT_DIM, font=font_status)
    draw.text((padding + 120, status_y), "ACTIVE", fill=TEXT_GREEN, font=font_status)

    # Top-right version label (like AIDEN)
    draw.text((width - padding - 80, padding), "openml v1.0.6", fill=TEXT_ACCENT, font=font_label)

    # Center the ASCII art vertically and horizontally
    line_height = 30
    total_art_height = len(ascii_art) * line_height

    # Calculate available vertical space
    top_space = padding + 60  # Below status lines
    bottom_space = 80  # Above subtitle
    available_height = height - top_space - bottom_space

    # Center vertically in available space
    art_start_y = top_space + (available_height - total_art_height) // 2

    # Find the widest line of ASCII art for horizontal centering
    max_art_width = 0
    for line in ascii_art:
        bbox = draw.textbbox((0, 0), line, font=font_art)
        line_width = bbox[2] - bbox[0]
        if line_width > max_art_width:
            max_art_width = line_width

    # Center horizontally
    art_start_x = (width - max_art_width) // 2

    # Draw ASCII art centered with shadow effect
    y = art_start_y
    for line in ascii_art:
        # Shadow effect — offset darker version
        draw.text((art_start_x + 2, y + 2), line, fill=BANNER_SHADOW, font=font_art)
        # Main text
        draw.text((art_start_x, y), line, fill=BANNER_COLOR, font=font_art)
        y += line_height

    # Draw subtitle centered below ASCII art
    y += 25
    for line in subtitle_lines:
        bbox = draw.textbbox((0, 0), line, font=font_sub)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y), line, fill=TEXT_WHITE, font=font_sub)
        y += 30

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'openml-banner.png')
    img.save(out_path, 'PNG')
    print(f"Banner saved to: {out_path}")

if __name__ == '__main__':
    generate_banner()
