"""Generate the OPENML PIPE ASCII art banner as a PNG image."""

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

# Brand colors (match AIDEN style)
BG_COLOR = (13, 17, 23)         # #0D1117 (near-black like AIDEN)
BANNER_COLOR = (255, 107, 53)   # #FF6B35 (orange — AIDEN exact)
BANNER_SHADOW = (180, 70, 30)   # Darker orange for shadow/depth effect
SUBTITLE_COLOR = (148, 163, 184)  # #94A3B8 (muted)
ACCENT_COLOR = (255, 107, 53)   # #FF6B35 (orange)
WHITE = (248, 250, 252)         # #F8FAFC

def generate_banner():
    width = 1200
    height = 400
    padding = 40

    img = Image.new('RGB', (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Try to get a monospace font
    try:
        font_art = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 22)
        font_sub = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
        font_label = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
    except OSError:
        try:
            font_art = ImageFont.truetype("C:/Windows/Fonts/cour.ttf", 22)
            font_sub = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
            font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
        except OSError:
            font_art = ImageFont.load_default()
            font_sub = ImageFont.load_default()
            font_label = ImageFont.load_default()

    # Draw border (subtle dark gray like AIDEN)
    border_color = (48, 54, 61)  # #30363D
    draw.rectangle([3, 3, width - 4, height - 4], outline=border_color, width=1)

    # Draw "// INITIATING OPENML PIPE..." top-left (like AIDEN)
    draw.text((padding, padding - 20), "// INITIATING OPENML PIPE...", fill=(148, 163, 184), font=font_label)
    draw.text((width - padding - 80, padding - 20), "openml v1.0.6", fill=ACCENT_COLOR, font=font_label)

    # Draw ASCII art
    y = padding + 30
    for line in ascii_art:
        # Shadow effect — offset darker version
        draw.text((padding + 2, y + 2), line, fill=BANNER_SHADOW, font=font_art)
        # Main text
        draw.text((padding, y), line, fill=BANNER_COLOR, font=font_art)
        y += 28

    # Draw subtitle (like AIDEN: "Autonomous AI Engine" → "Production ML Pipeline")
    y += 20
    for line in subtitle_lines:
        # Center the subtitle
        bbox = draw.textbbox((0, 0), line, font=font_sub)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y), line, fill=WHITE, font=font_sub)
        y += 30

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'openml-banner.png')
    img.save(out_path, 'PNG')
    print(f"Banner saved to: {out_path}")

if __name__ == '__main__':
    generate_banner()
