from PIL import Image

def load_and_fit_logo(path: str, box: int = 72, padding: int = 6) -> Image.Image:
    """
    Charge un logo PNG, le 'crope' sur son contenu visible (alpha),
    puis le redimensionne pour tenir dans un carré box×box, centré,
    avec un léger padding.
    Retourne une image PIL RGBA (transparente).
    """
    img = Image.open(path).convert("RGBA")

    # Détecter la boîte englobante des pixels visibles (alpha > 0)
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Espace utile (on laisse un padding visuel)
    target = max(16, box - padding * 2)
    w, h = img.size
    if w == 0 or h == 0:
        # sécurité : image vide -> canvas transparent
        return Image.new("RGBA", (box, box), (0, 0, 0, 0))

    scale = min(target / w, target / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Créer canvas final et centrer
    canvas = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    off_x = (box - new_w) // 2
    off_y = (box - new_h) // 2
    canvas.paste(img, (off_x, off_y), img)
    return canvas
