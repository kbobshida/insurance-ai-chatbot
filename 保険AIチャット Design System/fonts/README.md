# Fonts

**Noto Sans JP** (400, 700) — the only typeface in the system.

## Current loading method
Loaded from Google Fonts at runtime:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
```

## Offline / self-hosted
No local `.ttf` / `.woff2` files are included. If the project needs offline support,
download Noto Sans JP 400 + 700 from [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+JP) and drop the `.woff2` files in this folder, then swap the `@import` in `colors_and_type.css` for `@font-face` declarations.

**Flagged to user**: the source codebase has no bundled font files; we inherit that choice.
