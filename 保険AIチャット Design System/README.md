# Hoken AI Chat — Design System

A trust-forward corporate design system for the **保険ドキュメント AIチャット** (Insurance Document AI Chat) — a Japanese-language RAG chatbot that answers insurance questions from PDF policy documents with grounded citations.

> **Note on direction.** The original shipped UI used a generic blue + gray Tailwind aesthetic. This system intentionally replaces it with a corporate, trust-forward identity (navy + deep green + warm stone) more appropriate for a financial/insurance product.

## Product Context

A single-screen web chat interface in Japanese. Users ask insurance questions; the AI answers with markdown prose and cites the exact PDF + page number.

- **Chat window** (main) — conversation transcript with AI and user bubbles.
- **Source panel** (right aside) — citation list; each item is a filename + page badge.
- **Example chips** — pre-populated starter questions.

Tone: polite, calm, authoritative. It's a financial product — no emoji, no playful gradients, restrained color use.

## Sources

- **Codebase**: `insurance-ai-chatbot/` — FastAPI + LangChain backend, static HTML/CSS/JS frontend. Single screen.
- **Sample PDFs**: Tokio Marine Nichido "Total Assist Auto Insurance" policy + pamphlet (referenced from the original README).
- No Figma, no brand guidelines, no logo file. The identity below is newly authored.

## Index

- `README.md` — this file
- `SKILL.md` — agent entry point
- `colors_and_type.css` — tokens (CSS variables) for color, type, space, radius, shadow
- `fonts/` — font loading notes (Google Fonts runtime)
- `assets/` — logo mark, icons
- `preview/` — design-system cards (what populates the Design System tab)
- `ui_kits/chat/` — React recreation of the chat screen (new visual direction)

---

## CONTENT FUNDAMENTALS

### Language
- **Japanese only.** All UI copy is in Japanese; no English fallback.
- **Voice: polite, slightly formal.** です／ます form in user-facing text. Uses honorific ご liberally (ご契約, ご質問).
- **Self-reference**: the product refers to itself as `AI` or by the full product name. Avoids 私/僕.
- **Addressing the user**: avoids あなた; prefers implicit subject.
- **Punctuation**: full-width Japanese (。、). No trailing period on short labels/buttons.
- **Mixed-script is fine**: `保険ドキュメント AIチャット` blends kanji + katakana + Latin + katakana.
- **No emoji.** No Unicode pictograms. Emphasis comes from type weight, color, and icon glyphs only.

### Example copy (copied from source, preserved verbatim)
- Greeting: `こんにちは。ご契約のしおりやパンフレットに関するご質問をどうぞ。`
- Input placeholder: `質問を入力してください…`
- Send button: `送信` → `送信中…`
- Source panel empty: `引用元はここに表示されます`
- Source panel title: `引用元`
- Off-topic refusal: `申し訳ありませんが、その質問にはお答えできません。保険の約款に関するご質問にのみ回答いたします。`

### Example starter questions
Concrete, specific, customer-style:
- `ロードアシストでレッカー搬送は何キロまで無料ですか？`
- `レンタカー費用補償は事故と故障で日数が違いますか？`
- `入院時選べるアシスト特約ではどんなサービスが選べますか？`

---

## VISUAL FOUNDATIONS

### Color
- **Primary (Navy)**: `--navy-700 #16375f` as the default interactive color; `--navy-900 #0b1d33` for headings and deep accents. Used on the Send button, user bubbles, links.
- **Secondary (Deep Green)**: `--green-700 #19573f`. Used for confirmation states, "cited" indicators, a second typographic accent.
- **Rare accent (Muted Gold)**: `--gold-500 #b88a3a`. Used ONLY on the citation page-number badge — a single touch of warmth that signals "verified / from the document."
- **Neutrals (Warm Stone)**: `--stone-50 #faf8f5` page background, `--stone-100 #f3efe9` secondary surface, `--stone-200 #e6dfd5` borders, `--stone-500 #847a6b` muted text, `--stone-900 #211d17` darkest body.
- **Card**: pure `#ffffff`.
- No blue-purple gradients, no bright blues, no rainbow palettes, no colored left-border accents.

### Typography
- **Display (headlines, card titles, slide titles)**: `Noto Serif JP` 500/700. The serif is the corporate-trust signal.
- **Body, UI, buttons, chips**: `Noto Sans JP` 400/500/700.
- **Mono (code, hex values, page numbers)**: `IBM Plex Mono` 400.
- Line-height: **1.7** on Japanese body (generous leading for CJK readability); **1.25** on headings.
- Letter-spacing: `palt` font-feature for proportional kana metrics in display copy.
- **No Inter, no Roboto, no system sans for Japanese.**

### Spacing & Layout
- 4px grid. Primary rhythm: 8 / 16 / 24 / 32.
- Page max-width: 960px for the chat card (narrower than the old 1152 — more focused, editorial).
- Source panel: 320px fixed.
- Card interior padding: 24px (body), 20px (header/footer).
- Message rows: 20px vertical gap.

### Radius
- **Conservative, rectilinear.** This is the biggest visual pivot from the old system.
- Cards: `--radius-lg 10px`
- Buttons, inputs: `--radius-md 6px`
- Message bubbles: `--radius-sm 4px` — nearly rectangular, document-like
- Page-number badge: `--radius-sm 4px`
- Avatar: `--radius-full` (still circle — the one soft element)

### Shadows
- Very subtle, cool-tinted (navy-based rgba) — never warm or pure black.
- `--shadow-1`: tokens/list items at rest
- `--shadow-2`: cards
- `--shadow-3`: elevated cards (composer focus, dropdown)
- No inner shadows.

### Borders
- 1px solid `--stone-200` default, `--stone-300` on hover.
- The AI bubble uses a **1px navy-900 top border** to distinguish it from the user bubble — not a colored background.
- User bubble is solid `--navy-700`, white text.
- No colored left-border stripes.

### Backgrounds & Imagery
- **Solid colors only.** No gradients on surfaces, no images, no textures, no patterns.
- Optional: a hair-thin horizontal rule in `--navy-700` (1px, 40px wide) as a section ornament — used sparingly, never decorative.

### Animation
- **Restrained.** Easing: `cubic-bezier(0.4, 0, 0.2, 1)` for everything.
- Message entry: 8px translate-up + opacity fade over 240ms. More subtle than the old 20px/400ms.
- Typing indicator: three navy-700 dots, 1.2s opacity-pulse (no scale bounce).
- Hover transitions: 150ms.

### Hover / Press States
- Primary button: background shifts from `--navy-700` to `--navy-800`.
- Secondary/ghost button: background from transparent to `--stone-100`.
- Chips: 1px inset border appears in `--navy-700`.
- Press: no scale/shrink. We use 200ms color hold for tactile feedback.
- Focus: 2px `--focus-ring` (navy, rgba 0.35) box-shadow at 2px offset.

### Transparency & Blur
- **Not used.** All surfaces fully opaque.

### Card pattern
White background + `--shadow-2` + `--radius-lg 10px` + 1px `--stone-200` border. Optional top hairline in `--navy-700` for "premium" cards.

### Fixed Elements
- Page header static; chat card scrolls internally.
- No floating action buttons, no sticky chips.

---

## ICONOGRAPHY

- **Style**: 1.5px stroke, rounded caps/joins, `currentColor`. Thinner and more refined than the old 2.5-stroke — matches the editorial serif display type.
- **Library**: **Lucide** at stroke-width 1.5 is the house substitute. Loaded per-icon as inline SVG — no icon font, no sprite sheet, no PNG.
- **Native icons** (authored for this system):
  - `assets/icon-mark.svg` — the brand mark (shield + 保 kanji).
  - `assets/icon-cite.svg` — a document-pin glyph for citations.
- **Emoji**: never used.
- **Unicode pictograms**: never used.

### When more icons are needed
Pick from Lucide, set stroke-width=1.5. Flag each addition when committing so the set stays curated.

---

## FONT SUBSTITUTION NOTES

- Loaded from Google Fonts at runtime (Noto Sans JP, Noto Serif JP, IBM Plex Mono). Not self-hosted.
- No local `.woff2` files. For offline bundling, download from Google Fonts and swap `@import` for `@font-face`. Flagged.

---

## CAVEATS (please iterate with me)

1. **No real logo.** `assets/icon-mark.svg` is a type-set 保 (insurance) kanji on a navy shield. If you have an official mark, drop it in `assets/` and update `preview/logo.html`.
2. **Palette is newly authored.** I chose navy + deep green + warm stone because you asked for a corporate, trust-forward direction — if the brand has an actual color code book, swap the tokens in `colors_and_type.css`.
3. **Serif display is bold.** Noto Serif JP on headings is a strong choice. If it feels too traditional, I can swap to Zen Kaku Gothic New for a modern/rounded feel, or to a sans-only system.
4. **Single surface.** The codebase ships one screen. The UI kit mirrors that — I have NOT invented login, settings, or history views.
5. **No bundled fonts / PDFs.** Fonts come from Google Fonts; sample PDF assets (policy docs) are not included.

---

## Bold ask for you

Please eyeball **`preview/colors-accent.html`**, **`preview/type-scale.html`**, and the UI kit at **`ui_kits/chat/index.html`**. I need your call on three things:

1. Is the **navy-700 + deep-green + gold accent** palette right, or should we shift (deeper navy? less green? no gold?)?
2. Is **Noto Serif JP for display** too editorial? Try the Zen Kaku Gothic New alternative I can swap in on a single token change.
3. Should the user bubble be **solid navy with white text** (current), or should we flip — **AI bubble navy, user bubble white/stone**?

Once those are locked I'll tune the rest — spacing rhythm, chip shape, composer treatment — to match.
