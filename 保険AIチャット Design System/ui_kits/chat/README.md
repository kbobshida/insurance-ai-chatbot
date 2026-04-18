# UI Kit — Chat

Corporate, trust-forward recreation of the single chat screen. Navy + deep green + warm stone, with a serif display face for editorial weight.

## Files
- `index.html` — click-through prototype. Pick a starter question or type freely.
- `ChatApp.jsx` — state + mocked replies + mini markdown renderer.
- `Header.jsx`, `Composer.jsx`, `SourcePanel.jsx` — shell components.
- `MessageBubble.jsx`, `Avatar.jsx`, `TypingIndicator.jsx`, `ExampleChips.jsx` — message atoms.

## Notes
- Entirely new visual direction — does NOT inherit the original blue/gray Tailwind look.
- Backend is mocked (3 canned answers + a fallback). No real RAG call.
- Avatars are serif monograms (`AI` / `私`) on navy/stone discs — deliberately formal.
- The citation page-number badge uses gold — the only place gold appears in the system.
