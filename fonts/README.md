# Fonts

Khmer fonts bundled so any HTML/PDF output, demos, or documentation in this
repo render Khmer text correctly even on a system without Khmer fonts
installed.

| Family | File | Use |
|---|---|---|
| Noto Serif Khmer | `NotoSerifKhmer/NotoSerifKhmer.ttf` | body text / documents |
| Noto Sans Khmer | `NotoSansKhmer/NotoSansKhmer.ttf` | UI text |

Both are variable fonts (weight + width axes), full coverage of the 114
assigned code points in the Khmer Unicode block (U+1780–U+17FF), verified
with `fontTools` when they were added.

Source: [google/fonts](https://github.com/google/fonts) (`ofl/notoserifkhmer`,
`ofl/notosanskhmer`), maintained upstream by the
[Noto Project](https://github.com/notofonts/khmer).

License: SIL Open Font License 1.1 — see `OFL.txt` in each font's
directory. Redistribution and bundling is explicitly permitted by that
license.
