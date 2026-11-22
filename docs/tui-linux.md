# Running the Agent TUI on Linux

The Agent TUI (`scripts/agent_tui.py`) shows live camera views using
`textual_image`. Image quality depends on the terminal emulator, so
the terminal matters as much as the code.

## Terminal requirements

The camera panel needs a terminal that supports the Kitty graphics
protocol (Kitty >= 0.28) or Sixel. Without either, `textual_image`
falls back to Unicode half-blocks, which look very low resolution.

| Terminal | Image support | Notes |
|---|---|---|
| Kitty >= 0.28 | Yes (recommended) | Use the official installer (see below) |
| WezTerm | Yes (Kitty protocol and Sixel) | |
| iTerm2 (macOS) | Yes (Sixel and inline images) | Development machine |
| Terminator | No | VTE has no Sixel or Kitty support on Ubuntu 22.04 |
| GNOME Terminal (VTE < 0.76) | No | Same limitation as Terminator |

Ubuntu 22.04 ships **Kitty 0.21**, which is too old. Its log shows
`Malformed GraphicsCommand ... invalid key character: 0x55` and no
image appears.

## Install a current Kitty (no sudo)

```bash
curl -L https://sw.kovidgoyal.net/kitty/installer.sh | sh /dev/stdin
~/.local/kitty.app/bin/kitty --version   # expect 0.28 or newer
```

Optional, so plain `kitty` starts the new build:

```bash
mkdir -p ~/.local/bin && ln -sf ~/.local/kitty.app/bin/kitty ~/.local/bin/kitty
hash -r && kitty --version
```

This requires `~/.local/bin` to come before `/usr/bin` in `PATH`.

## Run the TUI

From the repository root:

```bash
~/.local/kitty.app/bin/kitty uv run --offline --managed-python python scripts/agent_tui.py
```

Then press **Show Camera Views** in the UI.

## Expected log messages

- `[PARSE ERROR] Unknown OSC code: 22` is harmless. It comes from
  Textual setting the mouse pointer shape on older Kitty builds.
- `Malformed GraphicsCommand ... 0x55` means the Kitty version is too
  old. Upgrade as described above.

## Camera display notes

- The camera images keep their real aspect ratio (`.cam-img` uses
  `width: auto`), so there is empty space on either side of each image.
- The panel height is 15 rows. Maximising the terminal window gives
  more pixels. Increase `.cam-img` height in `scripts/agent_tui.py` for
  larger previews.
- Camera polling runs every 0.2 s, and refreshes never overlap. The
  TUI calls `get_frame` with `save_npy=False`, so it does not write a
  full-resolution `.npy` file on every poll.
- Frame rate is limited by the MCP round trip and by re-encoding each
  frame for the terminal. The Station web view is usually faster.

## SSH sessions

When running over SSH, the terminal emulator on the machine you are
typing at must support images. The remote host does not need any
terminal support.
