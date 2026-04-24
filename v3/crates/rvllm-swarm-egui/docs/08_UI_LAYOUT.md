# 08 — UI layout for a 45" monitor

## Target display

Primary target: **45" widescreen monitor, 3840x2160 (4K UHD)**. Many 45"
displays are ultra-wide 5120x1440 or 5120x2160; the layout must work on
all of them.

Secondary target: **1440p laptop** (the developer's everyday screen). We
keep the layout functional here too, via scrolling, by designing the
grid to degrade from 6x5 to 5x4 visible + scroll.

## Panel map (4K UHD primary)

```
 width: 3840                 height: 2160
+------------------------------------------------------------------------+
| TOP BAR: master agent strip                                 ~200 px    |
+------------------------------------------------------------------------+
|                                              |                         |
|  AGENT GRID                                  |  SIDE PANEL             |
|  6 cols x 5 rows = 30 tiles                  |  - goal input           |
|  tile size: ~560 x 350 px with 8 px gutter   |  - task DAG view        |
|                                              |  - global log stream    |
|                                              |  - git overview         |
|                                              |  - scheduler config     |
|                                              |                         |
|  ~3360 px wide                               |  ~460 px wide           |
+------------------------------------------------------------------------+
|                                              |                         |
```

Top bar: 200 px. Remaining 1960 px divided among 5 grid rows (8 px
gutters): `(1960 - 6*8) / 5 = ~382 px` per tile height. Width: `3360 /
6 = 560 px`. These are sized in `ui/grid.rs` with fractional constraints
so they flex down on smaller screens.

## Tile anatomy (560 x 382 px)

```
+----------------------------------------------+
| [#] agent-01       kernels        RUNNING ● | <- 28 px header
| task: add 64x64 tile (3/6)                   |
|----------------------------------------------|
| tok/s 78.2        out 412      tool: cargo   | <- 22 px metric row
|----------------------------------------------|
| flash: "compiling rvllm-cutlass..."          | <- 24 px status
|----------------------------------------------|
|                                              |
|  history tail:                               |
|  > fs_read src/foo.rs                        |
|  > cargo_check()                             |
|    error: trait not impl for bar             | <- ~230 px body
|  > fs_patch src/foo.rs ...                   |
|  > cargo_check()                             |
|    ok                                        |
|                                              |
|----------------------------------------------|
|  worktree agent-01   HEAD 7d2a91f            | <- 28 px footer
|  [Pause] [Swap-out] [Reset]                  |
+----------------------------------------------+
```

Design tokens (in `src/theme.rs`):

- `TILE_BG_IDLE`, `TILE_BG_ACTIVE` — base fill by lifecycle.
- `BORDER_*` — one of 6 state colours (see `00_OVERVIEW.md`).
- `METRIC_FONT` — monospace, 13 px.
- `FLASH_FONT` — 12 px italic.
- `HISTORY_FONT` — monospace 11.5 px, with hang-indent.

A hover on any tile brings up a small overlay with the full prompt.
A click opens a modal (`ui/detail_modal.rs`) with the full transcript
and a "Send message to agent" footer (power-user feature).

## Top bar

```
+------------------------------------------------------------------------+
| rvLLM SWARM   [goal box...................................]   [Dispatch]|
| master: <status>    in-flight 7    queued 11    done 184    merges 23  |
| slots: [agent-01 HOT][agent-04 HOT][agent-12 HOT][agent-19 HOT]        |
| KV 12.3/14 GB   tok/s swarm 412   PPL spot 5.91   goal: <summary>      |
+------------------------------------------------------------------------+
```

The goal box is a single-line text field that accepts Cmd+Enter to
submit. Dispatch button is disabled while `MasterState::decomposition_in_flight`.

## Side panel

Fixed 460 px, scrollable vertically, five collapsible sections:

1. **Goal input** — multi-line, commit with `Cmd+Enter`.
2. **Current goal tree** — DAG view. Nodes 40x24 px, coloured by status,
   drawn via `egui::Painter` lines. Fallback to dense list past 40 tasks.
3. **Global log** — reverse chronological ring, last 200 lines.
4. **Git overview** — `main` HEAD, staging HEAD, open worktrees count,
   pending merge queue.
5. **Scheduler** — read-only `N_LIVE` indicator plus a "Adjust" button
   that opens a modal for live reconfig.

## Density and scaling

Every font size and padding is scaled by `ui_scale` in `AppState::settings`.
The user can bump it via hotkeys `Cmd+=` / `Cmd+-`, or set
`SWARM_UI_SCALE=1.2` env var on boot. Default `1.0`.

On a 1440p laptop (2560x1440), with `ui_scale = 0.85`, the layout
degrades cleanly: the side panel shrinks to 380 px, the grid becomes
5x4 visible + vertical scroll for the remaining 10 tiles. That's the
expected dev-laptop experience.

## Frame rate

Target: 60 FPS steady-state. Empirically `egui 0.31` with 30 custom
tiles + painter overlays is comfortable at ~0.7 ms CPU per frame on an
M2. The body of each tile uses `ScrollArea::vertical` with
`auto_shrink([false, false])` to avoid reflow flicker.

`ctx.request_repaint_after(Duration::from_millis(100))` during busy state
and event-driven repaints otherwise.

## Accessibility

- High-contrast border colours against the dark bg.
- No information conveyed by colour alone: state names render as text too.
- Keyboard nav: Tab cycles tiles, Space opens detail modal, `P`
  pause/resume, `R` reset, `Esc` closes modal.

## Screenshots (to be captured after first build)

`docs/screenshots/45in.png` and `docs/screenshots/1440p.png` — populated
during the UI smoke test.
