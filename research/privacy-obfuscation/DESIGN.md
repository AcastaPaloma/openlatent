---
name: privharness
description: A local proof sheet for choosing how to disguise a dataset, with the evidence one step away.
colors:
  grease: "oklch(54% .205 28)"
  grease-dark: "oklch(46% .18 28)"
  wax: "oklch(86% .16 95)"
  wax-ink: "oklch(45% .09 90)"
  edge: "oklch(82% .13 78)"
  fail: "oklch(45% .16 25)"
  fail-bg: "oklch(95% .02 25)"
  paper: "oklch(97% .002 250)"
  paper-2: "oklch(93.5% .003 250)"
  sheet-white: "#ffffff"
  rule: "oklch(86% .004 250)"
  ink: "oklch(21% .008 260)"
  ink-2: "oklch(42% .008 260)"
  ink-3: "oklch(52% .008 260)"
  film: "oklch(15% .004 60)"
  film-2: "oklch(24% .005 60)"
  film-text-dim: "oklch(68% .01 80)"
  film-text: "oklch(88% .01 80)"
  film-text-bright: "oklch(92% .01 80)"
typography:
  display:
    fontFamily: "\"Helvetica Neue\", Helvetica, -apple-system, BlinkMacSystemFont, Arial, sans-serif"
    fontSize: "2rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-.02em"
  headline:
    fontFamily: "\"Helvetica Neue\", Helvetica, -apple-system, BlinkMacSystemFont, Arial, sans-serif"
    fontSize: "1.75rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-.02em"
  title:
    fontFamily: "\"Helvetica Neue\", Helvetica, -apple-system, BlinkMacSystemFont, Arial, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-.01em"
  subtitle:
    fontFamily: "\"Helvetica Neue\", Helvetica, -apple-system, BlinkMacSystemFont, Arial, sans-serif"
    fontSize: ".95rem"
    fontWeight: 600
    lineHeight: 1.5
  body:
    fontFamily: "\"Helvetica Neue\", Helvetica, -apple-system, BlinkMacSystemFont, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  small:
    fontFamily: "\"Helvetica Neue\", Helvetica, -apple-system, BlinkMacSystemFont, Arial, sans-serif"
    fontSize: ".87rem"
    fontWeight: 400
    lineHeight: 1.5
  score:
    fontFamily: "\"Helvetica Neue\", Helvetica, -apple-system, BlinkMacSystemFont, Arial, sans-serif"
    fontSize: ".84rem"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tnum"
  mono:
    fontFamily: "ui-monospace, \"SF Mono\", Menlo, Consolas, monospace"
    fontSize: ".78rem"
    fontWeight: 400
    lineHeight: 1.5
  film-mark:
    fontFamily: "ui-monospace, \"SF Mono\", Menlo, Consolas, monospace"
    fontSize: "10px"
    fontWeight: 400
    lineHeight: 1
rounded:
  xs: "2px"
  sm: "3px"
  hand-cut: "5px 3px 6px 3px / 3px 6px 3px 5px"
  full: "50%"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  column: "36px"
  section: "44px"
components:
  button:
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: "6px 14px"
    height: "36px"
  button-hover:
    backgroundColor: "{colors.paper-2}"
  button-active:
    backgroundColor: "{colors.rule}"
  button-primary:
    backgroundColor: "{colors.grease}"
    textColor: "{colors.sheet-white}"
    rounded: "{rounded.hand-cut}"
    padding: "6px 14px"
    height: "36px"
  button-primary-hover:
    backgroundColor: "{colors.grease-dark}"
  button-danger:
    textColor: "{colors.fail}"
    rounded: "{rounded.sm}"
    padding: "6px 14px"
    height: "36px"
  button-danger-hover:
    backgroundColor: "{colors.fail-bg}"
  input:
    backgroundColor: "{colors.sheet-white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "6px 10px"
    height: "36px"
  choice:
    backgroundColor: "{colors.sheet-white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "10px 12px"
  film-frame:
    backgroundColor: "{colors.film}"
    padding: "15px 4px"
  film-frame-small:
    backgroundColor: "{colors.film}"
    padding: "11px 2px"
  stages:
    backgroundColor: "{colors.film}"
    textColor: "{colors.film-text-dim}"
    padding: "9px 10px"
  stage-done:
    textColor: "{colors.film-text}"
  stage-active:
    backgroundColor: "{colors.edge}"
    textColor: "{colors.film}"
  stage-halted:
    backgroundColor: "{colors.fail}"
    textColor: "{colors.sheet-white}"
  loupe:
    backgroundColor: "{colors.sheet-white}"
    rounded: "{rounded.sm}"
    padding: "16px"
    width: "380px"
  note:
    backgroundColor: "{colors.paper-2}"
    rounded: "{rounded.sm}"
    padding: "10px 12px"
  error:
    backgroundColor: "{colors.fail-bg}"
    textColor: "{colors.fail}"
    rounded: "{rounded.sm}"
    padding: "9px 12px"
  log:
    backgroundColor: "{colors.film}"
    textColor: "{colors.film-text}"
    typography: "{typography.mono}"
    padding: "12px"
  path-line:
    backgroundColor: "{colors.film}"
    textColor: "{colors.film-text-bright}"
    rounded: "{rounded.xs}"
    padding: "18px 14px"
---

# Design System: privharness

## Overview

**Creative North Star: "The Editor's Proof Sheet"**

Choosing a disguise works like a photo editor choosing a frame. The interface is cool photo paper with black film laid on it: frames butt together into strips with sprocket holes and amber edge numbers, and a red grease pencil circles the pick. Results are read as a sheet of pictures and plain sentences, with the attacker scores, intervals and checks one step away in the loupe.

Density is that of a working document. Rules and whitespace separate content rather than cards; only the loupe lifts off the paper. Film carries what the machine recorded, paper carries what the tool says about it, and the hand-drawn marks carry the decision. Frames also report progress: they sit undeveloped under a dim safelight while attacks run and develop once those attacks finish.

The refused look is the metric-card ML dashboard. Scores appear as captions under frames and as ticks on a thin scale, never as tiles of large numbers.

**Key Characteristics:**
- Cool paper, black film, and three hand or machine marks: grease red, wax yellow, edge amber.
- A single Helvetica-style sans at a compact ramp, with monospace for the machine's literal strings.
- Flat by default; the loupe is the one lifted surface.
- Tight, square corners, with a slightly uneven cut on the primary button.
- Hand-drawn SVG marks with deliberately imperfect paths.
- Good news is black ink; bad news gets its own red and extra weight.

## Colors

A near-achromatic paper and film base with marking colors reserved for decisions, state and failure.

### Primary
- **Grease Pencil Red** (oklch(54% .205 28)): The editor's hand. Circles and names the pick, crosses out failed frames, fills the primary action, marks the file-name piece to hide, draws the score tick and its interval band, and carries focus rings, text selection and the caret.
- **Pressed Grease** (oklch(46% .18 28)): Primary button hover, and the score named in a legend where red text must read at caption size.

### Secondary
- **Yellow Wax** (oklch(86% .16 95)): The second-choice mark: a stroke along the bottom of an alternate frame, and the highlight on file-name pieces marked as recording or scene. Used only on or against film, never for text on paper.
- **Wax Ink** (oklch(45% .09 90)): The legible text form of wax, for the alternate's caption tag on paper.

### Tertiary
- **Edge Amber** (oklch(82% .13 78)): Film edge markings and machine activity: frame numbers, the VM check time, the active pipeline stage, the running status dot, the selected frame's inner outline, and the selected point on the trade-off map.

### Status
- **Report Red** (oklch(45% .16 25)): What the system reports as wrong: error boxes, failed and weak statuses, failed checks, the halted stage, the destructive button.
- **Report Wash** (oklch(95% .02 25)): The background behind error boxes and the destructive button's hover.

### Neutral
- **Cool Photo Paper** (oklch(97% .002 250)): The page, and the holes punched in the film.
- **Shaded Paper** (oklch(93.5% .003 250)): Notes, the final-result box, button hover.
- **Clean Sheet** (#ffffff): Surfaces that sit on the paper and take input: the loupe, fields, choices, folder buttons.
- **Hairline** (oklch(86% .004 250)): Section and row rules, table lines, resting choice borders, button pressed state.
- **Ink** (oklch(21% .008 260)): Primary text, finished status dots, the current-page underline.
- **Soft Ink** (oklch(42% .008 260)): Supporting sentences, captions, inactive navigation.
- **Pale Ink** (oklch(52% .008 260)): Control borders, placeholders, the resting ticks on a scale and points on the map.
- **Film Black** (oklch(15% .004 60)): Frames, the stage bar, logs, the file-name line, VM headers.
- **Film Shadow** (oklch(24% .005 60)): Empty frame ground, stage dividers, token hover.
- **Film Light Grey** (oklch(68% .01 80)): Stages not yet reached.
- **Film Light** (oklch(88% .01 80)): Completed stages and log text.
- **Film Bright** (oklch(92% .01 80)): File-name text and the kept-token fill.

### Named Rules
**The Hand and Report Rule.** Grease is a decision someone marks; Report Red is the system saying something went wrong. A failed frame is crossed out in grease because the sheet is being edited, while its status sentence is in Report Red.

**The Black Ink Good News Rule.** Interface chrome has no success color. A passed check, a finished dot and a strong result are all ink, so failure is the only thing that changes color. The drawn scene inside a frame has its own illustration colors, which include greens and are not system tokens.

**The Silver Heat Rule.** Density tables shade from paper to dark in one neutral hue (oklch lightness 96% to 28% at .004 chroma, hue 260), switching to white text below 62% lightness. Red stays free to mean the pick.

## Typography

**Display Font:** Helvetica Neue (with Helvetica, then the platform sans, then Arial)
**Label/Mono Font:** the platform monospace (ui-monospace, SF Mono, Menlo, Consolas)

**Character:** Labels on a darkroom box: plain, tight and unornamented. Hierarchy comes from size and a single 600 weight. Nothing is loaded over the network, so the tool runs with nothing to install.

### Hierarchy
- **Display** (600, 2rem, 1.2; 1.55rem under 720px): The verdict sentence on a dataset page, balanced across its lines.
- **Headline** (600, 1.75rem, 1.2): Page titles such as Datasets.
- **Title** (600, 1.15rem, 1.3): Dataset names, section headings, the loupe's disguise name.
- **Subtitle** (600, .95rem): Named blocks in the loupe and beside the verdict.
- **Body** (400, 15px, 1.5): Sentences, capped at 68ch where they run long. The verdict's explanation steps up slightly to 1.03rem.
- **Small** (400, .87rem): Status lines, crumbs, supporting notes.
- **Score** (400, .84rem, tabular numerals): Scores under frames and scale legends, in the sans with the numeral in ink and its word in soft ink.
- **Mono** (400, .78rem, 1.5): Logs and raw configuration. Inline mono runs at .92em with tabular numerals.
- **Film Mark** (400, 10px, 1): Frame numbers such as 1A on the film, amber with a dark halo. The VM check time uses the same voice at 11px.

### Named Rules
**The Machine's Own Words Rule.** Monospace is for strings the machine wrote verbatim and for film markings: file names, logs, JSON, harness names, column names, frame numbers. Scores people read stay in the sans with tabular numerals. Sentences are never mono.

**The Sentence Case Rule.** The build has no uppercase styling and no tracked-out labels. Headings and labels are sentence case and say what they mean.

## Layout

A centered page up to 1360px wide with 24px gutters and generous bottom padding (28px 24px 72px), under a slim header with brand, navigation and the primary action at the far right. Sections are separated by a hairline and 44px rather than by containers.

A dataset page opens with the verdict sentence beside what it does not prove, in two equal columns (48px gap), with the stage bar above them until the run finishes. The sheet follows beside a 380px sticky loupe (36px gap). The sheet is an auto-filling grid of frames at least 168px wide with no horizontal gap and 22px between rows, so frames join into film. On the home page each dataset is a ruled row with a six-frame strip; a row's heading is a three-part grid of name, status sentence and location.

Spacing values are literal rather than variables. The recurring steps are 4, 8, 12, 16 and 24px, with 6, 10, 14, 18 and 22px common for internal padding and gaps. Long forms stay within 860px and are split by rules into steps.

At 1040px the loupe drops below the sheet and stops sticking, and the verdict stacks. At 720px gutters narrow to 16px, the full sheet shows two frames across, the home strip shows the first three frames, the stage bar stacks vertically, and buttons and choices grow to 44px.

## Elevation & Depth

Flat by default. Depth comes from placing dark film on paper and white sheets on paper, with hairlines between. Shadows exist only as state and as the single lifted loupe.

### Shadow Vocabulary
- **Loupe lift** (`box-shadow: 0 8px 26px -14px oklch(21% .008 260 / .35)`): The loupe only, a soft ink-tinted shadow.
- **Selected frame** (`box-shadow: inset 0 0 0 2px var(--edge)`): An amber outline inside the film.
- **Pressed choice** (`box-shadow: inset 0 0 0 1px var(--grease)`): Doubles the grease border.
- **Running halo** (`box-shadow: 0 0 0 3px oklch(82% .13 78 / .3)`): Breathes out to 6px around the running dot.
- **Film mark halo** (`text-shadow: 0 0 2px #000, 0 0 2px #000`): Keeps edge numbers legible over a bright frame.

### Named Rules
**The One Lifted Sheet Rule.** Only the loupe casts a shadow into the page. Everything else is flat at rest or shows state through inset lines.

## Shapes

Corners are nearly square (3px on controls, notes, and the loupe; 2px on the file-name line), and film has none at all because it is cut. The primary button alone gets an uneven hand-cut corner set. Full round appears in status dots, in map points, and in the loose ellipse around a file-name piece marked to hide.

The sprocket row is the system's recurring silhouette: paper-colored holes repeated along the top and bottom of film through a repeating gradient, at 17px intervals on frames and VM headers and 12px on small frames. Grease and wax marks are SVG strokes with round caps and non-scaling width, drawn with slightly overlapping, imperfect paths, and they overhang the frame by a few pixels.

## Components

### Buttons
Plain darkroom controls, with one red exception.
- **Shape:** Almost square (3px), 36px tall, 44px under 720px.
- **Default:** Transparent with a Pale Ink border and ink text.
- **Primary:** Grease fill, white 600 text, hand-cut corners. It marks the main action of a view, such as Test a dataset.
- **Hover / Focus:** Default fills with Shaded Paper and darkens its border; primary darkens to Pressed Grease. Transitions run .15s. Focus is a 2px grease outline, offset 2px.
- **Disabled:** 50% opacity with a not-allowed cursor.
- **Danger:** Transparent with Report Red border and text, washing Report Wash on hover.
- **Link:** Borderless underlined text for minor actions.

### Choices
- **Style:** A white sheet with a hairline border, a bold line and a smaller soft-ink hint, arranged in auto-fitting grids at least 200px wide.
- **State:** Hover darkens the border to ink. Selected takes a doubled grease border. Disabled drops to 55% opacity.

### Containers
- **Loupe:** A 380px white sheet with a hairline border, 16px padding, 3px corners, and the Loupe lift. It sticks to the viewport and scrolls internally.
- **Note:** Shaded Paper, 3px corners, no border. Neutral information.
- **Error box:** Report Wash with a Report Red border and text.
- **Details:** Evidence folds behind a bold summary line under a hairline, underlining on hover.

### Inputs / Fields
- **Style:** White, Pale Ink 1px border, 3px corners, 36px tall, with a bold label above. Text areas switch to mono at .85rem.
- **Focus:** Border and a flush 2px outline both turn grease.
- **Checkbox:** Native, with a grease accent color.

### Navigation
- **Style:** A slim header with a hairline below. Links are body text in Soft Ink with a transparent 2px underline.
- **States:** Hover turns ink; the current page is ink with an ink underline.
- **Mobile:** Spacing tightens and all items stay on the single row.

### Status Line
- **Style:** An 8px dot with a small sentence in plain words.
- **State:** Running has an amber breathing dot; failed and weak are Report Red and bold; partial is a half-filled ink dot in bold ink; finished is a solid ink dot; stopped is a hollow dot.

### Film Frame
The signature component: a pixelated 4:3 illustration inside film with sprocket rows, an optional amber frame number, a caption below with the name and scores.
- **Undeveloped:** Dark, low-contrast, warm filtered image while waiting.
- **Under the safelight:** A slightly brighter filtered image that pulses over 2.4s while attacks run.
- **Developing:** The filter clears over 1.8s on the long ease when its attacks finish.
- **Failed:** Grayscale and dim, crossed out in grease.
- **Selected:** Amber inner outline. **Hover:** 1px amber outline on the image.
- **Pick:** A 3.5px hand-drawn grease circle overhanging the frame, plus a hand-drawn word above its caption, which appears only on the pick.
- **Alternate:** A 2.5px wax stroke along the bottom edge, plus a Wax Ink tag.

### Stage Bar
A cut of film divided into the pipeline's stages. Later stages are Film Light Grey, completed ones Film Light, the active one solid Edge Amber with film text, the halted one solid Report Red with white text, and skipped ones struck through.

### Score Scale
A 22px measure: a hairline, a Pale Ink span between the two reference ticks, a translucent grease interval, and a 4px grease tick for this score. It is decorative to assistive technology, with a legend and sentence beside it carrying the numbers in text.

### Film Line
Terminal output and file names on film. Logs are mono on Film Black with wrapping and a 420px scroll limit. On the file-name line each piece is a dashed-underlined button: kept pieces fill Film Bright, hidden ones get a loose grease ellipse, and recording or scene pieces fill wax.

## Do's and Don'ts

### Do:
- **Do** let film carry the machine's record: frames, progress, logs, file names and VM identity.
- **Do** keep frames touching horizontally inside a strip, with paper showing only between cut rows.
- **Do** tie frame appearance to real progress: undeveloped while waiting, safelight while attacking, developed when finished.
- **Do** caption every drawn frame or strip as a drawing, not the user's footage, for as long as the harness has no real previews.
- **Do** keep a plain sentence beside every score mark, so nothing depends on reading the scale alone.
- **Do** make every animation honor reduced motion, as the global rule does.

### Don't:
- **Don't** build metric cards or tiles of large numbers; scores are captions and scale marks.
- **Don't** add a success green to interface chrome.
- **Don't** add uppercase tracked labels or kickers above headings. The drawn pick word is a mark for the pick alone, not a pattern for other headings.
- **Don't** add shadows to surfaces other than the loupe, or hard offset shadows anywhere.
- **Don't** spend grease on decoration; every appearance marks a decision, an action, this frame's score, or current focus.
- **Don't** round frames or film; they are cut stock.
