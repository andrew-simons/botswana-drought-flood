# Product

## Register

product

## Users

Three overlapping audiences, all within Botswana:
- **Citizens / farmers** in rural and peri-urban areas who need to understand drought risk in plain language and take timely action (reduce livestock, delay planting, contact authorities).
- **NGO field workers and district administrators** who monitor conditions across multiple districts and need clear status signals to trigger response protocols.
- **Government analysts** (Ministry of Agriculture, DWA) who need scientific credibility — correct indices, correct sources, traceable methodology.

Primary context: low-bandwidth connections common; users may access on mobile. Decision stakes are high (livelihoods, livestock, water).

## Product Purpose

A satellite-based monthly drought early-warning dashboard for Botswana. Surfaces three continuous risk indices (SPI-3, NDVI anomaly, soil-moisture anomaly) derived from CHIRPS, MODIS, SMAP, and ERA5-Land via GEE and a PyTorch ConvLSTM model. Lets users see current national and district conditions, drill into a pixel-level time series, and understand what action to take. Success = users make better, earlier decisions about water and planting.

## Brand Personality

Authoritative, accessible, clear. Trusted like a government alert system but readable by a first-time user. Not academic, not charity-appeal, not SaaS-dashboard.

## Anti-references

- **Generic SaaS dashboards**: avoid blue-purple gradients, hero-metric giant numbers, rounded-card grids, "powered by" bottom strips.
- **NGO / charity aesthetic**: avoid feel-good greens, donation-drive tone, stock-photo compassion visuals.
- **Academic research tools**: avoid dense tables with no hierarchy, grey-on-grey, text walls, publication-style layout.
- **Consumer weather apps**: avoid cartoonish icons, over-simplified single-number risk readouts, playful illustration.

## Design Principles

1. **Signal before noise** — the most urgent information (national risk, action guidance) is above the fold and impossible to miss. Supporting detail (time series, model source) lives beneath.
2. **Clarity over cleverness** — every label, axis, and status indicator should be self-explanatory to a farmer with a smartphone, not just a data scientist.
3. **Earned authority** — the visual system communicates scientific credibility through restraint, precision, and consistency — not through decoration.
4. **Actionable by default** — every risk state surfaces a concrete next step. The app does not just report; it guides.
5. **Colour carries meaning** — risk colours (severe, moderate, normal, wet) are used exclusively for risk. No decorative use of red or orange.

## Accessibility & Inclusion

WCAG AA minimum (4.5:1 body text contrast, 3:1 large text). Risk states must never rely on colour alone — always pair with text label and/or icon. Mobile-first layout (primary users may be on phones with variable signal).
