"""Generate VNA diagrams following diagram-design's primitives (default skin)."""
from pathlib import Path
import sys

PAPER, INK, MUTED, SOFT, ACCENT = '#f5f5f5', '#2d3142', '#4f5d75', '#5a6580', '#bf4520'
ACCENT_TINT, RULE = 'rgba(191,69,32,0.08)', 'rgba(45,49,66,0.10)'
SANS, MONO, SERIF = "'Geist', sans-serif", "'Geist Mono', monospace", "'Instrument Serif', serif"
OUT = Path(sys.argv[1])


def text(x, y, s, size=9, fill=MUTED, family=MONO, weight=None, anchor='middle', extra=''):
    w = f' font-weight="{weight}"' if weight else ''
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="{family}"{w} '
            f'text-anchor="{anchor}"{extra}>{s}</text>')


def lines(cx, top, name, subs):
    out = [text(cx, top, name, 12, INK, SANS, 600)]
    out += [text(cx, top + 14 + 13 * i, s) for i, s in enumerate(subs)]
    return out


def box(x, y, w, h, name, subs, tag=None, focal=False, dashed=False):
    stroke = ACCENT if focal else INK
    fill = ACCENT_TINT if focal else '#ffffff'
    dash = ' stroke-dasharray="5,4"' if dashed else ''
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>']
    if tag:
        tw = max(28, 6 * len(tag) + 8)
        out += [f'<rect x="{x + 8}" y="{y + 6}" width="{tw}" height="12" rx="2" fill="transparent" stroke="{stroke}" stroke-opacity="0.4" stroke-width="0.8"/>',
                text(x + 8 + tw / 2, y + 15, tag, 7, stroke, MONO, extra=' letter-spacing="0.08em" fill-opacity="0.8"')]
    n = len(subs)
    top = y + h / 2 + 4 - (14 + 13 * (n - 1)) / 2 + (6 if tag else 0)
    return out + lines(x + w / 2, top, name, subs)


def oval(x, y, w, h, name, subs, focal=False):
    stroke = ACCENT if focal else 'rgba(45,49,66,0.30)'
    fill = ACCENT_TINT if focal else 'rgba(45,49,66,0.03)'
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{PAPER}"/>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="1"/>']
    n = len(subs)
    top = y + h / 2 + 4 - (13 * n) / 2
    return out + lines(x + w / 2, top, name, subs)


def diamond(cx, cy, hw, hh, l1, l2, focal=False):
    stroke = ACCENT if focal else INK
    fill = ACCENT_TINT if focal else '#ffffff'
    pts = f'{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}'
    return [f'<polygon points="{pts}" fill="{PAPER}"/>',
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
            text(cx, cy - 2, l1, 12, INK, SANS, 600), text(cx, cy + 12, l2, 12, INK, SANS, 600)]


def label_w(s):
    return int(len(s) * 8 * 0.62 + 8 + 3) // 4 * 4 + 4


def arrow(d, accent=False, dashed=False):
    c, m = (ACCENT, 'arrow-accent') if accent else (MUTED, 'arrow')
    dash = ' stroke-dasharray="5,4"' if dashed else ''
    sw = 1.4 if accent else 1.2
    return f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{sw}"{dash} marker-end="url(#{m})"/>'


def hlabel(mx, y, s, accent=False):
    """Label above a horizontal segment at y: mask bottom 6px above the stroke."""
    w = label_w(s)
    return [f'<rect x="{mx - w / 2}" y="{y - 20}" width="{w}" height="12" rx="2" fill="{PAPER}"/>',
            text(mx, y - 11, s, 8, ACCENT if accent else SOFT, MONO, extra=' letter-spacing="0.06em"')]


def vlabel(x, my, s, side=1):
    """Label beside a vertical segment at x, 8px gap."""
    w = label_w(s)
    left = x + 8 if side > 0 else x - 8 - w
    return [f'<rect x="{left}" y="{my - 6}" width="{w}" height="12" rx="2" fill="{PAPER}"/>',
            text(left + w / 2, my + 3, s, 8, SOFT, MONO, extra=' letter-spacing="0.06em"')]


def elbow(x1, y1, x2, y2, first='h', r=8):
    """Orthogonal one-bend path with an r=8 quarter arc."""
    if first == 'h':
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        return (f'M{x1},{y1} H{x2 - sx * r} Q{x2},{y1} {x2},{y1 + sy * r} V{y2}')
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    return (f'M{x1},{y1} V{y2 - sy * r} Q{x1},{y2} {x1 + sx * r},{y2} H{x2}')


def legend(y, w, items):
    out = [f'<line x1="40" y1="{y}" x2="{w - 40}" y2="{y}" stroke="{RULE}" stroke-width="0.8"/>',
           text(40, y + 16, 'LEGEND', 8, MUTED, MONO, anchor='start', extra=' letter-spacing="0.18em"')]
    x = 40
    for kind, label in items:
        iy = y + 32
        if kind == 'oval':
            out.append(f'<rect x="{x}" y="{iy}" width="24" height="12" rx="6" fill="rgba(45,49,66,0.03)" stroke="rgba(45,49,66,0.30)"/>')
        elif kind == 'box':
            out.append(f'<rect x="{x}" y="{iy}" width="24" height="12" rx="2" fill="#ffffff" stroke="{INK}"/>')
        elif kind == 'focal':
            out.append(f'<rect x="{x}" y="{iy}" width="24" height="12" rx="2" fill="{ACCENT_TINT}" stroke="{ACCENT}"/>')
        elif kind == 'dashedbox':
            out.append(f'<rect x="{x}" y="{iy}" width="24" height="12" rx="2" fill="#ffffff" stroke="{INK}" stroke-dasharray="4,3"/>')
        elif kind == 'diamond':
            out.append(f'<polygon points="{x},{iy + 6} {x + 12},{iy} {x + 24},{iy + 6} {x + 12},{iy + 12}" fill="#ffffff" stroke="{INK}"/>')
        elif kind == 'accent':
            out.append(f'<line x1="{x}" y1="{iy + 6}" x2="{x + 24}" y2="{iy + 6}" stroke="{ACCENT}" stroke-width="1.4" marker-end="url(#arrow-accent)"/>')
        elif kind == 'arrow':
            out.append(f'<line x1="{x}" y1="{iy + 6}" x2="{x + 24}" y2="{iy + 6}" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>')
        elif kind == 'dashed':
            out.append(f'<line x1="{x}" y1="{iy + 6}" x2="{x + 24}" y2="{iy + 6}" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5,4" marker-end="url(#arrow)"/>')
        out.append(text(x + 32, iy + 10, label, 8.5, MUTED, SANS, anchor='start'))
        x += 32 + int(len(label) * 5.2) + 28
    return out


def callout(x, y, rows):
    return [text(x, y + 20 * i, r, 14, INK if i else MUTED, SERIF, anchor='start', extra=' font-style="italic"')
            for i, r in enumerate(rows)]


def page(slug, eyebrow, title, desc, w, h, body):
    head = Path(sys.argv[2]).read_text()  # template.html from the skill
    head = head.split('<body>')[0].replace('<title>Diagram</title>', f'<title>{title}</title>')
    head = head.replace('svg { width: 100%; min-width: 960px;', f'svg {{ width: 100%; min-width: {w}px;')
    head = head.replace('.frame { max-width: 1200px;', f'.frame {{ max-width: {w + 40}px;')
    markers = ''.join(
        f'<marker id="{i}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{c}"/></marker>'
        for i, c in (('arrow', MUTED), ('arrow-accent', ACCENT), ('arrow-link', '#2e5aa8')))
    svg = (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-labelledby="{slug}-title {slug}-desc">\n<title id="{slug}-title">{title}</title>\n'
           f'<desc id="{slug}-desc">{desc}</desc>\n<defs>{markers}</defs>\n'
           f'<rect width="100%" height="100%" fill="{PAPER}"/>\n' + '\n'.join(body) + '\n</svg>')
    html = (head + f'<body>\n  <div class="frame">\n    <p class="eyebrow">{eyebrow}</p>\n    <h1>{title}</h1>\n'
            f'    <div class="diagram-container">\n{svg}\n    </div>\n  </div>\n</body>\n</html>\n')
    (OUT / f'{slug}.html').write_text(html)


# ---------------------------------------------------------------- 1. pipeline
b = []
Y = 160
b += [arrow(f'M200,{Y} H280'), arrow(f'M440,{Y} H520'), arrow(f'M680,{Y} H760'),
      arrow(f'M920,{Y} H1000'), arrow(f'M1160,{Y} H1240', accent=True),
      arrow('M840,208 V280'), arrow('M1080,208 V280')]
b += hlabel(480, Y, 'μ + NULLS') + hlabel(720, Y, 'PER SETTING') + hlabel(960, Y, 'YES') + hlabel(1200, Y, 'PASS', True)
b += vlabel(840, 244, 'NO') + vlabel(1080, 244, 'FAIL')
b += oval(40, 124, 160, 72, 'Phase 0', ['ResNet-18 · CIFAR-100', 'ρ ≈ 0.03 · no-go'])
b += box(280, 124, 160, 72, 'Stage A', ['nulls N1/N2 · μ sweep', 'selection · joint K_E'], tag='DIAG')
b += box(520, 124, 160, 72, 'Stage B', ['B1–B5 on upstream code', 'measure only · no VNA'], tag='MEAS')
b += diamond(840, 160, 80, 48, 'Baseline', 'reproduced?')
b += diamond(1080, 160, 80, 48, 'Gate G', 'passes?', focal=True)
b += oval(1240, 124, 160, 72, 'Stage C pilot', ['VNA vs NGD control', 'random-A · tiny_U'], focal=True)
b += oval(760, 280, 160, 56, 'Report', ['fix before measuring'])
b += oval(1000, 280, 160, 56, 'Stop', ['rethink direction'])
b += callout(40, 384, ['Gate G — per setting, r = 16, converged CG only, thresholds fixed before data:',
                       'ρ_joint ≥ 0.3 on ≥ 50% of points · N1 significant on ≥ 50% · V(tiny_V) > V(random) on ≥ 70% · train acc < 99% (n/a for B3, B5)'])
b += legend(428, 1440, [('oval', 'Start / end'), ('box', 'Stage'), ('diamond', 'Decision'),
                        ('focal', 'Focal'), ('accent', 'Go path'), ('arrow', 'Flow')])
page('vna-pipeline', 'Flowchart · VNA Stage A/B', 'VNA research pipeline',
     'Flowchart from the Phase 0 negative result through Stage A diagnostics and Stage B screening to the baseline check and gate G, which decides between a Stage C pilot and stopping.',
     1440, 488, b)

# ---------------------------------------------------------------- 2. algorithm
b = []
R1, R2 = 160, 360
b += [arrow(f'M200,{R1} H280'), arrow(f'M440,{R1} H520'), arrow(f'M680,{R1} H760'),
      arrow(f'M920,{R1} H1000'), arrow(f'M1080,196 V324'),
      arrow(f'M1000,{R2} H920'), arrow(f'M760,{R2} H680', accent=True), arrow(f'M440,{R2} H520', dashed=True)]
b += hlabel(240, R1, 'LOGITS z') + hlabel(480, R1, 'H') + hlabel(720, R1, 'FILTERED z') + hlabel(960, R1, 'A · B = 0')
b += vlabel(1080, 260, 'K_E') + hlabel(960, R2, 'IF ρ > τ') + hlabel(720, R2, 'λd', True) + hlabel(480, R2, 'CONTROL')
b += box(40, 124, 160, 72, 'Batch + loss', ['g = ∂L/∂z, autograd', 'upstream loss object'], tag='IN')
b += box(280, 124, 160, 72, 'GGN metric H', ['true Fisher only', 'CE · MSE · KD · iCaRL'])
b += box(520, 124, 160, 72, 'Damped solve', ['(F + μI) u = Jᵀg · CG', 'μ = μ_rel·tr(F)/n_out'])
b += box(760, 124, 160, 72, 'Select A · tiny_V', ['TINY on cotangent z', 'normalised · B = 0'], focal=True)
b += box(1000, 124, 160, 72, 'Virtual kernel K_E', ['J_E at B = 0, hooks', 'never installed'])
b += box(1000, 324, 160, 72, 'Signal V, ρ', ['V = ‖J_Eᵀ z‖²', 'ρ = V / ‖Jᵀg‖²'], tag='OUT')
b += box(760, 324, 160, 72, 'Correction d', ['(F+μ)⁻¹C_OE C_EO(F+μ)⁻¹Jᵀg', '= Jᵀ R K_E R K g'], focal=True)
b += box(280, 324, 160, 72, 'Controls', ['NGD Jᵀ R K g, norm-matched', 'random_frob · tiny_U'], dashed=True)
b += box(520, 324, 160, 72, 'Optimizer step', ['Stage C only: g + λd', 'Stage A/B: measure only'])
b += callout(40, 440, ['Nested-projection lemma:',
                       'under exact Gauss–Newton the virtual neurons add nothing — VNA targets first-order (SGD) dynamics.'])
b += legend(476, 1200, [('box', 'Computation'), ('focal', 'Focal step'), ('dashedbox', 'Control arm'),
                        ('accent', 'Applied correction'), ('arrow', 'Data'), ('dashed', 'Substitute')])
page('vna-algorithm', 'Data flow · VNA correction', 'VNA correction algorithm',
     'Data flow of one VNA step: upstream residual and GGN metric feed a damped solve, tiny_V selects virtual neurons whose kernel gives the transferred gain and the correction applied only in Stage C.',
     1200, 536, b)

# ---------------------------------------------------------------- 3. measurement
b = []
b += [arrow(f'M200,{R1} H280'), arrow(f'M440,{R1} H520'), arrow(f'M680,{R1} H760'),
      arrow(f'M920,{R1} H1000'), arrow(f'M1160,{R1} H1240', accent=True),
      arrow(elbow(680, R2, 800, 196, 'h')), arrow(elbow(880, 196, 1000, R2, 'v')),
      arrow(f'M1080,324 V196'), arrow(f'M360,196 V324', dashed=True)]
b += hlabel(240, R1, 'LEARN STEP') + hlabel(480, R1, 'SEEN BATCH') + hlabel(720, R1, 'STAT/CORR')
b += hlabel(960, R1, 'ROWS') + hlabel(1200, R1, 'PER SETTING', True)
b += hlabel(736, R2, 'SITES') + hlabel(944, R2, 'NULL RESID') + vlabel(1080, 260, 'P-VALUES') + vlabel(360, 260, 'STATE')
b += box(40, 124, 160, 72, 'Upstream loop', ['LoP · PyCIL · RepDistiller', 'unchanged code, pinned'], tag='UPSTREAM')
b += box(280, 124, 160, 72, 'Thin hook', ['learn / scheduler / LR wrap', 'restores params, RNG'], focal=True)
b += box(520, 124, 160, 72, 'Measurement buffer', ['last m seen samples', 'task-scoped, no test data'])
b += box(760, 124, 160, 72, 'VNA core', ['ρ, ρ_joint, U, V', 'tiny_V · random · tiny_U'])
b += box(1000, 124, 160, 72, 'measurements.csv', ['stage1d-v1', 'μ ∈ {1e-3, 1e-2}'], tag='OUT')
b += box(1240, 124, 160, 72, 'Gate G', ['4 pre-registered tests', 'official μ from Stage A'], focal=True)
b += box(520, 324, 160, 72, 'HookSite', ['first → act → second', 'virtual A, B = 0'])
b += box(1000, 324, 160, 72, 'Nulls N1 / N2', ['8 × permute · gaussian', 'p = (1+#)/(1+n)'])
b += box(280, 324, 160, 72, 'Checkpoint', ['bit-identical resume', 'exit 75 before 11.5 h'], dashed=True)
b += callout(40, 440, ['T-align guarantee:',
                       'hook on vs off gives bit-identical parameters and optimizer state — the baseline is the upstream code.'])
b += legend(476, 1440, [('box', 'Component'), ('focal', 'Focal'), ('dashedbox', 'Persistence'),
                        ('accent', 'Decision input'), ('arrow', 'Data'), ('dashed', 'State save')])
page('vna-measurement', 'Data flow · Stage B measurement', 'VNA measurement and gate',
     'Data flow of Stage B: a thin hook around the unchanged upstream loop feeds seen batches and declared sites into the VNA core and residual-matched nulls, whose rows decide gate G, with checkpoints for resume.',
     1440, 536, b)
print('ok')
