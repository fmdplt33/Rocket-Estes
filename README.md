# RocketOpt

Automated design, analysis, optimisation and CAD export of high-performance
model rockets for Estes motors.

RocketOpt sizes a rocket from first principles rather than from a template. You
give it a motor and a set of constraints; it searches the design space and
returns a near-optimal airframe, a full flight simulation, a structural
assessment, manufacturing drawings and a parametric Fusion 360 script.

Every physical quantity is computed in SI from published aerospace relations,
and every equation carries a citation in its docstring.

---

## Quick start

```bash
pip install -e ".[ui]"
```

```bash
rocketopt ui
```

Or from the command line:

```bash
rocketopt motors
```

```bash
rocketopt design C6-5 --for altitude --export out/
```

---

## What it does

| Layer | What it computes |
|---|---|
| `propulsion` | Estes motor database (A8 to F15), thrust curves, RASP `.eng` import |
| `geometry` | Nine nose-cone families, airframe components, mass and inertia |
| `aerodynamics` | Atmosphere, Barrowman stability, component drag build-up, damping |
| `flight` | Six-degree-of-freedom trajectory with rail, wind, recovery, descent |
| `structures` | Fin flutter, tube buckling, bending, joints, landing loads |
| `optimisation` | NSGA-II multi-objective search with constrained domination |
| `cfd` | Axisymmetric potential flow and boundary-layer separation |
| `cad` | Printable STL solids, SVG templates, DXF, STEP, parametric Fusion 360 script |
| `reports` | Engineering report, manufacturing guide, bill of materials |
| `ui` | Desktop application with live analysis and flight animation |

---

## The interface

Edit on the left, see the consequence immediately in the centre and right.
There is no "calculate" button — every change re-analyses in about 60 ms.

- **Rocket** — a to-scale side elevation. Drag the circled points on the fin to
  reshape it; the numeric fields follow, and vice versa.
- **Animation** — the flight replayed, with a speed-against-time curve that
  draws itself out in step with the rocket. Events (rail exit, max
  acceleration, max Q, top speed, booster cutoff, apogee, ejection, landing)
  are read from the trajectory, not assumed.
- **Trajectory**, **Aerodynamics** — altitude/speed/acceleration/dynamic
  pressure, and the drag breakdown by mechanism.
- **Convergence**, **Trade-off** — optimiser progress and the Pareto front.

Optimisation runs on a worker thread with a real generation counter. Cancel
stops it at the next generation boundary and keeps the best design found.

---

## Engineering approach

### Where the numbers come from

Every correlation is cited in the docstring of the function that uses it.
The main sources are Barrowman (1967) for stability, Hoerner (1965) for drag,
NACA TN 4197 for fin flutter, NASA SP-8007 for shell buckling, USSA-1976 for
the atmosphere, and Deb et al. (2002) for NSGA-II.

### Thrust curve provenance

Built-in thrust curves are **synthesised**, not measured. The generator solves
a black-powder burn model so that the certified total impulse, peak thrust and
burn time are reproduced exactly — integrated results (apogee, burnout
velocity) are therefore reliable, but instantaneous peak-load detail is a
model.

For measured data, drop a ThrustCurve.org `.eng` file into
`rocketopt/assets/motors/`. It is picked up automatically and takes precedence,
and the provenance warning disappears from reports.

### Known limits

- The drag correlations are subsonic and validated to about Mach 0.8. Above
  that the wave-drag term is an approximation and says so in
  `DragBreakdown.validity_warning`.
- The potential-flow solver is slender-body theory. It is a genuine numerical
  solution of Laplace's equation for the body's displacement effect, but it is
  not Navier-Stokes and loses accuracy at a blunt tip or an abrupt shoulder.
- Specific impulse derived from Estes' published propellant masses ranges from
  61 s to 96 s. That spread is in the published data, not the model — Estes
  does not define "propellant weight" consistently. Nothing computed here
  depends on it.
- Altitudes are geopotential, not geometric. Below 1 km the difference is under
  0.2 m.
- **E and F motors need a longer rail than the standard 0.91 m rod.** An F15 is
  a long, soft burn and leaves that rod at about 12.5 m/s, below the 15 m/s the
  fins need for authority; roughly 1.3 m is required. This is a property of the
  motor, not of the software — set `--rail-length` accordingly.

### Validation

Against published reference data:

| Case | Result |
|---|---|
| USSA-1976 atmosphere, 0–11 km | within 1e-3 relative |
| Barrowman nose CP (cone, ogive, ellipse) | within 2e-3 |
| Von Kármán volume coefficient | 0.500 (analytic) |
| Certified motor impulse and peak thrust | within 1e-3 |
| Fin flutter thickness exponent | 2^1.5 exactly |
| Estes Alpha III on B6-4 / C6-5 | −6.4% / +4.4% |
| Exported STEP solid vs the mesh built alongside it | within 0.2% |
| NACA four-digit section area | 0.685 t·c (published) |

The A8-3 case reads about 19% below Estes' published 350 ft. Flyer-reported
altitudes for that combination cluster around 250–300 ft, which matches the
model; the published figure looks optimistic. It has not been tuned to match.

```bash
python -m pytest tests/ -q
```

158 tests, including regression tests for seven real bugs found during
development: a secant-ogive construction that produced `X_cp/L = -51428`; a
discontinuity at the laminar–turbulent transition that would have put a false
cliff in the optimiser's objective; a potential-flow tip singularity that
reported the nose tip as the suction peak; a duplicated vertex that made STEP
export fail in the OCC kernel; a tangent-ogive tip radius that came out 3e-14
from zero instead of zero, so that revolving it left a ring of slivers and an
open apex rather than a point; and a small optimiser budget returning no
feasible design at all.

Exports are checked for validity, not merely for existence: every STL is parsed
back off disk and must be watertight, manifold and outward-facing with the
volume it was built with, STEP files must carry an ISO-10303 header and
re-import as geometry matching the mesh from the same profile, the PDF must have
a valid header and multiple pages, and the Fusion 360 script must parse as
Python with parameters matching the design.

---

## Printable parts

`rocketopt design --export` writes an STL of every part, in millimetres, each
oriented with the face that should sit on the print bed at `z = 0`. They are
built as single revolves and lofts of closed profiles, so each is a closed
manifold by construction — no booleans, no stitching.

| Part | Geometry |
|---|---|
| `nose_cone.stl` | The analytic profile, shelled to the design's wall thickness, with an **integrated locating spigot** turned to the body tube's bore |
| `body_tube.stl` | The airframe at nominal dimensions: the datum the other parts are cut to |
| `motor_mount.stl` | A tube whose **outside diameter is the tube bore** and whose **bore is the motor case**, one motor casing long |
| `fin.stl` | Lofted through its true section — square, rounded, NACA four-digit aerofoil or double wedge — so the leading edge, trailing edge and thickness distribution are the ones the drag model was given |

Three interfaces decide whether the parts go together, and all three are derived
from the design rather than assumed: the spigot into the tube bore, the mount
into the same bore, and the motor into the mount. The nominal diameters are
exact matches, so each part that goes *inside* another carries a small print
clearance (0.3 mm on diameter for the airframe fits, 0.4 mm for the motor);
pass your own `PrintTolerances` for a press fit or a looser machine.

Nothing is written until it has been checked. Before export, the mating
diameters are measured off the meshes themselves and compared, the spigot is
checked for enough engagement to carry the ejection load, the fin is checked
against its specified planform, thickness and section area, and every part is
checked for watertightness, manifoldness and outward-facing normals. A failure
raises `AssemblyValidationError` listing every problem and writes nothing:

```python
from rocketopt.cad import validate_printable_assembly

print(validate_printable_assembly(rocket).report())
```

The self-centring motor mount fills the annulus between the motor and the
airframe, which is heavier than a paper motor tube on centring rings — about 8 g
against 1.4 g on an 18 mm mount. On a small airframe that is worth a quarter of
a calibre of static margin, so check the margin after switching to one, and pass
`build_rocket` an explicit `MotorMount` if you want the lighter ring-carried
construction.

---

## Optimisation

The design space is 19 mixed continuous, integer and categorical variables:
nose profile and fineness, body diameter and length, wall and fin stock
thickness, fin count, planform, section, material, fillet, surface finish and
ballast.

Constraints are **not** folded into the objective as weighted penalties.
Weighted penalties silently trade safety against performance, which is exactly
wrong for fin flutter where any violation is catastrophic. NSGA-II uses Deb's
constrained-domination rule instead: a feasible design always beats an
infeasible one, so a design that flutters can never win on altitude.

Constraints applied: static margin band, thrust-to-weight, rail exit velocity,
flutter and divergence margin, every structural margin, landing speed, and
maximum length, diameter and mass.

The initial population is seeded with a conventional baseline design that
scales with the motor and repairs its own static margin. Without it, a small
population in a nineteen-dimensional constrained space can contain no feasible
design at all and the run fails outright; with it the optimiser is guaranteed
to return something at least as good as a sound conventional rocket.

On the reference case the optimiser finds **652 m** on a C6-5 against **415 m**
for an Alpha III on the same motor.

---

## Performance

| Mode | Time | Apogee error |
|---|---|---|
| 6-DOF, 1 ms | 2.6 s | reference |
| 3-DOF, 5 ms, apogee cut-off | 200 ms | +0.006% |
| Preview, 20 ms | 55 ms | −0.010% |

The optimiser's inner loop uses the fast mode; the winning design is re-flown
in full 6-DOF before anything is reported.

---

## Installation

Core install needs no compiler — every dependency is a pure-Python or
wheel-only package.

```bash
pip install -e .
```

Extras:

```bash
pip install -e ".[ui]"
```

```bash
pip install -e ".[cad]"
```

```bash
pip install -e ".[dev]"
```

Without the `cad` extra, the printable STLs, SVG templates, DXF and the
Fusion 360 script are all still written; only STEP is skipped.

Requires Python 3.11 or newer.

---

## Licence

MIT.
