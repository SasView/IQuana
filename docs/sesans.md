# SESANS and SEMSANS

Spin-echo SANS (SESANS) measures the depolarisation of a neutron beam against
the **spin-echo length** ξ, a real-space distance from tens of nanometres to
several micrometres. IQuana fits it with the same I(Q) models as SANS data:
sasmodels turns each model into the SESANS quantity through a Hankel
transform. Reduced **SEMSANS** data (spin-echo modulated SANS) is the same
measurement made another way and is analysed the same way.

```python
from iquana import IQuana

fitter = IQuana()
fitter.load_data('sphere2micron.ses')
fitter.set_model('sphere')
fitter.set_param('radius', value=9000, min=2000, max=30000, vary=True)
fitter.set_param('scale',  value=0.01, min=1e-4, max=1,     vary=True)
fitter.set_param('sld', value=1.0)
fitter.set_param('sld_solvent', value=0.0)
fitter.fit()
fitter.plot_results()
```

```text
✓ Loaded data from sphere2micron.ses
  Spin-echo length: 391.56 to 46099 Å
  Data points: 40
  Error column: yes
  Transform: wavelength 2.11 Å, acceptance 0.0168 rad
  Weighting mode: dI (the dataset's own dI column)
...
χ²/dof              1.05225
radius       10.38(11)e3  fitted
scale        0.1331(13)   fitted
```

Nothing in this script is SESANS-specific: the kind comes from the data, and
`fitter.is_sesans` says which kind is loaded. The fit recovers the 2 µm
polystyrene spheres of the file. `sphere2micron.ses` and `sphere_isis.ses` ship
with sasdata, in `sasdata/example_data/sesans_data/`.

## What is fitted

With P/P₀ the polarisation with and without the sample, t the sample
thickness and λ the wavelength:

```text
P/P₀ = exp( Σt · [G(ξ) − 1] )

ln(P/P₀) / (t·λ²) = G̃(ξ) − G̃(0)        [Å⁻² cm⁻¹]

G̃(ξ) = (1/2π) ∫ I(q) J₀(qξ) q dq,   G(ξ) = G̃(ξ)/G̃(0),   Σ = λ²·G̃(0)
```

The left-hand side of the second line is the `Depolarisation` column of a
`.ses` file and the quantity IQuana fits. It starts at zero and falls to a
negative plateau, which a dilute system reaches at its largest size (the
diameter, for spheres). The exponential accounts for multiple scattering,
**provided the instrument collects nearly all the scattered neutrons** (see
[The acceptance angle](#the-acceptance-angle)).

For SEMSANS the ratio of the fringe visibilities with and without the sample,
Vs/V₀, obeys the same law (Li et al., J. Appl. Cryst. 49 (2016) 55, Eq. 7) and
takes the place of P/P₀.

Models, polydispersity, structure factors, composite models and links work as
for SANS data: they all act on I(Q) before the transform.

## Data in

Two routes:

- **A `.ses` file**, read with `load_data()` (or `data_ops.load()`). sasdata's
  reader handles the format.
- **Arrays**, through `data_ops.sesans()`: reduced SEMSANS data, or any table
  read with `np.loadtxt`.

```python
import numpy as np
from iquana import IQuana, data_ops

xi, ratio, ratio_err = np.loadtxt('semsans_reduced.txt', unpack=True)

data = data_ops.sesans(
    xi,                    # spin-echo length, Å
    polarisation=ratio,    # P/P0, or Vs/V0 for SEMSANS
    error=ratio_err,       # one standard deviation of the ratio
    wavelength=5.0,        # Å, one number or one per point
    thickness=0.2,         # cm
    acceptance=0.025,      # rad, see below
)
fitter = IQuana()
fitter.set_data(data)
```

With `polarisation=` the ratio is turned into ln(P/P₀)/(tλ²) and its error
into σ_P/(P·tλ²), which needs the thickness. A ratio above one is accepted
(it occurs in measured data); a ratio at or below zero is refused. Data that
is already normalised goes in as `depolarisation=`, with its own error and
without `thickness=`. Errors are treated as independent between points; a
shared reference measurement correlates them, which this ignores.

Every SESANS dataset is brought to one form on entry, by both routes: spin-echo
lengths and wavelengths in Å, the acceptance in radians, the depolarisation in
Å⁻² cm⁻¹, one wavelength per point. An unknown unit is refused and named.
Entry also refuses a missing wavelength or acceptance and spin-echo lengths
that are not positive and strictly increasing.

!!! warning "Row order in `.ses` files"
    sasdata sorts the spin-echo lengths of a `.ses` file but not the
    wavelengths that belong to them, so a file whose rows run backwards would
    load with every point carrying another point's wavelength. `load_data()`
    refuses such a file; sort its rows by `SpinEchoLength` first.

## The fit range

`set_q_range()` is the range method for SESANS data too. Its limits are
spin-echo lengths in Å, and the messages say so:

```python
fitter.set_q_range(5000, 30000)
```

```text
✓ Spin-echo length range for fitting: 5000 to 30000 Å
  Points in fit: 22 of 40
```

The range selects which points are fitted; the transform is still computed on
the whole grid, so the curve at a kept point does not depend on the range. The
same holds for masked and NaN points. (sasmodels itself ignores the range of
SESANS data; IQuana applies it.)

## Settings that do not apply

| Setting | For SESANS data |
|---|---|
| `set_resolution()` | Refused for any mode but the default `'data'`: the transform uses the wavelength and acceptance carried by the data. Loading SESANS data into a fitter set to another mode is refused too. |
| `set_weighting()` | `'dI'` only: SESANS data is weighted by its own errors. Without errors, the bumps engine and `fit_bayesian()` cannot fit; `fit(engine='lmfit')` fits unweighted with a warning. |
| `background` | sasmodels sets it to zero for SESANS data, so a fit with `background` free is refused. |
| `calculate(q=...)`, `compare(q=...)` | Refused: each point needs a wavelength, so the model is evaluated on the data's own spin-echo lengths. |
| `MultiFitter`, `data_ops` arithmetic, P(r) inversion, `modelfree` | Not supported for SESANS data in this version. |

Plots use linear axes, because the depolarisation is negative:
`plot_results()`, `plot_model()`, `compare()` and `plot_posterior_predictive()`
take `log_scale=None` by default, which means log for I(Q) and linear for
SESANS. `save_results()` writes the columns `spin_echo_length`,
`depolarisation`, `error`, `fit` and `residual`.

## The acceptance angle

The acceptance is the largest scattering angle that still reaches the
measurement; scattering beyond it is lost.

- **SESANS**: it is the `Theta_zmax` field of the `.ses` file, the angular
  acceptance in the encoding direction, set by the analyser and detector
  aperture as seen from the sample (Mantid's `SaveSESANS` defines the field
  so). The example files carry 0.0168 rad and 0.09 rad.
- **SEMSANS**: for the central detector pixels, Kusmin, Bouwman, van Well and
  Pappas (Nucl. Instrum. Methods A 856 (2017) 119, Eq. 21) give

  ```text
  acceptance = S/d₁ + (S_C + S)/d_C
  ```

  with 2S the sample aperture, 2S_C the collimation aperture, d₁ the
  sample-to-detector distance and d_C the distance between the two apertures:
  half the sample size seen from the detector, plus the largest beam
  divergence. Their worked example (d₁ = d_C = 8 m, S_C = 1.5 cm, S = 0.5 cm)
  gives 3.1 mrad. Use only the central region of the detector; summing over
  the detector height widens the range in that direction.

**When the acceptance is narrow, the law itself changes.** The measured ratio
then follows a more general expression (Kusmin et al., Appendix C; Li et al.,
Sci. Rep. 2019), not the exponential above, and the lost scattering also
affects the measured transmission. IQuana fits the exponential, so the data
must come from a setup with a generous acceptance.

sasmodels' own handling of the angle is approximate: it compares an angle with
a Q value, uses the longest wavelength for every point, and leaves the
G̃(0) it subtracts uncut. In the example files this has no visible effect, but
a model with scattering at high Q, or a narrower acceptance, would be biased.

## The integration limit and its warning

sasmodels takes the largest q of the transform from the first step between
spin-echo lengths, so structure much smaller than that step is cut off. After
every SESANS fit IQuana computes the curve once more with a fourfold higher
limit and warns, without stopping, when that moves the curve by more than 1 %
of its depth or χ² by more than 1:

```text
UserWarning: The SESANS curve depends on the integration limit: a finer first
spin-echo step changes it by 4.3 % of its depth (χ² by 12.1). Structure much
smaller than the first step (1172 Å) is not resolved. ...
```

On the grid of `sphere2micron.ses` (first step 1172 Å) a sphere of 10 000 Å is
computed to 0.05 %, one of 2 000 Å to 0.9 % and one of 1 000 Å to 4.5 %, which
warns. **Silence is not proof**: the check compares two numerical limits, so
scattering beyond both goes unnoticed, and it says nothing about a narrow
acceptance. A warning in a series is recorded on its row.

## Series, saving and reports

`fit_series()` fits SESANS datasets like any others. A series holds one kind
of data: the template's when it holds data, otherwise the first dataset's that
loads; a dataset of the other kind becomes a failed row, and `update()`
refuses a repair of the other kind. `series.is_sesans` says which kind a series
holds. The template's spin-echo length range is replayed on every row.

`save_analysis()` writes SESANS analyses as schema version 4, with
`data.measurement = 'sesans'` and, for an in-memory dataset, its wavelengths
and acceptance. An analysis of I(Q) data stays at version 3, so older IQuana
versions keep reading it; they refuse a SESANS one, which they would fit
without its range. A SESANS analysis written before version 4 loads without its
fit result, because that fit ignored the range. A changed wavelength or
acceptance invalidates a saved result, as changed data does.

## Measuring well

From Andersson et al. (J. Appl. Cryst. 41 (2008) 868): measure out to the
saturation level, which lies at the largest size in the sample, and choose
the sample so that the level lies between 0.3 and 0.9 in polarisation.

## References

- Andersson, van Heijkamp, de Schepper, Bouwman, J. Appl. Cryst. 41 (2008)
  868. The polarisation law and the sphere. Its Eq. 61 (and Eq. 25 of Bakker et
  al.) misprints a square: the factor before the logarithm is
  1 − (ξ/4R)², not (1 − ξ/4R)².
- Bakker, Washington, Parnell, van Well, Pappas, Bouwman, J. Neutron Res. 22
  (2020) 57. The numerical Hankel transform in SasView.
- Kusmin, Bouwman, van Well, Pappas, Nucl. Instrum. Methods A 856 (2017) 119.
  The SEMSANS acceptance.
- Li et al., J. Appl. Cryst. 49 (2016) 55. SEMSANS with Wollaston prisms.
- Li et al., Sci. Rep. (2019). Data correction of intensity-modulated SANS.
