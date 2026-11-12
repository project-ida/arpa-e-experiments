---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="cdd5c8b3-5827-4054-8abd-b52af9e1a4c0" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/tutorials/Spotting-Outliers-Eljen-Background-Characterization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/tutorials/Spotting-Outliers-Eljen-Background-Characterization.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="e99c3707" -->
# Spotting Outliers from Neutron Detection Background

This tutorial examines individual one-minute neutron counts that are unusually high under a constant-rate Poisson background model. We estimate the model's mean from the selected background record and compare two ways of identifying minutes for closer inspection:

1. **A mean-plus-Z-standard-deviations rule.** This provides a familiar count threshold and shows why a Gaussian interpretation can be inaccurate when the mean count is small.
2. **Poisson upper-tail probabilities with a Bonferroni reference threshold.** These account for the discrete count distribution and show how the number of minutes examined affects the threshold for flagging a result.

For details of the dataset, see [Eljen Background Characterization](Eljen-Background-Characterization.ipynb). Here we use a constant-rate Poisson model to illustrate the outlier calculations.
<!-- #endregion -->

<!-- #region id="56b5e3f2" -->
# Step 0 - Loading the Data and Estimating the Background Mean

We load the neutron and gamma count exports, sum their one-second counts into one-minute bins, and select the background period. The analysis below uses the neutron counts.

The code reports the selected time range, the number of minute bins, and their mean count. Minutes with zero counts remain in the calculation.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2KviDUsiprMt" outputId="41ae36ad-1d58-4754-9f4a-10e1034440f2"
# Makes Libs accessible and runs notebook from same location regardless of whether colab or local
!pip install colocal --upgrade -q
import colocal
root, branch, cwd = colocal.setup()
```

```python id="cd0188ee"
# Libraries and helper functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import ipywidgets as widgets
from IPython.display import display

from IPython.display import Image
from IPython.display import Video
from IPython.display import HTML

# Use our custom helper functions
# - process_data
# - plot_panels
# - plot_panels_with_scatter
# - print_info
from libs.helpers import *
```

```python id="6a7ef520"
meta = {
    "descriptor" : "Eljen 2 inch 12-2024" # This will go into the title of all plots
}
```

```python id="1de73abf"
neutron_df = pd.read_csv(
    'https://nucleonics.mit.edu/data/csv-files/eljen/eljen-2inch-long-term/eljen-2inch-long-term-1-fullres.csv',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)

# The current panel's gamma export for this detector is graph 5.
gamma_df = pd.read_csv(
    'https://nucleonics.mit.edu/data/csv-files/eljen/eljen-2inch-long-term/eljen-2inch-long-term-5-fullres.csv',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)
```

```python id="abd0acea"
neutron_df_1_minute = neutron_df.resample('1min').sum()
gamma_df_1_minute = gamma_df.resample('1min').sum()

start_time = "2024-12-19 00:00:00"
end_time = "2025-01-23 23:59:00"

neutron_df_1_minute_background = neutron_df_1_minute.loc[start_time:end_time].copy()
gamma_df_1_minute_background = gamma_df_1_minute.loc[start_time:end_time].copy()

# Estimate the mean number of neutron counts in a one-minute bin.
lambda_ = neutron_df_1_minute_background["Counts"].mean()

print(f"Selected minute bins: {neutron_df_1_minute_background.index.min()} to "
      f"{neutron_df_1_minute_background.index.max()}")
print(f"Number of minute bins: {len(neutron_df_1_minute_background):,}")
print(f"Estimated background mean: {lambda_:.6f} counts/min")
```

<!-- #region id="44592eba" -->
## Step 1 - A Mean-Plus-Z-Standard-Deviations Threshold

For a Poisson-distributed count $K$ with mean $\lambda$, the standard deviation is $\sqrt{\lambda}$. A simple screening rule flags a minute when

$$
K > \lambda + Z\sqrt{\lambda}.
$$

This rule is easy to visualize. However, when $\lambda$ is only a few counts per minute, the Poisson distribution is discrete and asymmetric. A threshold three standard deviations above its mean does not have the same upper-tail probability as a three-standard-deviation threshold in a normal distribution.

The function below keeps this simple rule as an illustration. It also reports the first integer count that it flags and the corresponding Poisson upper-tail probability at the estimated background mean.
<!-- #endregion -->

```python id="7ff9e36f"
def plot_outliers(Z):
    threshold_poisson = lambda_ + Z * np.sqrt(lambda_)
    first_flagged_count = max(0, int(np.floor(threshold_poisson)) + 1)
    poisson_outliers = neutron_df_1_minute_background[
        neutron_df_1_minute_background["Counts"] > threshold_poisson
    ]
    tail_probability = stats.poisson.sf(first_flagged_count - 1, mu=lambda_)

    plt.figure(figsize=(14, 5))
    plt.plot(neutron_df_1_minute_background.index, neutron_df_1_minute_background["Counts"],
             label="Counts per Minute", color='blue', linewidth=1)
    plt.scatter(poisson_outliers.index, poisson_outliers["Counts"],
                color='red', label=f"Flagged minutes (counts >= {first_flagged_count})", zorder=5)
    plt.axhline(threshold_poisson, color='gray', linestyle='--', linewidth=1.5,
                label=f"Mean + {Z:g} standard deviations")
    plt.title(f"One-minute Neutron Counts with a Mean-plus-{Z:g}-SD Threshold")
    plt.xlabel("Time")
    plt.ylabel("Counts per Minute")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Count threshold: {threshold_poisson:.6f}")
    print(f"First flagged integer count: {first_flagged_count}")
    print(f"Poisson probability of reaching that count or higher: "
          f"{tail_probability:.6g} ({100 * tail_probability:.3f}% per minute)")
    print(f"Minutes flagged by this count threshold: {len(poisson_outliers):,}")
    return threshold_poisson
```

<!-- #region id="f90857e8" -->
## Step 2 - Interpreting the Count-Threshold Plot

- **Blue line:** the observed one-minute neutron counts.
- **Gray dashed line:** the mean-plus-$Z$-standard-deviations threshold.
- **Red dots:** the minutes whose counts exceed that threshold.

The output beneath the plot gives the actual Poisson tail probability associated with the chosen integer cutoff. At this dataset's mean of about 2.11 counts per minute, setting $Z=3$ flags counts of 7 or more, with a model probability of about **0.605% per minute**. The one-sided normal $3\sigma$ tail is about **0.135%**. These probabilities differ because the normal approximation is inaccurate at this low mean count.

Changing $Z$ changes the screening threshold. Lower values flag more minutes, while higher values flag fewer. To choose a threshold from a specified Poisson tail probability, use Step 4.

The next section explains why even a small probability for one minute can produce many flags when a long record is scanned.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 414} id="ac069f07" outputId="7532f937-7b5d-4225-e5b5-346b81ed3579"
threshold_3sigma = plot_outliers(Z=3)
```

<!-- #region id="75737a86" -->
## Step 3 - Accounting for Multiple Comparisons

Suppose $N$ minutes are examined, and each test has an actual false-positive probability $\alpha$ under the background model. Let $V$ be the number of false flags. Its expected value is

$$
\mathbb{E}[V] = N\alpha.
$$

The family-wise error rate (FWER) is the probability of at least one false flag. For independent tests,

$$
\mathrm{FWER} = P(V\ge1) = 1-(1-\alpha)^N.
$$

As a one-day illustration, take $N=1440$ and $\alpha=0.00135$, approximately the one-sided normal $3\sigma$ tail probability. Then

$$
\mathbb{E}[V] = 1440\times0.00135 = 1.944,
$$

$$
\mathrm{FWER} = 1-(1-0.00135)^{1440} \approx 0.8571.
$$

Thus, this example has about an **86% chance of at least one false flag**, with **1.94 false flags expected on average**. The approximation $\mathrm{FWER}\approx N\alpha$ is useful only when $N\alpha\ll1$. This example uses a hypothetical per-minute probability; the low-count rule in Step 1 has the different Poisson probability printed beside its plot.

### The Bonferroni Correction

To bound the probability of any false flag by a chosen family level $\alpha_{\rm family}$, the [Bonferroni method](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm) uses

$$
\alpha_{\rm per} = \frac{\alpha_{\rm family}}{N}.
$$

When each individual test has false-positive probability at most $\alpha_{\rm per}$,

$$
\mathrm{FWER} \le \sum_{i=1}^{N}\alpha_{\rm per} = \alpha_{\rm family}.
$$

This bound does not require independence. For a family level of 0.05 over 1,440 minutes,

$$
\alpha_{\rm per} = \frac{0.05}{1440} \approx 3.4722\times10^{-5}.
$$

Its equivalent one-sided normal threshold is

$$
Z_{\rm adj} = \Phi^{-1}(1-\alpha_{\rm per}) \approx 3.98,
$$

where $\Phi$ is the standard normal cumulative distribution function. The conversion uses the upper tail because we are looking for unusually high counts.

Step 4 calculates Poisson upper-tail probabilities and takes $N$ from the number of minutes actually plotted. Its Bonferroni reference therefore covers the entire selected record.
<!-- #endregion -->

<!-- #region id="9578a701" -->
## Step 4 - Plotting Poisson Upper-Tail Probabilities

For each observed one-minute count $k$, calculate the probability of obtaining that count or a larger one under a Poisson model with mean $\lambda$:

$$
p_k = P(K\ge k) = 1-F(k-1;\lambda).
$$

Here $F$ is the Poisson cumulative distribution function. We plot $\log_{10}(p_k)$ so that small tail probabilities remain easy to compare. The code uses [SciPy's Poisson survival functions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html) to evaluate the upper tail directly and avoid subtracting a value close to one.

The background mean is estimated from the same record being scanned. These calculations treat that estimate as fixed and omit its uncertainty. The plotted probabilities and Bonferroni line therefore illustrate screening under the fitted constant-rate Poisson model; a formal false-alarm guarantee requires calibrated individual tests that account for background estimation.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="afe106c4" outputId="4c2fa824-900f-4fbd-abe1-303168afdbf8"
count_values = neutron_df_1_minute_background["Counts"].to_numpy()
if len(count_values) == 0:
    raise ValueError("The selected background period contains no minute bins.")
if (not np.all(np.isfinite(count_values)) or np.any(count_values < 0)
        or not np.allclose(count_values, np.rint(count_values), rtol=0, atol=1e-10)):
    raise ValueError("Poisson tail calculations require nonnegative integer counts.")

# Round only floating-point representation noise before converting to integers.
counts = np.rint(count_values).astype(np.int64)
lambda_ = counts.mean()
N_tests = len(counts)
alpha_family = 0.05
alpha_per_test = alpha_family / N_tests
alpha_single = stats.norm.sf(3)  # One-sided normal 3-sigma probability as a reference.

# P(K >= k) = sf(k - 1). Calculate log tails directly for numerical stability.
p_tail = stats.poisson.sf(counts - 1, mu=lambda_)
log_p = stats.poisson.logsf(counts - 1, mu=lambda_) / np.log(10)
pointwise_flags = log_p <= np.log10(alpha_single)
bonferroni_flags = log_p <= np.log10(alpha_per_test)

print(f"Number of tested minute bins: {N_tests:,}")
print(f"Pointwise upper-tail reference: {alpha_single:.8g}")
print(f"Bonferroni reference: {alpha_family:g}/{N_tests} = {alpha_per_test:.8g}")
print(f"Log10 Bonferroni reference: {np.log10(alpha_per_test):.6f}")
print(f"Minutes at or below the pointwise reference: {pointwise_flags.sum():,}")
print(f"Minutes at or below the Bonferroni reference: {bonferroni_flags.sum():,}")

plt.figure(figsize=(14, 4))
plt.plot(neutron_df_1_minute_background.index, log_p, lw=1, color='blue')
plt.axhline(np.log10(alpha_single), color='gray', linestyle='--',
            label='Pointwise upper-tail reference (normal 3σ probability)')
plt.axhline(np.log10(alpha_per_test), color='red', linestyle='--',
            label=f'Bonferroni reference (0.05/{N_tests:,} minutes)')
plt.ylim(min(-10, np.floor(log_p.min()) - 1), 0)
plt.xlabel("Time")
plt.ylabel(r"$\log_{10} P(K \geq k)$")
plt.title("Upper-tail Probability of Each One-minute Count Under the Fitted Poisson Background")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<!-- #region id="4b1a8660" -->
## Step 5 - Interpreting the Log Tail-Probability Plot

- **Vertical axis:** a value of $-3$ means a model tail probability of $10^{-3}=0.001$. Lower points correspond to smaller probabilities of observing that many counts or more.
- **Gray dashed line:** the pointwise upper-tail reference is about 0.00135, or $\log_{10}(p)\approx-2.87$. This is a probability reference taken from the normal $3\sigma$ tail; the plotted count probabilities are evaluated with the Poisson distribution.
- **Red dashed line:** the nominal Bonferroni reference is $\log_{10}(0.05/N)$, with the actual $N$ and threshold printed above. For the separate one-day example with $N=1440$, this value would be approximately $-4.459$. The multiweek record shown here uses its full number of minute bins.
- **Points on or below a line:** these meet the corresponding screening threshold under the fitted model. The printed totals show how many minutes are flagged by each reference.

Use flagged minutes to guide follow-up checks of the detector and background conditions. A small tail probability identifies an unusual count under the chosen model; establishing its physical cause requires that further investigation.
<!-- #endregion -->

<!-- #region id="a7da1bf1" -->
## Conclusion

This notebook compares a simple mean-plus-$Z$-standard-deviations rule with upper-tail probabilities calculated from a fitted Poisson background. At low mean counts, the two approaches can flag different minutes. The Bonferroni reference then shows how scanning more minutes calls for a smaller per-minute probability threshold.

The displayed counts, probabilities, and flagged-minute totals provide a reproducible way to select candidates for investigation. Their interpretation depends on the constant-rate Poisson model and the treatment of the estimated background mean described above.

The same calculations can be explored for gamma counts in a future notebook.
<!-- #endregion -->
