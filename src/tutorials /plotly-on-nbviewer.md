---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/tutorials /plotly-on-nbviewer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/tutorials /plotly-on-nbviewer.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>

<!-- #region id="btHAjyTD6_9u" -->
# Getting Plotly to show in nbviewer
<!-- #endregion -->

<!-- #region id="EvPFMOKa7I_E" -->
Simply importing plotly, making your plots and saving the notebook to Github with the relevent [nbviewer link](https://nbviewer.org/) (which is automatically generated and attached as a button in all notebooks in this repository) will not make the plots render in nbviewer. You need to do a little extra work.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sXx3ReL_616h" outputId="d37bb0b4-e3a7-48c9-f911-4b6c47d3fb59"
# In earlier versions of plotly, nbviewer rendering was broken. We therefore need to force a version that works.
!pip install -q --upgrade --force-reinstall "plotly==6.8.0"
```

```python id="eZEkQpom7n6p"
import plotly.graph_objects as go

# This needs to be added to make the plots work locally, in Google Colab and on nbviewer.
import plotly.io as pio
pio.renderers.default = "notebook_connected+colab"

```

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="xy87W12c8BED" outputId="9595ff2d-1752-47ac-b02e-1f4631f08aee"
fig = go.Figure()
fig.add_trace(go.Scatter(y=[2, 1, 4, 3]))
fig.add_trace(go.Bar(y=[1, 4, 3, 2]))
```

```python id="X_wHq5Tx8UXv"

```
