# mars_dtm.npy

A 120x70 real elevation grid (meters, ~1.01 m/pixel) cropped from a HiRISE
Digital Terrain Model of Mars, used by `examples/mars_demo.py`.

**Source**: HiRISE DTM `DTEEC_010228_1490_016320_1490_A01`, from the
University of Arizona HiRISE DTM archive:
`https://www.uahirise.org/PDS/DTM/PSP/ORB_010200_010299/PSP_010228_1490_ESP_016320_1490/DTEEC_010228_1490_016320_1490_A01.IMG`
(McEwen et al., 2007, "Mars Reconnaissance Orbiter's High Resolution Imaging
Science Experiment (HiRISE)", JGR Planets 112(E5) -- the same dataset citation
used by the GOOSE paper's Mars exploration experiment, Turchetta, Berkenkamp &
Krause, NeurIPS 2019, Sec. 4).

**Extraction**: rows `[2890:3010]`, columns `[1955:2025]` of the full DTM
raster, read remotely via `rasterio`'s GDAL `/vsicurl/` streaming (no full
file download needed). These are the exact pixel offsets and crop size used
by the earlier, directly related Turchetta et al. (2016) "Safe Exploration in
Finite Markov Decision Processes with Gaussian Processes" Mars example
(`github.com/befelix/SafeMDP`, `examples/mars/mars_utilities.py`), which the
GOOSE paper's own Mars experiment builds on. The GOOSE paper does not publish
the exact identities of the 16 Mars locations it uses, so this is the closest
verifiable, reproducible stand-in: the same instrument, data source, and
extraction methodology, from the authors' own earlier public code.

Regenerate with:
```python
import rasterio
from rasterio.windows import Window
url = "/vsicurl/https://www.uahirise.org/PDS/DTM/PSP/ORB_010200_010299/PSP_010228_1490_ESP_016320_1490/DTEEC_010228_1490_016320_1490_A01.IMG"
with rasterio.open(url) as ds:
    arr = ds.read(1, window=Window(col_off=1955, row_off=2890, width=70, height=120))
```
