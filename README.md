# FITS file reader for Polars

FITS file reader for Polars.

```py
from polars_fits import scan_fits

ldf = scan_fits("/path/to/EUC_MER_FINAL-CAT_*.fits.gz")

...
```

## Features

* glob support to work on multiple files
* pattern support to parse data from file names

  ```py
  ldf = scan_fits(
      "/path/to/EUC_MER_FINAL-CAT_*.fits.gz"
      pattern="{}/EUC_MER_FINAL-CAT_TILE{tile_index:d}-{}.fits.gz",
  )
  # ldf has tile_index column inferred from each file name
  ```
