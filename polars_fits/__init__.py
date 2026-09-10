import glob

import polars as pl

from functools import partial
from typing import Iterator

from fitsio import FITS


def scan_fits(
    paths: str | list[str],
    *,
    ext: int | str | None = None,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """
    Read FITS files.
    """

    if isinstance(paths, str):
        paths = [paths]
    files = sum((glob.glob(path) for path in paths), [])

    if not files:
        raise FileNotFoundError(str(path))

    with FITS(files[0]) as fits:
        if ext is None:
            for i in range(len(fits)):
                if fits[i].has_data():
                    ext = i
                    break
            else:
                raise IOError("No extensions have data")

        schema = pl.from_numpy(fits[ext].read(columns=columns, rows=[])).schema

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """
        Generator function that creates the source.
        This function will be registered as IO source.
        """

        if with_columns is None:
            with_columns = columns

        for file in files:
            if n_rows is not None and n_rows < 1:
                break

            with FITS(file) as fits:
                hdu = fits[ext]

                file_n_rows = hdu.get_nrows()
                if n_rows is not None:
                    file_n_rows = min(file_n_rows, n_rows)

                file_batch_size = batch_size if batch_size is not None else file_n_rows

                for start in range(0, file_n_rows, file_batch_size):
                    stop = min(start + file_batch_size, file_n_rows)

                    df = pl.from_numpy(
                        hdu.read(
                            ext=ext,
                            columns=with_columns,
                            rows=list(range(start, stop)),
                        )
                    )

                    if predicate is not None:
                        df = df.filter(predicate)

                    yield df

                if n_rows is not None:
                    n_rows -= file_n_rows

    return pl.io.plugins.register_io_source(
        io_source=source_generator,
        schema=schema,
        validate_schema=True,
    )
