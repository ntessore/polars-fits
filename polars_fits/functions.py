import glob

from collections.abc import Iterator
from typing import Any
from warnings import warn

import polars as pl
import fitsio
import parse

from polars_fits.filter import CfitsioFilter


class PathDataParser:
    """Parser for extra data from paths."""

    def __init__(self, pattern: str) -> None:
        """Create a new parser with the given pattern."""
        self.parser = parse.compile(pattern)

    def __call__(self, path: str) -> dict[str, Any]:
        """Extract data from path."""
        result = self.parser.parse(path)
        if result is None:
            raise ValueError(f"path does not match pattern: {path}")
        return result.named


def get_hdu(
    fits: fitsio.FITS,
    ext: str | int | None = None,
) -> fitsio.hdu.ImageHDU | fitsio.hdu.TableHDU | fitsio.hdu.AsciiTableHDU:
    """Return HDU from *ext* or first extension with data."""
    if ext is None:
        for ext in range(len(fits)):
            if fits[ext].has_data():
                break
        else:
            raise IOError("No extensions have data")
    return fits[ext]


def scan_fits(
    paths: str | list[str],
    *,
    ext: int | str | None = None,
    columns: list[str] | None = None,
    row_filtering: bool = False,
    pattern: str | None = None,
) -> pl.LazyFrame:
    """
    Read FITS files.

    If *row_filtering* is true, use CFITSIO row filtering to read only a subset
    of rows from file.
    """

    if isinstance(paths, str):
        paths = [paths]
    paths = sum((glob.glob(path) for path in paths), [])

    if not paths:
        raise FileNotFoundError(str(path))

    path_parser = PathDataParser(pattern) if pattern is not None else None

    def schema() -> pl.Schema:
        """Produce schema of FITS files."""
        with fitsio.FITS(path := paths[0]) as fits:
            df = pl.from_numpy(get_hdu(fits, ext).read(columns=columns, rows=[]))
        if path_parser is not None:
            df = df.with_columns_seq(**path_parser(path))
        return df.schema

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

        rowfilter = None
        if row_filtering and predicate is not None:
            try:
                rowfilter = CfitsioFilter.from_expr(predicate)
            except ValueError as err:
                warn(f"FITS predicate pushdown: {err!s}")
            else:
                # all filtering can be done at FITS level
                predicate = None

        for path in paths:
            if n_rows is not None and n_rows < 1:
                break

            path_columns = path_parser(path) if path_parser is not None else None

            if with_columns is not None and path_columns is not None:
                fits_columns = [col for col in with_columns if col not in path_columns]
            else:
                fits_columns = with_columns

            with fitsio.FITS(path) as fits:
                hdu = get_hdu(fits, ext)

                file_n_rows = hdu.get_nrows()
                if n_rows is not None:
                    file_n_rows = min(file_n_rows, n_rows)

                file_batch_size = batch_size if batch_size is not None else file_n_rows

                for start in range(0, file_n_rows, file_batch_size):
                    stop = min(start + file_batch_size, file_n_rows)

                    if rowfilter is None:
                        rows = list(range(start, stop))
                    else:
                        rows = hdu.where(rowfilter, start, stop) + start

                    if len(rows) == 0:
                        continue

                    df = pl.from_numpy(
                        hdu.read(ext=ext, columns=fits_columns, rows=rows)
                    )

                    if path_columns is not None:
                        df = df.with_columns_seq(**path_columns)

                    if predicate is not None:
                        df = df.filter(predicate)

                    yield df

                if n_rows is not None:
                    n_rows -= file_n_rows

    return pl.io.plugins.register_io_source(
        io_source=source_generator,
        schema=schema,
        validate_schema=True,
        is_pure=True,
    )
