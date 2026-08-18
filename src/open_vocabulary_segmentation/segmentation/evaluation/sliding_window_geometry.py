"""Pure, deterministic, immutable sliding-window and patch-grid geometry.

This module reproduces the coordinate arithmetic of the production
``DINOTextSegInference.slide_inference`` sliding-window loop as reusable,
independently testable objects. It introduces geometry only: no caching,
crop consensus, or COVER-DR behavior lives here.

Coordinate convention
----------------------
Every coordinate in this module is ``(row, col)`` i.e. ``(y, x)`` in
integer pixel-index space. Crop/window intervals are half-open:
``[origin, end)``. This module never uses an ``(x, y)`` convention.

Patch-grid coordinates (Section 5) are also ``(row, col)`` and are kept in
a strictly separate namespace (``patch_row``/``patch_col``,
``grid_row``/``grid_col``) from pixel coordinates to avoid unit confusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class SlidingWindowGeometryError(ValueError):
    """Raised when geometry inputs violate the exact-type/shape contract."""


def _require_exact_int(value, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SlidingWindowGeometryError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise SlidingWindowGeometryError(f"{label} must be >= {minimum}")
    return value


def _require_bool(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise SlidingWindowGeometryError(f"{label} must be an exact boolean")
    return value


# ---------------------------------------------------------------------------
# Core value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialSize:
    """A positive ``(height, width)`` extent, in pixels or grid nodes."""

    height: int
    width: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "height", _require_exact_int(self.height, "height", minimum=1)
        )
        object.__setattr__(
            self, "width", _require_exact_int(self.width, "width", minimum=1)
        )

    def as_tuple(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class PixelCoordinate:
    """A non-negative ``(row, col)`` integer pixel-index coordinate."""

    row: int
    col: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "row", _require_exact_int(self.row, "row", minimum=0)
        )
        object.__setattr__(
            self, "col", _require_exact_int(self.col, "col", minimum=0)
        )

    def as_tuple(self) -> tuple[int, int]:
        return (self.row, self.col)


@dataclass(frozen=True)
class Rectangle:
    """A half-open pixel rectangle ``[origin, end)``; used for intersections."""

    origin: PixelCoordinate
    end: PixelCoordinate

    def __post_init__(self) -> None:
        if self.end.row < self.origin.row or self.end.col < self.origin.col:
            raise SlidingWindowGeometryError(
                "Rectangle end must not precede its origin"
            )

    @property
    def height(self) -> int:
        return self.end.row - self.origin.row

    @property
    def width(self) -> int:
        return self.end.col - self.origin.col

    @property
    def is_empty(self) -> bool:
        return self.height == 0 or self.width == 0

    @property
    def area(self) -> int:
        return self.height * self.width

    def as_slices(self) -> tuple[slice, slice]:
        return (
            slice(self.origin.row, self.end.row),
            slice(self.origin.col, self.end.col),
        )


# ---------------------------------------------------------------------------
# Sliding-window geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowGeometry:
    """One immutable crop window produced by the legacy sliding-window scan.

    ``origin``/``end`` are the actual (possibly back-shifted) half-open
    pixel bounds used to slice both the input image and the accumulation
    buffers. ``nominal_origin`` is the un-clamped grid position
    ``(grid_row * stride.height, grid_col * stride.width)`` before any
    terminal-window clamping was applied.
    """

    index: int
    grid_row: int
    grid_col: int
    nominal_origin: PixelCoordinate
    origin: PixelCoordinate
    end: PixelCoordinate
    is_top: bool
    is_bottom: bool
    is_left: bool
    is_right: bool
    clamped_vertical: bool
    clamped_horizontal: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index", _require_exact_int(self.index, "index", minimum=0)
        )
        object.__setattr__(
            self, "grid_row", _require_exact_int(self.grid_row, "grid_row", minimum=0)
        )
        object.__setattr__(
            self, "grid_col", _require_exact_int(self.grid_col, "grid_col", minimum=0)
        )
        for name in ("is_top", "is_bottom", "is_left", "is_right",
                     "clamped_vertical", "clamped_horizontal"):
            object.__setattr__(self, name, _require_bool(getattr(self, name), name))
        if not isinstance(self.nominal_origin, PixelCoordinate):
            raise SlidingWindowGeometryError("nominal_origin must be a PixelCoordinate")
        if not isinstance(self.origin, PixelCoordinate):
            raise SlidingWindowGeometryError("origin must be a PixelCoordinate")
        if not isinstance(self.end, PixelCoordinate):
            raise SlidingWindowGeometryError("end must be a PixelCoordinate")
        if self.end.row <= self.origin.row or self.end.col <= self.origin.col:
            raise SlidingWindowGeometryError(
                "window end must be strictly greater than its origin"
            )

    @property
    def extent(self) -> SpatialSize:
        return SpatialSize(
            self.end.row - self.origin.row, self.end.col - self.origin.col
        )

    @property
    def rectangle(self) -> Rectangle:
        return Rectangle(self.origin, self.end)

    @property
    def crop_slice(self) -> tuple[slice, slice]:
        """Slice used to extract this window's crop from the source image."""
        return self.rectangle.as_slices()

    @property
    def accumulation_slice(self) -> tuple[slice, slice]:
        """Slice used to accumulate this window's output into the full-image
        prediction/count buffers.

        Under the current uniform-stitching protocol this is numerically
        identical to :attr:`crop_slice` (both source and destination share
        the same global coordinate system). The accessor is kept distinct
        because a future non-uniform stitching scheme could accumulate into
        a different destination footprint than the one it read from.
        """
        return self.rectangle.as_slices()

    def is_lattice_aligned(self, patch_size: SpatialSize) -> bool:
        """Whether this window's origin falls exactly on a patch lattice.

        A window is aligned to ``patch_size`` when its top-left pixel
        origin is an integer multiple of the patch size along both axes,
        i.e. patch node boundaries computed globally and locally coincide.
        """
        if not isinstance(patch_size, SpatialSize):
            raise SlidingWindowGeometryError("patch_size must be a SpatialSize")
        return (
            self.origin.row % patch_size.height == 0
            and self.origin.col % patch_size.width == 0
        )


@dataclass(frozen=True)
class SlidingWindowPlan:
    """An ordered, immutable set of windows covering one image.

    Construction reproduces production ``slide_inference`` grid/window math
    exactly (see :meth:`build`); this type only stores the result.
    """

    image_size: SpatialSize
    crop_size: SpatialSize
    stride: SpatialSize
    grid_rows: int
    grid_cols: int
    windows: tuple[WindowGeometry, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("image_size", self.image_size),
            ("crop_size", self.crop_size),
            ("stride", self.stride),
        ):
            if not isinstance(value, SpatialSize):
                raise SlidingWindowGeometryError(f"{name} must be a SpatialSize")
        object.__setattr__(
            self, "grid_rows", _require_exact_int(self.grid_rows, "grid_rows", minimum=1)
        )
        object.__setattr__(
            self, "grid_cols", _require_exact_int(self.grid_cols, "grid_cols", minimum=1)
        )
        if not isinstance(self.windows, tuple):
            raise SlidingWindowGeometryError("windows must be an immutable tuple")
        if len(self.windows) != self.grid_rows * self.grid_cols:
            raise SlidingWindowGeometryError(
                "window count does not match grid_rows * grid_cols"
            )
        for expected_index, window in enumerate(self.windows):
            if not isinstance(window, WindowGeometry):
                raise SlidingWindowGeometryError("windows must contain WindowGeometry")
            if window.index != expected_index:
                raise SlidingWindowGeometryError(
                    "windows must be ordered row-major by flat index"
                )

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @classmethod
    def build(
        cls, *, image_size: SpatialSize, crop_size: SpatialSize, stride: SpatialSize
    ) -> "SlidingWindowPlan":
        """Reproduce the exact legacy ``slide_inference`` grid/window math.

        Legacy reference (``dinotext_seg.py::slide_inference``)::

            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
        """
        if not isinstance(image_size, SpatialSize):
            raise SlidingWindowGeometryError("image_size must be a SpatialSize")
        if not isinstance(crop_size, SpatialSize):
            raise SlidingWindowGeometryError("crop_size must be a SpatialSize")
        if not isinstance(stride, SpatialSize):
            raise SlidingWindowGeometryError("stride must be a SpatialSize")

        h_img, w_img = image_size.height, image_size.width
        h_crop, w_crop = crop_size.height, crop_size.width
        h_stride, w_stride = stride.height, stride.width

        grid_rows = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        grid_cols = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        windows: list[WindowGeometry] = []
        for grid_row in range(grid_rows):
            for grid_col in range(grid_cols):
                y_nom = grid_row * h_stride
                x_nom = grid_col * w_stride
                y_end = min(y_nom + h_crop, h_img)
                x_end = min(x_nom + w_crop, w_img)
                y0 = max(y_end - h_crop, 0)
                x0 = max(x_end - w_crop, 0)
                windows.append(
                    WindowGeometry(
                        index=grid_row * grid_cols + grid_col,
                        grid_row=grid_row,
                        grid_col=grid_col,
                        nominal_origin=PixelCoordinate(y_nom, x_nom),
                        origin=PixelCoordinate(y0, x0),
                        end=PixelCoordinate(y_end, x_end),
                        is_top=(grid_row == 0),
                        is_bottom=(grid_row == grid_rows - 1),
                        is_left=(grid_col == 0),
                        is_right=(grid_col == grid_cols - 1),
                        clamped_vertical=(y0 != y_nom),
                        clamped_horizontal=(x0 != x_nom),
                    )
                )

        return cls(
            image_size=image_size,
            crop_size=crop_size,
            stride=stride,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            windows=tuple(windows),
        )

    # -- lookups ------------------------------------------------------

    def window_at(self, grid_row: int, grid_col: int) -> WindowGeometry:
        grid_row = _require_exact_int(grid_row, "grid_row", minimum=0)
        grid_col = _require_exact_int(grid_col, "grid_col", minimum=0)
        if grid_row >= self.grid_rows or grid_col >= self.grid_cols:
            raise SlidingWindowGeometryError("grid position is out of range")
        return self.windows[grid_row * self.grid_cols + grid_col]

    def window_by_index(self, flat_index: int) -> WindowGeometry:
        flat_index = _require_exact_int(flat_index, "flat_index", minimum=0)
        if flat_index >= len(self.windows):
            raise SlidingWindowGeometryError("flat_index is out of range")
        return self.windows[flat_index]

    # -- coverage -------------------------------------------------------

    def _row_bands(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (self.window_at(row, 0).origin.row, self.window_at(row, 0).end.row)
            for row in range(self.grid_rows)
        )

    def _col_bands(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (self.window_at(0, col).origin.col, self.window_at(0, col).end.col)
            for col in range(self.grid_cols)
        )

    def _matching_row_indices(self, row: int) -> tuple[int, ...]:
        return tuple(
            index
            for index, (start, end) in enumerate(self._row_bands())
            if start <= row < end
        )

    def _matching_col_indices(self, col: int) -> tuple[int, ...]:
        return tuple(
            index
            for index, (start, end) in enumerate(self._col_bands())
            if start <= col < end
        )

    def coverage_count(self, row: int, col: int) -> int:
        """Number of windows covering global pixel ``(row, col)``.

        Computed without materializing a dense map: the row-major grid is
        separable, so the count is the product of how many row-bands and
        how many column-bands contain the point.
        """
        row = _require_exact_int(row, "row", minimum=0)
        col = _require_exact_int(col, "col", minimum=0)
        return len(self._matching_row_indices(row)) * len(
            self._matching_col_indices(col)
        )

    def windows_covering_point(self, row: int, col: int) -> tuple[int, ...]:
        """Ordered (row-major) flat indices of windows covering the point."""
        row = _require_exact_int(row, "row", minimum=0)
        col = _require_exact_int(col, "col", minimum=0)
        row_indices = self._matching_row_indices(row)
        col_indices = self._matching_col_indices(col)
        return tuple(
            r * self.grid_cols + c for r in row_indices for c in col_indices
        )

    def row_coverage_counts(self) -> tuple[int, ...]:
        """Coverage count of each image row, length ``image_size.height``."""
        return tuple(
            len(self._matching_row_indices(row))
            for row in range(self.image_size.height)
        )

    def col_coverage_counts(self) -> tuple[int, ...]:
        """Coverage count of each image column, length ``image_size.width``."""
        return tuple(
            len(self._matching_col_indices(col))
            for col in range(self.image_size.width)
        )

    def build_coverage_map(self) -> tuple[tuple[int, ...], ...]:
        """Materialize a dense ``[H, W]`` coverage-count map (testing only)."""
        row_counts = self.row_coverage_counts()
        col_counts = self.col_coverage_counts()
        return tuple(
            tuple(row_count * col_count for col_count in col_counts)
            for row_count in row_counts
        )


# ---------------------------------------------------------------------------
# Pure coordinate operations
# ---------------------------------------------------------------------------


def local_to_global(window: WindowGeometry, local: PixelCoordinate) -> PixelCoordinate:
    """Map a window-local pixel coordinate to a global image coordinate."""
    if not isinstance(window, WindowGeometry):
        raise SlidingWindowGeometryError("window must be a WindowGeometry")
    if not isinstance(local, PixelCoordinate):
        raise SlidingWindowGeometryError("local must be a PixelCoordinate")
    return PixelCoordinate(window.origin.row + local.row, window.origin.col + local.col)


def global_to_local(
    window: WindowGeometry, global_coordinate: PixelCoordinate
) -> Optional[PixelCoordinate]:
    """Map a global pixel coordinate to window-local coordinates.

    Returns ``None`` when the point lies outside the window's half-open
    bounds.
    """
    if not isinstance(window, WindowGeometry):
        raise SlidingWindowGeometryError("window must be a WindowGeometry")
    if not isinstance(global_coordinate, PixelCoordinate):
        raise SlidingWindowGeometryError("global_coordinate must be a PixelCoordinate")
    if not contains_point(window, global_coordinate):
        return None
    return PixelCoordinate(
        global_coordinate.row - window.origin.row,
        global_coordinate.col - window.origin.col,
    )


def contains_point(window: WindowGeometry, point: PixelCoordinate) -> bool:
    """Half-open containment test: ``origin <= point < end`` on both axes."""
    if not isinstance(window, WindowGeometry):
        raise SlidingWindowGeometryError("window must be a WindowGeometry")
    if not isinstance(point, PixelCoordinate):
        raise SlidingWindowGeometryError("point must be a PixelCoordinate")
    return (
        window.origin.row <= point.row < window.end.row
        and window.origin.col <= point.col < window.end.col
    )


def intersect_windows(
    window_a: WindowGeometry, window_b: WindowGeometry
) -> Optional[Rectangle]:
    """Half-open pixel intersection of two windows, or ``None`` if disjoint."""
    if not isinstance(window_a, WindowGeometry) or not isinstance(window_b, WindowGeometry):
        raise SlidingWindowGeometryError("window_a and window_b must be WindowGeometry")
    row0 = max(window_a.origin.row, window_b.origin.row)
    col0 = max(window_a.origin.col, window_b.origin.col)
    row1 = min(window_a.end.row, window_b.end.row)
    col1 = min(window_a.end.col, window_b.end.col)
    if row1 <= row0 or col1 <= col0:
        return None
    return Rectangle(PixelCoordinate(row0, col0), PixelCoordinate(row1, col1))


def overlap_area(window_a: WindowGeometry, window_b: WindowGeometry) -> int:
    """Pixel area of the intersection of two windows (0 if disjoint)."""
    rectangle = intersect_windows(window_a, window_b)
    return 0 if rectangle is None else rectangle.area


# ---------------------------------------------------------------------------
# Patch-grid geometry (Section 5): geometry only, no consensus/sampling.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchGridSpec:
    """Declares how one crop/window's pixel extent maps onto a patch grid.

    Canonical values (crop 448x448, patch 14x14, grid 32x32, stride
    224x224, align_corners=True) are supplied by callers; nothing here
    embeds them as defaults.
    """

    crop_size: SpatialSize
    patch_size: SpatialSize
    grid_size: SpatialSize
    align_corners: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("crop_size", self.crop_size),
            ("patch_size", self.patch_size),
            ("grid_size", self.grid_size),
        ):
            if not isinstance(value, SpatialSize):
                raise SlidingWindowGeometryError(f"{name} must be a SpatialSize")
        object.__setattr__(
            self, "align_corners", _require_bool(self.align_corners, "align_corners")
        )
        if self.crop_size.height != self.patch_size.height * self.grid_size.height:
            raise SlidingWindowGeometryError(
                "crop_size.height must equal patch_size.height * grid_size.height"
            )
        if self.crop_size.width != self.patch_size.width * self.grid_size.width:
            raise SlidingWindowGeometryError(
                "crop_size.width must equal patch_size.width * grid_size.width"
            )

    def _require_patch_node(self, patch_row: int, patch_col: int) -> tuple[int, int]:
        patch_row = _require_exact_int(patch_row, "patch_row", minimum=0)
        patch_col = _require_exact_int(patch_col, "patch_col", minimum=0)
        if patch_row >= self.grid_size.height or patch_col >= self.grid_size.width:
            raise SlidingWindowGeometryError("patch node is outside the patch grid")
        return patch_row, patch_col

    def patch_footprint_local(self, patch_row: int, patch_col: int) -> Rectangle:
        """Half-open local pixel footprint of one patch node within its window."""
        patch_row, patch_col = self._require_patch_node(patch_row, patch_col)
        row0 = patch_row * self.patch_size.height
        col0 = patch_col * self.patch_size.width
        return Rectangle(
            PixelCoordinate(row0, col0),
            PixelCoordinate(row0 + self.patch_size.height, col0 + self.patch_size.width),
        )

    def patch_footprint_global(
        self, window: WindowGeometry, patch_row: int, patch_col: int
    ) -> Rectangle:
        """Half-open global pixel footprint of one patch node in ``window``."""
        if not isinstance(window, WindowGeometry):
            raise SlidingWindowGeometryError("window must be a WindowGeometry")
        local = self.patch_footprint_local(patch_row, patch_col)
        return Rectangle(
            local_to_global(window, local.origin),
            local_to_global(window, local.end),
        )

    def patch_center_doubled_local(self, patch_row: int, patch_col: int) -> tuple[int, int]:
        """Doubled-integer local patch center ``(2*row, 2*col)``.

        For a half-open footprint ``[a, b)`` the exact center is
        ``(a + b - 1) / 2``; ``a + b - 1`` is always an integer, so the
        doubled coordinate ``a + b - 1`` (i.e. ``2 * center``) is returned
        directly, avoiding float rounding for the common half-integer case.
        """
        footprint = self.patch_footprint_local(patch_row, patch_col)
        return (
            footprint.origin.row + footprint.end.row - 1,
            footprint.origin.col + footprint.end.col - 1,
        )

    def patch_center_doubled_global(
        self, window: WindowGeometry, patch_row: int, patch_col: int
    ) -> tuple[int, int]:
        """Doubled-integer global patch center; see
        :meth:`patch_center_doubled_local`."""
        doubled_row, doubled_col = self.patch_center_doubled_local(patch_row, patch_col)
        return (
            doubled_row + 2 * window.origin.row,
            doubled_col + 2 * window.origin.col,
        )

    def flatten_node(self, patch_row: int, patch_col: int) -> int:
        patch_row, patch_col = self._require_patch_node(patch_row, patch_col)
        return patch_row * self.grid_size.width + patch_col

    def unflatten_node(self, flat_index: int) -> tuple[int, int]:
        flat_index = _require_exact_int(flat_index, "flat_index", minimum=0)
        total = self.grid_size.height * self.grid_size.width
        if flat_index >= total:
            raise SlidingWindowGeometryError("flat_index is outside the patch grid")
        return divmod(flat_index, self.grid_size.width)

    def native_patch_node(
        self,
        source_window: WindowGeometry,
        patch_row: int,
        patch_col: int,
        target_window: WindowGeometry,
    ) -> Optional[tuple[int, int]]:
        """Map a patch node from ``source_window`` to the node in
        ``target_window`` whose *global* patch center coincides exactly.

        Returns ``None`` when native alignment is undefined: the origin
        displacement between windows is not an exact multiple of the patch
        size, or the mapped node falls outside ``target_window``'s grid.
        This never interpolates features; it is pure index geometry.
        """
        if not isinstance(source_window, WindowGeometry) or not isinstance(
            target_window, WindowGeometry
        ):
            raise SlidingWindowGeometryError(
                "source_window and target_window must be WindowGeometry"
            )
        patch_row, patch_col = self._require_patch_node(patch_row, patch_col)

        displacement_row = source_window.origin.row - target_window.origin.row
        displacement_col = source_window.origin.col - target_window.origin.col
        if (
            displacement_row % self.patch_size.height != 0
            or displacement_col % self.patch_size.width != 0
        ):
            return None

        target_row = patch_row + displacement_row // self.patch_size.height
        target_col = patch_col + displacement_col // self.patch_size.width
        if (
            target_row < 0
            or target_col < 0
            or target_row >= self.grid_size.height
            or target_col >= self.grid_size.width
        ):
            return None
        return (target_row, target_col)


@dataclass(frozen=True)
class BilinearStencil:
    """Pure geometry of a 2x2 ``align_corners``-consistent bilinear stencil.

    Contains grid neighbor indices and weights only; it never samples a
    tensor.
    """

    row_low: int
    row_high: int
    col_low: int
    col_high: int
    weight_00: float
    weight_01: float
    weight_10: float
    weight_11: float

    def weights(self) -> tuple[float, float, float, float]:
        return (self.weight_00, self.weight_01, self.weight_10, self.weight_11)

    def neighbor_grid_indices(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        return (
            (self.row_low, self.col_low),
            (self.row_low, self.col_high),
            (self.row_high, self.col_low),
            (self.row_high, self.col_high),
        )

    def flattened_node_indices(self, grid_size: SpatialSize) -> tuple[int, int, int, int]:
        if not isinstance(grid_size, SpatialSize):
            raise SlidingWindowGeometryError("grid_size must be a SpatialSize")
        width = grid_size.width
        return (
            self.row_low * width + self.col_low,
            self.row_low * width + self.col_high,
            self.row_high * width + self.col_low,
            self.row_high * width + self.col_high,
        )


def _axis_stencil(
    coordinate_numerator: int, coordinate_denominator: int, output_extent: int, grid_extent: int
) -> tuple[int, int, float]:
    """One-axis ``align_corners=True`` stencil: ``u = p*(G-1)/(L-1)``.

    The coordinate is supplied as an exact rational
    ``coordinate_numerator / coordinate_denominator`` (denominator 1 for
    integer pixel-index coordinates, 2 for half-integer patch centers), so
    callers never have to construct a float coordinate by hand; only the
    output fraction/weights (inherently real-valued) use floating point.
    """
    if output_extent < 2:
        raise SlidingWindowGeometryError(
            "output_extent must be at least 2 for an align_corners=True stencil"
        )
    if grid_extent < 1:
        raise SlidingWindowGeometryError("grid_extent must be at least 1")
    coordinate = coordinate_numerator / coordinate_denominator
    if coordinate < 0 or coordinate > output_extent - 1:
        raise SlidingWindowGeometryError(
            "coordinate must lie within [0, output_extent - 1]"
        )
    if grid_extent == 1:
        return 0, 0, 0.0
    scaled = coordinate * (grid_extent - 1) / (output_extent - 1)
    low = int(scaled)
    if low >= grid_extent - 1:
        low = grid_extent - 2
    high = low + 1
    fraction = scaled - low
    fraction = min(max(fraction, 0.0), 1.0)
    return low, high, fraction


def _combine_axis_stencils(
    row_low: int, row_high: int, row_fraction: float,
    col_low: int, col_high: int, col_fraction: float,
) -> BilinearStencil:
    weight_00 = (1 - row_fraction) * (1 - col_fraction)
    weight_01 = (1 - row_fraction) * col_fraction
    weight_10 = row_fraction * (1 - col_fraction)
    weight_11 = row_fraction * col_fraction
    return BilinearStencil(
        row_low=row_low,
        row_high=row_high,
        col_low=col_low,
        col_high=col_high,
        weight_00=weight_00,
        weight_01=weight_01,
        weight_10=weight_10,
        weight_11=weight_11,
    )


def _require_align_corners_true(align_corners: bool) -> None:
    if not align_corners:
        raise SlidingWindowGeometryError(
            "only align_corners=True is implemented; canonical evaluation "
            "always uses align_corners=True"
        )


def bilinear_stencil(
    coordinate: PixelCoordinate,
    output_extent: SpatialSize,
    grid_size: SpatialSize,
    *,
    align_corners: bool,
) -> BilinearStencil:
    """Pure interpolation-stencil geometry mapping one integer output
    pixel-index coordinate into a patch-score grid, under the declared
    ``align_corners`` convention. Computes indices/weights only; never
    samples a tensor.
    """
    if not isinstance(coordinate, PixelCoordinate):
        raise SlidingWindowGeometryError("coordinate must be a PixelCoordinate")
    if not isinstance(output_extent, SpatialSize):
        raise SlidingWindowGeometryError("output_extent must be a SpatialSize")
    if not isinstance(grid_size, SpatialSize):
        raise SlidingWindowGeometryError("grid_size must be a SpatialSize")
    _require_align_corners_true(align_corners)

    row_low, row_high, row_fraction = _axis_stencil(
        coordinate.row, 1, output_extent.height, grid_size.height
    )
    col_low, col_high, col_fraction = _axis_stencil(
        coordinate.col, 1, output_extent.width, grid_size.width
    )
    return _combine_axis_stencils(
        row_low, row_high, row_fraction, col_low, col_high, col_fraction
    )


def bilinear_stencil_from_doubled(
    doubled_row: int,
    doubled_col: int,
    output_extent: SpatialSize,
    grid_size: SpatialSize,
    *,
    align_corners: bool,
) -> BilinearStencil:
    """Like :func:`bilinear_stencil`, but for a continuous/half-integer
    coordinate given as an exact doubled integer (``row = doubled_row/2``,
    ``col = doubled_col/2``) — the same representation produced by
    :meth:`PatchGridSpec.patch_center_doubled_local`/``_global``. This is
    how patch-center coordinates reach the stencil without ever
    constructing a float coordinate by hand or risking float equality
    errors on the input side.
    """
    doubled_row = _require_exact_int(doubled_row, "doubled_row", minimum=0)
    doubled_col = _require_exact_int(doubled_col, "doubled_col", minimum=0)
    if not isinstance(output_extent, SpatialSize):
        raise SlidingWindowGeometryError("output_extent must be a SpatialSize")
    if not isinstance(grid_size, SpatialSize):
        raise SlidingWindowGeometryError("grid_size must be a SpatialSize")
    _require_align_corners_true(align_corners)

    row_low, row_high, row_fraction = _axis_stencil(
        doubled_row, 2, output_extent.height, grid_size.height
    )
    col_low, col_high, col_fraction = _axis_stencil(
        doubled_col, 2, output_extent.width, grid_size.width
    )
    return _combine_axis_stencils(
        row_low, row_high, row_fraction, col_low, col_high, col_fraction
    )


__all__ = [
    "BilinearStencil",
    "PatchGridSpec",
    "PixelCoordinate",
    "Rectangle",
    "SlidingWindowGeometryError",
    "SlidingWindowPlan",
    "SpatialSize",
    "WindowGeometry",
    "bilinear_stencil",
    "bilinear_stencil_from_doubled",
    "contains_point",
    "global_to_local",
    "intersect_windows",
    "local_to_global",
    "overlap_area",
]
