"""trackside - measure track & field events from a single camera, with error bars.

Two pipelines share one idea: a number without an uncertainty is not a
measurement.

    finish  - sprint finish timing, resolved below the frame interval, and
              honest about margins too close to call.
    flight  - field-event implement tracking, release parameters, and a
              predicted distance you can check against the tape measure.

Both end in an event stream rather than a wall of numbers, and both emit events
about their own reliability alongside events about the athlete.
"""

from .calib import IMPLEMENT_DIAMETER, REGULATION, PlaneCalibration
from .detect import Blob, coverage, detect_blobs, median_background
from .events import (
    Event,
    Zone,
    detect_foul,
    events_from_finish,
    events_from_flight,
    summarise,
)
from .finish import FinishResult, LaneResult, analyse_finish, photo_finish_strip
from .fit import DragFit, fit_drag_model
from .flight import FlightResult, analyse_flight
from .fuse import Fusion, fuse
from .measure import Measurement, monte_carlo
from .physics import IMPLEMENTS, Trajectory, integrate, optimal_angle, range_no_drag

__version__ = "0.2.0"

__all__ = [
    "PlaneCalibration", "REGULATION", "IMPLEMENT_DIAMETER",
    "Blob", "detect_blobs", "median_background", "coverage",
    "analyse_finish", "FinishResult", "LaneResult", "photo_finish_strip",
    "analyse_flight", "FlightResult",
    "fit_drag_model", "DragFit",
    "fuse", "Fusion",
    "Event", "Zone", "detect_foul", "events_from_finish", "events_from_flight",
    "summarise",
    "Measurement", "monte_carlo",
    "integrate", "range_no_drag", "optimal_angle", "Trajectory", "IMPLEMENTS",
]
