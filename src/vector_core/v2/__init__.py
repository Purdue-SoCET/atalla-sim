"""The M x N vector core (v2): full-width FU groups, S slots each wired to a
bank pair through its own crossbar, a banked register file and a multi-write
writeback. See docs/vector_core_v2_change_plan.md and
docs/vector_core_v2_worklog.md."""

from vector_core.v2.config import (  # noqa: F401
    DEFAULT_V2_GROUPS, FUGroupSpec, VectorCoreConfig, make_vector_core)
