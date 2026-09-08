"""Edge transport shares its configuration and lifecycle with the server."""
import zenoh as zenoh  # Re-exported for diagnostics and existing adapters.
from dt_common.zenoh_transport import LatestPublisher, make_zenoh_config as make_zenoh_config


class ZenohSender(LatestPublisher):
    pass
