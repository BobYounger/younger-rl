from .metrics import MetricLogger

# Backward compatibility for earlier imports.
Logger = MetricLogger

__all__ = ["Logger", "MetricLogger"]
