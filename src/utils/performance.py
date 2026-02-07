"""
Performance monitoring utility.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from loguru import logger


class PerformanceMonitor:
    """Monitor pipeline performance."""
    
    def __init__(self, enabled: bool = True):
        """Initialize performance monitor."""
        self.enabled = enabled
        self.timings = defaultdict(list)
        self.frame_times = []
        self.last_time = time.time()
        self.fps_window = 30
    
    @contextmanager
    def measure(self, name: str):
        """Context manager to measure execution time."""
        if not self.enabled:
            yield
            return
        
        start = time.time()
        try:
            yield
        finally:
            elapsed = (time.time() - start) * 1000  # ms
            self.timings[name].append(elapsed)
            
            # Keep only recent measurements
            if len(self.timings[name]) > 100:
                self.timings[name].pop(0)
    
    def tick(self):
        """Record frame time for FPS calculation."""
        now = time.time()
        frame_time = now - self.last_time
        self.last_time = now
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.fps_window:
            self.frame_times.pop(0)
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if not self.frame_times:
            return 0.0
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_stats(self) -> str:
        """Get performance statistics string."""
        if not self.enabled or not self.timings:
            return ""
        
        stats = []
        for name, times in self.timings.items():
            if times:
                avg = sum(times) / len(times)
                stats.append(f"{name}:{avg:.1f}ms")
        
        return " | ".join(stats)
    
    def get_detailed_stats(self) -> str:
        """Get detailed performance statistics."""
        if not self.enabled:
            return "Performance monitoring disabled"
        
        lines = ["Performance Statistics:", "=" * 50]
        
        # FPS
        lines.append(f"Average FPS: {self.get_fps():.1f}")
        lines.append("")
        
        # Timings
        lines.append("Module Timings (ms):")
        for name, times in sorted(self.timings.items()):
            if times:
                avg = sum(times) / len(times)
                min_t = min(times)
                max_t = max(times)
                lines.append(f"  {name:15s}: avg={avg:6.2f}  min={min_t:6.2f}  max={max_t:6.2f}")
        
        return "\n".join(lines)
