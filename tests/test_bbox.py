"""
Test bounding box format conversions.
"""

import pytest
import numpy as np


def xywh_to_xyxy(bbox):
    """Convert xywh (top-left) to xyxy format."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox):
    """Convert xyxy to xywh (top-left) format."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


class TestBboxConversion:
    """Test bounding box conversions."""
    
    def test_xywh_to_xyxy(self):
        """Test xywh to xyxy conversion."""
        bbox_xywh = [100, 200, 50, 30]
        bbox_xyxy = xywh_to_xyxy(bbox_xywh)
        
        assert bbox_xyxy == [100, 200, 150, 230]
    
    def test_xyxy_to_xywh(self):
        """Test xyxy to xywh conversion."""
        bbox_xyxy = [100, 200, 150, 230]
        bbox_xywh = xyxy_to_xywh(bbox_xyxy)
        
        assert bbox_xywh == [100, 200, 50, 30]
    
    def test_round_trip(self):
        """Test round-trip conversion."""
        bbox_original = [100, 200, 50, 30]
        bbox_converted = xyxy_to_xywh(xywh_to_xyxy(bbox_original))
        
        assert bbox_converted == bbox_original
    
    def test_zero_size(self):
        """Test zero-size bbox."""
        bbox_xywh = [100, 200, 0, 0]
        bbox_xyxy = xywh_to_xyxy(bbox_xywh)
        
        assert bbox_xyxy == [100, 200, 100, 200]
    
    def test_floating_point(self):
        """Test floating point coordinates."""
        bbox_xywh = [100.5, 200.3, 50.7, 30.2]
        bbox_xyxy = xywh_to_xyxy(bbox_xywh)
        
        expected = [100.5, 200.3, 151.2, 230.5]
        assert all(abs(a - b) < 0.01 for a, b in zip(bbox_xyxy, expected))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
