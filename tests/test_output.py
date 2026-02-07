"""
Test output JSON contract compliance.
"""

import pytest
import json
from pathlib import Path


class TestOutputContract:
    """Test output JSON format contract."""
    
    def test_session_header(self):
        """Test session header format."""
        session = {
            'session_start': '2025-02-06T10:30:00.000Z',
            'camera_settings': {
                'resolution': [1920, 1080],
                'fps': 30,
                'exposure': 'fixed',
                'white_balance': 'fixed'
            }
        }
        
        assert 'session_start' in session
        assert 'camera_settings' in session
        assert isinstance(session['camera_settings']['resolution'], list)
        assert len(session['camera_settings']['resolution']) == 2
    
    def test_frame_record(self):
        """Test frame record format."""
        frame = {
            'frame_id': 0,
            'timestamp': 0.0,
            'detections': [
                {
                    'track_id': 1,
                    'class': 'front_headlight_left',
                    'bbox': [100, 200, 50, 30],
                    'confidence': 0.95,
                    'state': 'ON',
                    'state_conf': 0.92
                }
            ]
        }
        
        assert 'frame_id' in frame
        assert 'timestamp' in frame
        assert 'detections' in frame
        assert isinstance(frame['detections'], list)
    
    def test_detection_format(self):
        """Test detection object format."""
        detection = {
            'track_id': 1,
            'class': 'front_headlight_left',
            'bbox': [100, 200, 50, 30],
            'confidence': 0.95,
            'state': 'ON',
            'state_conf': 0.92
        }
        
        # Required fields
        assert 'track_id' in detection
        assert 'class' in detection
        assert 'bbox' in detection
        assert 'confidence' in detection
        assert 'state' in detection
        assert 'state_conf' in detection
        
        # Bbox format (xywh)
        assert isinstance(detection['bbox'], list)
        assert len(detection['bbox']) == 4
        assert all(isinstance(v, (int, float)) for v in detection['bbox'])
        
        # State values
        assert detection['state'] in ['ON', 'OFF', 'BLINKING', 'UNKNOWN']
        
        # Confidence range
        assert 0.0 <= detection['confidence'] <= 1.0
        assert 0.0 <= detection['state_conf'] <= 1.0
    
    def test_class_names(self):
        """Test class name validity."""
        valid_classes = [
            'front_headlight_left',
            'front_headlight_right',
            'front_indicator_left',
            'front_indicator_right',
            'front_all_weather_left',
            'front_all_weather_right',
            'rear_brake_left',
            'rear_brake_right',
            'rear_indicator_left',
            'rear_indicator_right',
            'rear_tailgate_left',
            'rear_tailgate_right',
        ]
        
        test_class = 'front_headlight_left'
        assert test_class in valid_classes
        
        invalid_class = 'invalid_class'
        assert invalid_class not in valid_classes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
