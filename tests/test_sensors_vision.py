import sensors
import fruit
import vision


def test_sensor_moisture_dry_and_low_light():
    summary = sensors.interpret_sensor_readings({"moisture": 20, "temperature": 22, "light": 100})
    assert "dry" in summary.lower()
    assert "low light" in summary.lower()


def test_sensor_overwatered_and_heat():
    summary = sensors.interpret_sensor_readings({"moisture": 90, "temperature": 35, "light": 800})
    assert "overwatered" in summary.lower()
    assert "high temperature stress" in summary.lower()


def test_sensor_optimal():
    summary = sensors.interpret_sensor_readings({"moisture": 50, "temperature": 24, "light": 500})
    assert "optimal" in summary.lower()


def test_vision_yellow_and_damage():
    dets = [
        {"class_name": "yellow_leaf", "confidence": 0.9},
        {"class_name": "damaged_leaf", "confidence": 0.8},
    ]
    summary = vision.interpret_detections(dets, conf_threshold=0.25)
    assert "discoloration" in summary.lower()
    assert "damage" in summary.lower() or "pests" in summary.lower()


def test_vision_healthy_only():
    dets = [{"class_name": "healthy_leaf", "confidence": 0.9}]
    summary = vision.interpret_detections(dets, conf_threshold=0.25)
    assert "healthy" in summary.lower()


def test_fruit_summary_counts():
    dets = [
        {"class_name": "b_green", "confidence": 0.8},
        {"class_name": "l_half_ripened", "confidence": 0.9},
        {"class_name": "fruit_fully_ripened", "confidence": 0.95},
    ]
    summary = fruit.interpret_detections(dets, conf_threshold=0.25).lower()
    assert "detected 3 tomatoes" in summary
    assert "green" in summary
    assert "half-ripened" in summary
    assert "fully ripened" in summary
