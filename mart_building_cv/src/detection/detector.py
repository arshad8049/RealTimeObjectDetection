# src/detection/detector.py

import cv2
import yaml
from ultralytics import YOLO
import argparse
import json
import time
import paho.mqtt.client as mqtt
import os

# --- HELPER FUNCTION DEFINITIONS ---

def load_config(config_path=None):
    """
    Loads the configuration from a YAML file.
    It robustly finds the config relative to this script's location.
    """
    if config_path is None:
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(script_dir, '..', '..', 'configs', 'main_config.yaml')
        
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at the expected path '{config_path}'.")
        exit()
    except Exception as e:
        print(f"FATAL: Error reading the configuration file: {e}")
        exit()

def setup_mqtt_client(config):
    """Sets up and connects the MQTT client if enabled in the config."""
    if not config.get('mqtt', {}).get('enabled', False):
        print("INFO: MQTT is disabled in the configuration.")
        return None
        
    broker = config['mqtt']['broker_address']
    port = config['mqtt']['port']
    # Use the latest callback API version to avoid deprecation warnings
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(broker, port, 60)
        client.loop_start()
        print(f"INFO: Successfully connected to MQTT Broker at {broker}:{port}")
        return client
    except Exception as e:
        print(f"WARNING: Could not connect to MQTT Broker: {e}")
        return None

def publish_detection(mqtt_client, topic, detection_data):
    """Publishes detection data as a JSON payload to the MQTT broker."""
    if not mqtt_client:
        return
    try:
        payload = json.dumps(detection_data)
        mqtt_client.publish(topic, payload)
        print(f"Published to MQTT topic '{topic}': {payload}")
    except Exception as e:
        print(f"WARNING: Failed to publish to MQTT: {e}")

def is_image_file(filename):
    """Checks if a filename has a common image extension."""
    return str(filename).lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def process_detections(results, mqtt_client, config, last_detection_times, detection_topic):
    """
    Processes detection results, handles cooldown logic, and publishes to MQTT.
    Returns the annotated frame for display.
    """
    frame = results.plot()
    
    if not mqtt_client or not detection_topic:
        return frame

    current_time = time.time()
    cooldown = config['mqtt'].get('message_interval_seconds', 10)
    
    # Correctly and safely iterate through detected boxes by index
    boxes = results.boxes
    for i in range(len(boxes)):
        confidence = boxes.conf[i].item()
        class_id = int(boxes.cls[i].item())
        detected_class = results.names[class_id]

        last_time = last_detection_times.get(detected_class, 0)
        if current_time - last_time > cooldown:
            detection_data = {
                'timestamp': int(current_time),
                'building': config.get('device_context', {}).get('building', 'unknown'),
                'zone': config.get('device_context', {}).get('zone', 'unknown'),
                'detection_class': detected_class,
                'confidence': round(confidence, 4),
                'source': config.get('input_source', 'unknown')
            }
            publish_detection(mqtt_client, detection_topic, detection_data)
            last_detection_times[detected_class] = current_time
            
    return frame

def process_video_stream(model, source, conf, mqtt_client, config, detection_topic):
    """Handles processing of a live camera feed or a video file."""
    last_detection_times = {}
    results_stream = model.predict(source=source, conf=conf, stream=True, verbose=False)
    
    for results in results_stream:
        # Check if there are any detections before processing
        if len(results.boxes) > 0:
            annotated_frame = process_detections(results, mqtt_client, config, last_detection_times, detection_topic)
            cv2.imshow("SCOPE Detector", annotated_frame)
        else:
            # If no detections, just show the original, un-annotated frame
            cv2.imshow("SCOPE Detector", results.orig_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def process_single_image(model, source, conf, mqtt_client, config, detection_topic):
    """Handles processing of a single static image."""
    last_detection_times = {}
    results = model.predict(source=source, conf=conf, verbose=False)
    
    # The result from a non-stream predict is a list, so we take the first element
    result = results[0]
    
    if len(result.boxes) > 0:
        annotated_frame = process_detections(result, mqtt_client, config, last_detection_times, detection_topic)
        cv2.imshow("SCOPE Detector", annotated_frame)
    else:
        cv2.imshow("SCOPE Detector", result.orig_img)

    print("INFO: Detection complete on image. Press any key to exit.")
    cv2.waitKey(0)

# --- MAIN EXECUTION LOGIC ---

def main():
    parser = argparse.ArgumentParser(description="SCOPE: Smart Building Computer Vision Detector")
    parser.add_argument('--input', type=str, default=None, help='Path to input file or camera index. Overrides config.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection (e.g., 0.5).')
    args = parser.parse_args()

    config = load_config()
    input_source = args.input if args.input is not None else config.get('input_source', 0)
    config['input_source'] = input_source

    print(f"INFO: Processing source: {input_source}")
    print(f"INFO: Using confidence threshold: {args.conf}")

    # --- MQTT Setup ---
    mqtt_client = setup_mqtt_client(config)
    detection_topic = None
    if mqtt_client:
        context = config.get('device_context', {})
        building = context.get('building', 'unknown_building')
        zone = context.get('zone', 'unknown_zone')
        detection_topic = f"scope/detection/{building}/{zone}/event"
        print(f"INFO: Publishing to MQTT Topic: {detection_topic}")

    # --- Model Loading ---
    print("INFO: Loading YOLO-World model...")
    model = YOLO(config['yolo_model'])
    model.set_classes(config['detection_prompts'])
    print("INFO: Model loaded and classes set.")

    try:
        if is_image_file(input_source):
            process_single_image(model, input_source, args.conf, mqtt_client, config, detection_topic)
        else:
            source_for_stream = int(input_source) if str(input_source).isdigit() else input_source
            process_video_stream(model, source_for_stream, args.conf, mqtt_client, config, detection_topic)

    except Exception as e:
        print(f"FATAL: An unexpected error occurred during detection: {e}")
    finally:
        cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()
            print("INFO: MQTT client disconnected.")
        print("INFO: Detection finished and resources released.")

if __name__ == "__main__":
    main()
