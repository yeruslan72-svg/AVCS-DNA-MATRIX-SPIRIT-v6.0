version: "3.9"

services:
  # =========================================================
  # 🧠 AVCS DNA-MATRIX SPIRIT — Main Orchestrator
  # =========================================================
  avcs-dna-matrix:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: avcs-dna-matrix
    ports:
      - "8501:8501"
    env_file:
      - .env
    environment:
      PYTHONUNBUFFERED: 1
      STREAMLIT_SERVER_PORT: 8501
      STREAMLIT_SERVER_ADDRESS: 0.0.0.0
      DIGITAL_TWIN_URL: http://digital-twin:8080
      MQTT_BROKER: mqtt-broker
      MQTT_PORT: 1883
    depends_on:
      - mqtt-broker
      - digital-twin
    volumes:
      - ./data:/app/data
      - ./assets:/app/assets
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # =========================================================
  # 🌐 Digital Twin Simulator — Industrial Process Sandbox
  # =========================================================
  digital-twin:
    image: python:3.10-slim
    container_name: digital-twin
    working_dir: /sim
    volumes:
      - ./digital_twin:/sim
    command: ["python", "industrial_digital_twin.py"]
    expose:
      - "8080"
    environment:
      SIMULATION_MODE: "true"
      UPDATE_INTERVAL: "5"
    restart: unless-stopped

  # =========================================================
  # 📡 MQTT Broker — Data & Event Bus
  # =========================================================
  mqtt-broker:
    image: eclipse-mosquitto:2.0
    container_name: mqtt-broker
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - ./data/mqtt:/mosquitto/data
      - ./data/mqtt/log:/mosquitto/log
    restart: unless-stopped

networks:
  default:
    name: avcs_network
    driver: bridge
