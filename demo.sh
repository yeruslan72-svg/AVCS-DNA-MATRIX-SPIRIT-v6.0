#!/bin/bash

# AVCS DNA-MATRIX SPIRIT v7.0 - Demo Launcher
# Quick start for demonstrations and pilot deployments

set -e  # Exit on any error

echo "🚀 AVCS DNA-MATRIX SPIRIT v7.0 - Demo Launcher"
echo "=============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available."
    exit 1
fi

# Function to check Docker Compose command
get_compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    elif docker compose version &> /dev/null; then
        echo "docker compose"
    else
        echo ""
    fi
}

COMPOSE_CMD=$(get_compose_cmd)
if [ -z "$COMPOSE_CMD" ]; then
    echo "❌ Cannot find Docker Compose command"
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Build and start services
echo ""
echo "📦 Building and starting services..."
$COMPOSE_CMD up -d --build

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ AVCS System is running!"
else
    echo "⚠️  System is starting... please wait a moment"
    sleep 10
fi

# Display access information
echo ""
echo "🎯 ACCESS INFORMATION:"
echo "   • AVCS Dashboard:  http://localhost:8501"
echo "   • API Documentation: http://localhost:8501/docs"
echo "   • System Health:   http://localhost:8501/health"
echo ""
echo "🔧 MANAGEMENT COMMANDS:"
echo "   • View logs:       $COMPOSE_CMD logs -f"
echo "   • Stop services:   $COMPOSE_CMD down"
echo "   • Restart:         $COMPOSE_CMD restart"
echo "   • Full cleanup:    $COMPOSE_CMD down -v"
echo ""
echo "📊 DEMO FEATURES:"
echo "   • Real-time vibration monitoring"
echo "   • AI-powered anomaly detection"
echo "   • Digital twin simulations"
echo "   • Predictive maintenance alerts"
echo ""
echo "💡 For technical support: engineering@avcs-systems.com"
echo "=============================================="
