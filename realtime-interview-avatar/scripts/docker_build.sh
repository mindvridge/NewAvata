#!/bin/bash
# ============================================================================
# Docker Build Script for Realtime Interview Avatar
# ============================================================================

set -e

echo "=============================================="
echo "Building Realtime Interview Avatar Docker Image"
echo "=============================================="

# Change to project root
cd "$(dirname "$0")/.."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check if NVIDIA Container Toolkit is available
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "Warning: NVIDIA Container Toolkit may not be installed"
    echo "GPU support may not work. Install from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Build the image
echo ""
echo "Building Docker image..."
docker build \
    -f docker/Dockerfile.realtime \
    -t interview-avatar-realtime:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Image: interview-avatar-realtime:latest"
echo ""
echo "Next steps:"
echo "1. Create .env file with API keys"
echo "2. Run: docker-compose -f docker-compose.realtime.yml up"
echo ""
