#!/bin/bash

# Production Deployment Script for Security Monitoring System
# Optimized for Jetson Orin Nano Super

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${1:-production}"

echo -e "${GREEN}Security Monitoring System - Deployment Script${NC}"
echo -e "${GREEN}================================================${NC}"
echo "Environment: $DEPLOYMENT_ENV"
echo "Project Root: $PROJECT_ROOT"

# Check if running on Jetson
check_jetson() {
    if [ -f /etc/nv_tegra_release ]; then
        echo -e "${GREEN}✓ Running on NVIDIA Jetson${NC}"
        
        # Enable maximum performance mode
        sudo nvpmodel -m 0
        sudo jetson_clocks
        
        echo -e "${GREEN}✓ Jetson performance mode enabled${NC}"
    else
        echo -e "${YELLOW}⚠ Not running on Jetson - some optimizations will be skipped${NC}"
    fi
}

# Check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker installed${NC}"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}✗ Docker Compose not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker Compose installed${NC}"
    
    # Check NVIDIA Docker runtime (for GPU support)
    if docker info 2>/dev/null | grep -q nvidia; then
        echo -e "${GREEN}✓ NVIDIA Docker runtime available${NC}"
    else
        echo -e "${YELLOW}⚠ NVIDIA Docker runtime not found - GPU acceleration disabled${NC}"
    fi
    
    # Check camera device
    if [ -e /dev/video0 ]; then
        echo -e "${GREEN}✓ Camera device found (/dev/video0)${NC}"
    else
        echo -e "${YELLOW}⚠ No camera device found - video feed will not work${NC}"
    fi
}

# Create necessary directories
setup_directories() {
    echo -e "\n${YELLOW}Setting up directories...${NC}"
    
    mkdir -p "$PROJECT_ROOT"/{storage,logs,backups,config}
    mkdir -p "$PROJECT_ROOT"/storage/{videos,snapshots,exports}
    
    # Set permissions
    chmod 755 "$PROJECT_ROOT"/{storage,logs,backups}
    
    echo -e "${GREEN}✓ Directories created${NC}"
}

# Generate secure passwords and keys
generate_secrets() {
    echo -e "\n${YELLOW}Generating secrets...${NC}"
    
    ENV_FILE="$PROJECT_ROOT/.env"
    
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# Auto-generated secrets - Change these in production!
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
SECRET_KEY=$(openssl rand -base64 64)
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# API Keys (add your own)
ROBOFLOW_API_KEY=
WEBHOOK_URL=
PERSON_WEBHOOK_URL=
DOOR_WEBHOOK_URL=

# Environment
ENVIRONMENT=$DEPLOYMENT_ENV
EOF
        chmod 600 "$ENV_FILE"
        echo -e "${GREEN}✓ Secrets generated in .env file${NC}"
    else
        echo -e "${YELLOW}⚠ .env file already exists - skipping${NC}"
    fi
}

# Build Docker images
build_images() {
    echo -e "\n${YELLOW}Building Docker images...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Build backend
    echo "Building backend image..."
    docker build -f docker/Dockerfile.backend -t security-backend:latest .
    
    # Build frontend
    echo "Building frontend image..."
    docker build -f docker/Dockerfile.frontend -t security-frontend:latest .
    
    echo -e "${GREEN}✓ Docker images built${NC}"
}

# Initialize database
init_database() {
    echo -e "\n${YELLOW}Initializing database...${NC}"
    
    cd "$PROJECT_ROOT/docker"
    
    # Start only PostgreSQL
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Run migrations (if needed)
    # docker-compose run --rm backend alembic upgrade head
    
    echo -e "${GREEN}✓ Database initialized${NC}"
}

# Start services
start_services() {
    echo -e "\n${YELLOW}Starting services...${NC}"
    
    cd "$PROJECT_ROOT/docker"
    
    # Start all services
    docker-compose up -d
    
    # Wait for services to be ready
    echo "Waiting for services to start..."
    sleep 15
    
    # Check service health
    if docker-compose ps | grep -q "unhealthy"; then
        echo -e "${RED}✗ Some services are unhealthy${NC}"
        docker-compose ps
        exit 1
    fi
    
    echo -e "${GREEN}✓ All services started${NC}"
}

# Configure firewall
configure_firewall() {
    echo -e "\n${YELLOW}Configuring firewall...${NC}"
    
    if command -v ufw &> /dev/null; then
        sudo ufw allow 80/tcp    # HTTP
        sudo ufw allow 443/tcp   # HTTPS
        sudo ufw allow 8000/tcp  # API
        sudo ufw allow 3000/tcp  # Frontend (dev)
        sudo ufw allow 9090/tcp  # Prometheus
        sudo ufw allow 3001/tcp  # Grafana
        
        echo -e "${GREEN}✓ Firewall configured${NC}"
    else
        echo -e "${YELLOW}⚠ UFW not installed - skipping firewall configuration${NC}"
    fi
}

# Setup monitoring
setup_monitoring() {
    echo -e "\n${YELLOW}Setting up monitoring...${NC}"
    
    # Create Prometheus configuration
    cat > "$PROJECT_ROOT/docker/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
EOF
    
    echo -e "${GREEN}✓ Monitoring configured${NC}"
}

# Setup automatic backups
setup_backups() {
    echo -e "\n${YELLOW}Setting up automatic backups...${NC}"
    
    # Create backup script
    cat > "$PROJECT_ROOT/docker/backup.sh" << 'EOF'
#!/bin/sh
BACKUP_DIR=/backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.sql.gz"

# Create backup
pg_dump -h $PGHOST -U $PGUSER -d $PGDATABASE | gzip > $BACKUP_FILE

# Keep only last 7 days of backups
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
EOF
    
    chmod +x "$PROJECT_ROOT/docker/backup.sh"
    
    # Add cron job for daily backups
    if ! crontab -l 2>/dev/null | grep -q "backup.sh"; then
        (crontab -l 2>/dev/null; echo "0 2 * * * $PROJECT_ROOT/docker/backup.sh") | crontab -
        echo -e "${GREEN}✓ Automatic backups configured (daily at 2 AM)${NC}"
    else
        echo -e "${YELLOW}⚠ Backup cron job already exists${NC}"
    fi
}

# Print access information
print_access_info() {
    echo -e "\n${GREEN}================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}================================${NC}\n"
    
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    
    echo "Access URLs:"
    echo -e "  Dashboard:   ${GREEN}http://$LOCAL_IP${NC}"
    echo -e "  API:         ${GREEN}http://$LOCAL_IP:8000${NC}"
    echo -e "  API Docs:    ${GREEN}http://$LOCAL_IP:8000/docs${NC}"
    echo -e "  Prometheus:  ${GREEN}http://$LOCAL_IP:9090${NC}"
    echo -e "  Grafana:     ${GREEN}http://$LOCAL_IP:3001${NC}"
    echo ""
    echo "Default Credentials:"
    echo "  Grafana: admin / (check .env file)"
    echo ""
    echo "Commands:"
    echo "  View logs:     docker-compose -f $PROJECT_ROOT/docker/docker-compose.yml logs -f"
    echo "  Stop services: docker-compose -f $PROJECT_ROOT/docker/docker-compose.yml down"
    echo "  Restart:       docker-compose -f $PROJECT_ROOT/docker/docker-compose.yml restart"
    echo ""
    echo -e "${YELLOW}⚠ Remember to:${NC}"
    echo "  1. Change default passwords in .env file"
    echo "  2. Configure SSL certificates for HTTPS"
    echo "  3. Set up proper firewall rules"
    echo "  4. Configure webhook URLs in .env file"
}

# Main execution
main() {
    echo -e "${YELLOW}Starting deployment...${NC}\n"
    
    check_jetson
    check_prerequisites
    setup_directories
    generate_secrets
    build_images
    init_database
    setup_monitoring
    setup_backups
    start_services
    configure_firewall
    
    print_access_info
}

# Handle errors
trap 'echo -e "${RED}Deployment failed!${NC}"; exit 1' ERR

# Run main function
main

exit 0