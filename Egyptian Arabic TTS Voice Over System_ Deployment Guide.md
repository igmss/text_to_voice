# Egyptian Arabic TTS Voice Over System: Deployment Guide

**Version:** 1.0.0  
**Target Audience:** System Administrators, DevOps Engineers, Production Deployment  
**Last Updated:** June 2024  

## Overview

This deployment guide provides comprehensive instructions for deploying the Egyptian Arabic TTS Voice Over System in production environments. The guide covers various deployment scenarios including cloud hosting, on-premises installation, containerized deployment, and scaling considerations.

## Deployment Architecture Options

### Option 1: Single Server Deployment (Recommended for Small-Medium Scale)

**Architecture:**
```
┌─────────────────────────────────────────┐
│              Load Balancer              │
│            (Nginx/Apache)               │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│           Application Server            │
│  ┌─────────────────┬─────────────────┐  │
│  │   Frontend      │    Backend      │  │
│  │   (React)       │    (Flask)      │  │
│  │   Port: 3000    │   Port: 5000    │  │
│  └─────────────────┴─────────────────┘  │
│  ┌─────────────────────────────────────┐ │
│  │        File Storage                 │ │
│  │     (Generated Audio)               │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Specifications:**
- **CPU:** 8+ cores (Intel Xeon or AMD EPYC)
- **RAM:** 32GB minimum, 64GB recommended
- **Storage:** 500GB SSD for system, 2TB+ for audio files
- **Network:** 1Gbps connection
- **OS:** Ubuntu 20.04 LTS or CentOS 8

### Option 2: Microservices Deployment (Recommended for Large Scale)

**Architecture:**
```
┌─────────────────────────────────────────┐
│              Load Balancer              │
│            (Nginx + SSL)                │
└─────────┬───────────────┬───────────────┘
          │               │
┌─────────┴─────────┐   ┌─┴───────────────┐
│   Frontend        │   │   API Gateway   │
│   Service         │   │   Service       │
│   (React/Nginx)   │   │   (Flask)       │
└───────────────────┘   └─┬───────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
┌─────────┴─────────┐   ┌─┴─────────┐   ┌─┴─────────┐
│   TTS Engine      │   │  Audio    │   │  Quality  │
│   Service         │   │  Processor│   │  Evaluator│
│   (PyTorch)       │   │  Service  │   │  Service  │
└───────────────────┘   └───────────┘   └───────────┘
          │               │               │
┌─────────┴───────────────┴───────────────┴─────────┐
│              Shared Storage                       │
│         (Models, Audio Files, Cache)              │
└───────────────────────────────────────────────────┘
```

### Option 3: Cloud Deployment (AWS/Azure/GCP)

**AWS Architecture Example:**
```
┌─────────────────────────────────────────┐
│              CloudFront                 │
│            (CDN + SSL)                  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│         Application Load Balancer       │
└─────────┬───────────────┬───────────────┘
          │               │
┌─────────┴─────────┐   ┌─┴───────────────┐
│   EC2 Instance    │   │   EC2 Instance  │
│   (Frontend)      │   │   (Backend)     │
│   t3.large        │   │   c5.2xlarge    │
└───────────────────┘   └─┬───────────────┘
                          │
┌─────────────────────────┴───────────────┐
│              S3 Bucket                  │
│         (Audio Storage)                 │
└─────────────────────────────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Production Requirements:**
- **CPU:** 8 cores, 2.4GHz+
- **RAM:** 32GB
- **Storage:** 500GB SSD (system) + 2TB (data)
- **Network:** 100Mbps dedicated bandwidth
- **OS:** Ubuntu 20.04 LTS (recommended)

**Recommended Production Requirements:**
- **CPU:** 16+ cores, 3.0GHz+
- **RAM:** 64GB+
- **Storage:** 1TB NVMe SSD (system) + 5TB+ (data)
- **Network:** 1Gbps+ dedicated bandwidth
- **GPU:** NVIDIA Tesla T4 or better (optional, for acceleration)

### Software Dependencies

**Core Dependencies:**
```bash
# System packages
sudo apt update && sudo apt install -y \
    python3.8 python3.8-dev python3.8-venv \
    nodejs npm \
    nginx \
    git curl wget \
    build-essential \
    libsndfile1-dev \
    ffmpeg
```

**Python Environment:**
```bash
# Create virtual environment
python3.8 -m venv /opt/egyptian-tts/venv
source /opt/egyptian-tts/venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Node.js Environment:**
```bash
# Install Node.js 18 LTS
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install pnpm
npm install -g pnpm
```

## Step-by-Step Deployment

### Step 1: Server Preparation

1. **Create deployment user:**
```bash
sudo adduser egyptian-tts
sudo usermod -aG sudo egyptian-tts
sudo su - egyptian-tts
```

2. **Create directory structure:**
```bash
sudo mkdir -p /opt/egyptian-tts/{app,data,logs,backups}
sudo chown -R egyptian-tts:egyptian-tts /opt/egyptian-tts
```

3. **Configure firewall:**
```bash
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 3000  # Frontend (development)
sudo ufw allow 5000  # Backend API
sudo ufw enable
```

### Step 2: Application Deployment

1. **Clone and setup application:**
```bash
cd /opt/egyptian-tts/app
git clone https://github.com/egyptian-tts/egyptian-voice-studio.git .

# Setup backend
cd voice_api
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup frontend
cd ../web_app/egyptian-voice-studio
pnpm install
pnpm run build
```

2. **Configure environment variables:**
```bash
# Create environment file
cat > /opt/egyptian-tts/app/.env << EOF
# Production Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=$(openssl rand -hex 32)

# Audio Configuration
AUDIO_STORAGE_PATH=/opt/egyptian-tts/data/audio
MAX_AUDIO_FILE_SIZE=100MB
AUDIO_RETENTION_DAYS=30

# Performance Configuration
MAX_WORKERS=4
WORKER_TIMEOUT=300
MAX_BATCH_SIZE=50

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=/opt/egyptian-tts/logs/app.log

# Security Configuration
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
CORS_ORIGINS=https://your-domain.com
EOF
```

### Step 3: Database and Storage Setup

1. **Create storage directories:**
```bash
mkdir -p /opt/egyptian-tts/data/{audio,models,cache,temp}
mkdir -p /opt/egyptian-tts/logs
```

2. **Set permissions:**
```bash
chmod 755 /opt/egyptian-tts/data
chmod 755 /opt/egyptian-tts/logs
chmod 700 /opt/egyptian-tts/data/temp
```

3. **Configure log rotation:**
```bash
sudo cat > /etc/logrotate.d/egyptian-tts << EOF
/opt/egyptian-tts/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 egyptian-tts egyptian-tts
    postrotate
        systemctl reload egyptian-tts-api
    endscript
}
EOF
```

### Step 4: Service Configuration

1. **Create systemd service for backend:**
```bash
sudo cat > /etc/systemd/system/egyptian-tts-api.service << EOF
[Unit]
Description=Egyptian TTS API Service
After=network.target

[Service]
Type=simple
User=egyptian-tts
Group=egyptian-tts
WorkingDirectory=/opt/egyptian-tts/app/voice_api
Environment=PATH=/opt/egyptian-tts/app/voice_api/venv/bin
ExecStart=/opt/egyptian-tts/app/voice_api/venv/bin/python src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=egyptian-tts-api

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF
```

2. **Create systemd service for frontend (if serving with Node.js):**
```bash
sudo cat > /etc/systemd/system/egyptian-tts-frontend.service << EOF
[Unit]
Description=Egyptian TTS Frontend Service
After=network.target

[Service]
Type=simple
User=egyptian-tts
Group=egyptian-tts
WorkingDirectory=/opt/egyptian-tts/app/web_app/egyptian-voice-studio
Environment=NODE_ENV=production
Environment=PORT=3000
ExecStart=/usr/bin/npm run start
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=egyptian-tts-frontend

[Install]
WantedBy=multi-user.target
EOF
```

3. **Enable and start services:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable egyptian-tts-api
sudo systemctl enable egyptian-tts-frontend
sudo systemctl start egyptian-tts-api
sudo systemctl start egyptian-tts-frontend
```

### Step 5: Nginx Configuration

1. **Install and configure Nginx:**
```bash
sudo apt install nginx

# Create Nginx configuration
sudo cat > /etc/nginx/sites-available/egyptian-tts << EOF
# Rate limiting
limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone \$binary_remote_addr zone=upload:10m rate=1r/s;

# Upstream servers
upstream api_backend {
    server 127.0.0.1:5000;
    keepalive 32;
}

upstream frontend_backend {
    server 127.0.0.1:3000;
    keepalive 32;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # Client upload limits
    client_max_body_size 100M;
    client_body_timeout 60s;
    client_header_timeout 60s;
    
    # API routes
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Audio file serving
    location /audio/ {
        alias /opt/egyptian-tts/data/audio/;
        expires 1h;
        add_header Cache-Control "public, immutable";
        
        # Security
        location ~* \.(php|pl|py|jsp|asp|sh|cgi)\$ {
            deny all;
        }
    }
    
    # Frontend routes
    location / {
        proxy_pass http://frontend_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        
        # Fallback for React Router
        try_files \$uri \$uri/ /index.html;
    }
    
    # Static assets
    location /static/ {
        alias /opt/egyptian-tts/app/web_app/egyptian-voice-studio/build/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF
```

2. **Enable site and restart Nginx:**
```bash
sudo ln -s /etc/nginx/sites-available/egyptian-tts /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 6: SSL Certificate Setup

1. **Install Certbot:**
```bash
sudo apt install certbot python3-certbot-nginx
```

2. **Obtain SSL certificate:**
```bash
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

3. **Setup automatic renewal:**
```bash
sudo crontab -e
# Add this line:
0 12 * * * /usr/bin/certbot renew --quiet
```

## Monitoring and Maintenance

### Health Monitoring

1. **Create health check script:**
```bash
cat > /opt/egyptian-tts/scripts/health-check.sh << EOF
#!/bin/bash

# Check API health
API_STATUS=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/health)
if [ "\$API_STATUS" != "200" ]; then
    echo "API health check failed: \$API_STATUS"
    systemctl restart egyptian-tts-api
fi

# Check frontend health
FRONTEND_STATUS=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "\$FRONTEND_STATUS" != "200" ]; then
    echo "Frontend health check failed: \$FRONTEND_STATUS"
    systemctl restart egyptian-tts-frontend
fi

# Check disk space
DISK_USAGE=\$(df /opt/egyptian-tts/data | awk 'NR==2 {print \$5}' | sed 's/%//')
if [ "\$DISK_USAGE" -gt 85 ]; then
    echo "Disk usage high: \${DISK_USAGE}%"
    # Cleanup old audio files
    find /opt/egyptian-tts/data/audio -name "*.wav" -mtime +7 -delete
fi
EOF

chmod +x /opt/egyptian-tts/scripts/health-check.sh
```

2. **Setup monitoring cron job:**
```bash
crontab -e
# Add this line:
*/5 * * * * /opt/egyptian-tts/scripts/health-check.sh >> /opt/egyptian-tts/logs/health-check.log 2>&1
```

### Log Monitoring

1. **Setup log monitoring with journalctl:**
```bash
# Monitor API logs
sudo journalctl -u egyptian-tts-api -f

# Monitor frontend logs
sudo journalctl -u egyptian-tts-frontend -f

# Monitor Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Performance Monitoring

1. **Install monitoring tools:**
```bash
sudo apt install htop iotop nethogs
```

2. **Create performance monitoring script:**
```bash
cat > /opt/egyptian-tts/scripts/performance-monitor.sh << EOF
#!/bin/bash

LOG_FILE="/opt/egyptian-tts/logs/performance.log"
DATE=\$(date '+%Y-%m-%d %H:%M:%S')

# CPU and Memory usage
CPU_USAGE=\$(top -bn1 | grep "Cpu(s)" | awk '{print \$2}' | sed 's/%us,//')
MEM_USAGE=\$(free | grep Mem | awk '{printf "%.2f", \$3/\$2 * 100.0}')

# Disk usage
DISK_USAGE=\$(df /opt/egyptian-tts/data | awk 'NR==2 {print \$5}' | sed 's/%//')

# Network connections
CONNECTIONS=\$(netstat -an | grep :5000 | wc -l)

echo "\$DATE,\$CPU_USAGE,\$MEM_USAGE,\$DISK_USAGE,\$CONNECTIONS" >> \$LOG_FILE
EOF

chmod +x /opt/egyptian-tts/scripts/performance-monitor.sh

# Add to crontab
crontab -e
# Add this line:
*/1 * * * * /opt/egyptian-tts/scripts/performance-monitor.sh
```

## Backup and Recovery

### Automated Backup

1. **Create backup script:**
```bash
cat > /opt/egyptian-tts/scripts/backup.sh << EOF
#!/bin/bash

BACKUP_DIR="/opt/egyptian-tts/backups"
DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="egyptian-tts-backup-\$DATE"

# Create backup directory
mkdir -p "\$BACKUP_DIR/\$BACKUP_NAME"

# Backup application code
tar -czf "\$BACKUP_DIR/\$BACKUP_NAME/app.tar.gz" -C /opt/egyptian-tts app

# Backup configuration
cp -r /etc/nginx/sites-available/egyptian-tts "\$BACKUP_DIR/\$BACKUP_NAME/"
cp -r /etc/systemd/system/egyptian-tts-*.service "\$BACKUP_DIR/\$BACKUP_NAME/"

# Backup logs (last 7 days)
find /opt/egyptian-tts/logs -name "*.log" -mtime -7 -exec cp {} "\$BACKUP_DIR/\$BACKUP_NAME/" \;

# Backup models and important data
tar -czf "\$BACKUP_DIR/\$BACKUP_NAME/models.tar.gz" -C /opt/egyptian-tts/data models

# Remove backups older than 30 days
find \$BACKUP_DIR -name "egyptian-tts-backup-*" -mtime +30 -exec rm -rf {} \;

echo "Backup completed: \$BACKUP_NAME"
EOF

chmod +x /opt/egyptian-tts/scripts/backup.sh
```

2. **Schedule daily backups:**
```bash
crontab -e
# Add this line:
0 2 * * * /opt/egyptian-tts/scripts/backup.sh >> /opt/egyptian-tts/logs/backup.log 2>&1
```

### Recovery Procedures

1. **Application Recovery:**
```bash
# Stop services
sudo systemctl stop egyptian-tts-api egyptian-tts-frontend

# Restore from backup
cd /opt/egyptian-tts/backups/egyptian-tts-backup-YYYYMMDD_HHMMSS
tar -xzf app.tar.gz -C /opt/egyptian-tts/
tar -xzf models.tar.gz -C /opt/egyptian-tts/data/

# Restore configuration
sudo cp egyptian-tts /etc/nginx/sites-available/
sudo cp egyptian-tts-*.service /etc/systemd/system/

# Restart services
sudo systemctl daemon-reload
sudo systemctl start egyptian-tts-api egyptian-tts-frontend
```

## Scaling Considerations

### Horizontal Scaling

1. **Load Balancer Configuration:**
```nginx
upstream api_cluster {
    least_conn;
    server 10.0.1.10:5000 weight=3;
    server 10.0.1.11:5000 weight=3;
    server 10.0.1.12:5000 weight=2;
    keepalive 32;
}
```

2. **Shared Storage Setup:**
```bash
# NFS setup for shared audio storage
sudo apt install nfs-common
sudo mount -t nfs 10.0.1.100:/shared/audio /opt/egyptian-tts/data/audio
```

### Vertical Scaling

1. **Resource Optimization:**
```bash
# Increase worker processes
export FLASK_WORKERS=8
export FLASK_WORKER_TIMEOUT=600

# Optimize memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Auto-scaling (Cloud)

1. **AWS Auto Scaling Group:**
```json
{
  "AutoScalingGroupName": "egyptian-tts-asg",
  "MinSize": 2,
  "MaxSize": 10,
  "DesiredCapacity": 3,
  "TargetGroupARNs": ["arn:aws:elasticloadbalancing:..."],
  "HealthCheckType": "ELB",
  "HealthCheckGracePeriod": 300
}
```

## Security Hardening

### System Security

1. **Firewall Configuration:**
```bash
# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

2. **SSH Hardening:**
```bash
# Edit SSH config
sudo nano /etc/ssh/sshd_config

# Add these settings:
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
Port 2222  # Change default port
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
```

### Application Security

1. **Environment Variables:**
```bash
# Secure environment file
chmod 600 /opt/egyptian-tts/app/.env
chown egyptian-tts:egyptian-tts /opt/egyptian-tts/app/.env
```

2. **API Security:**
```python
# Add to Flask configuration
RATELIMIT_STORAGE_URL = "redis://localhost:6379"
RATELIMIT_DEFAULT = "100 per hour"
```

## Troubleshooting

### Common Issues

**Service Won't Start:**
```bash
# Check service status
sudo systemctl status egyptian-tts-api
sudo journalctl -u egyptian-tts-api -n 50

# Check permissions
ls -la /opt/egyptian-tts/app/
sudo chown -R egyptian-tts:egyptian-tts /opt/egyptian-tts/
```

**High Memory Usage:**
```bash
# Monitor memory usage
sudo htop
sudo ps aux --sort=-%mem | head

# Restart services if needed
sudo systemctl restart egyptian-tts-api
```

**Audio Generation Fails:**
```bash
# Check disk space
df -h /opt/egyptian-tts/data/

# Check audio directory permissions
ls -la /opt/egyptian-tts/data/audio/
sudo chown -R egyptian-tts:egyptian-tts /opt/egyptian-tts/data/
```

**SSL Certificate Issues:**
```bash
# Check certificate status
sudo certbot certificates

# Renew certificate
sudo certbot renew --dry-run
sudo certbot renew
```

This deployment guide provides comprehensive instructions for setting up the Egyptian Arabic TTS Voice Over System in production environments. Follow the steps carefully and adapt the configuration to your specific requirements and infrastructure.

