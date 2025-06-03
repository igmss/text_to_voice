# Egyptian Arabic TTS - Production Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Egyptian Arabic Text-to-Speech system to production using AWS Amplify for the frontend and a cloud provider for the backend.

## Architecture

- **Frontend**: React application deployed on AWS Amplify
- **Backend**: Flask API server deployed on cloud infrastructure
- **TTS Engine**: Enhanced voice synthesizer with espeak-ng integration
- **Audio Processing**: Real-time voice generation and quality evaluation

## Prerequisites

- AWS Account with Amplify access
- GitHub repository with the project code
- Domain name (optional, for custom domains)
- Cloud provider account for backend (AWS, Google Cloud, or DigitalOcean)

## Deployment Steps

### Phase 1: Frontend Deployment (AWS Amplify)

#### 1.1 Prepare the Repository
```bash
# Ensure your code is pushed to GitHub
git add .
git commit -m "Prepare for production deployment"
git push origin main
```

#### 1.2 Create Amplify App
1. Go to AWS Amplify Console
2. Click "New app" > "Host web app"
3. Connect your GitHub repository
4. Select the repository and branch (main)
5. Configure build settings:
   - Build command: `npm run build`
   - Base directory: `egyptian-tts-frontend`
   - Publish directory: `egyptian-tts-frontend/dist`

#### 1.3 Configure Environment Variables
In Amplify Console > App Settings > Environment Variables, add:
```
VITE_API_BASE_URL=https://your-backend-api-url.com
VITE_APP_TITLE=Egyptian Arabic TTS - Professional Voice Over System
VITE_APP_VERSION=2.0.0
```

#### 1.4 Deploy Frontend
1. Click "Save and deploy"
2. Wait for build to complete (5-10 minutes)
3. Note the Amplify app URL (e.g., https://main.d1234567890.amplifyapp.com)

### Phase 2: Backend Deployment

#### 2.1 Choose Backend Platform

**Option A: AWS Elastic Beanstalk**
- Best for AWS integration
- Automatic scaling
- Easy deployment

**Option B: Google Cloud Run**
- Serverless container deployment
- Pay-per-use pricing
- Good for variable traffic

**Option C: DigitalOcean App Platform**
- Simple deployment
- Predictable pricing
- Good for small to medium scale

#### 2.2 Prepare Backend for Deployment

Create `Dockerfile` for containerized deployment:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=5000

# Run the application
CMD ["python", "src/main.py"]
```

#### 2.3 Deploy Backend

**For AWS Elastic Beanstalk:**
1. Install EB CLI: `pip install awsebcli`
2. Initialize: `eb init`
3. Create environment: `eb create production`
4. Deploy: `eb deploy`

**For Google Cloud Run:**
1. Build container: `gcloud builds submit --tag gcr.io/PROJECT-ID/tts-backend`
2. Deploy: `gcloud run deploy --image gcr.io/PROJECT-ID/tts-backend --platform managed`

**For DigitalOcean:**
1. Create new app in DigitalOcean App Platform
2. Connect GitHub repository
3. Configure build settings for Python/Flask
4. Deploy

#### 2.4 Configure Backend Environment
Set these environment variables in your backend platform:
```
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key
CORS_ORIGINS=https://your-amplify-app-url.amplifyapp.com
PORT=5000
```

### Phase 3: Connect Frontend and Backend

#### 3.1 Update Frontend Configuration
1. Go to Amplify Console > Environment Variables
2. Update `VITE_API_BASE_URL` with your backend URL
3. Redeploy the frontend

#### 3.2 Test the Connection
1. Open your Amplify app URL
2. Go to Settings tab
3. Test API connection
4. Try generating a voice over

### Phase 4: Production Optimization

#### 4.1 Enable HTTPS
- Amplify automatically provides HTTPS
- Ensure backend also uses HTTPS
- Update CORS settings if needed

#### 4.2 Configure Custom Domain (Optional)
1. In Amplify Console > Domain management
2. Add your custom domain
3. Configure DNS settings
4. Wait for SSL certificate provisioning

#### 4.3 Set Up Monitoring
- Enable CloudWatch logs for backend
- Set up error tracking (Sentry)
- Configure performance monitoring

#### 4.4 Optimize Performance
- Enable CDN for static assets
- Configure caching headers
- Optimize audio file delivery

## Security Considerations

### 4.1 API Security
- Use HTTPS only
- Implement rate limiting
- Validate all inputs
- Use secure secret keys

### 4.2 CORS Configuration
```python
CORS(app, origins=[
    "https://your-domain.com",
    "https://your-app.amplifyapp.com"
])
```

### 4.3 Environment Variables
- Never commit secrets to Git
- Use platform-specific secret management
- Rotate keys regularly

## Monitoring and Maintenance

### 5.1 Health Checks
- Monitor `/api/health` endpoint
- Set up automated alerts
- Check TTS system status

### 5.2 Logging
- Enable application logs
- Monitor error rates
- Track API usage

### 5.3 Backups
- Regular code backups via Git
- Monitor temporary file cleanup
- Database backups (if using database)

## Troubleshooting

### Common Issues

**Frontend Build Fails:**
- Check Node.js version compatibility
- Verify all dependencies are installed
- Check for TypeScript errors

**Backend Deployment Fails:**
- Verify Python version (3.11+)
- Check system dependencies (espeak-ng)
- Validate requirements.txt

**API Connection Issues:**
- Verify CORS configuration
- Check environment variables
- Test backend health endpoint

**TTS Generation Fails:**
- Ensure espeak-ng is installed
- Check temporary directory permissions
- Verify audio processing pipeline

### Support Resources

- AWS Amplify Documentation
- Flask Deployment Guides
- TTS System Logs
- GitHub Issues

## Cost Optimization

### Frontend (Amplify)
- Free tier: 1000 build minutes/month
- Pay-per-use for additional builds
- CDN included

### Backend
- Choose appropriate instance size
- Use auto-scaling for variable traffic
- Monitor resource usage

### Storage
- Clean up temporary audio files
- Use CDN for static assets
- Optimize audio file sizes

## Scaling Considerations

### Horizontal Scaling
- Load balancer for multiple backend instances
- Shared storage for audio files
- Database for session management

### Performance Optimization
- Cache frequently used voice presets
- Optimize audio generation pipeline
- Use async processing for batch operations

## Conclusion

This deployment guide provides a complete production setup for the Egyptian Arabic TTS system. Follow the steps carefully and test thoroughly before going live. Monitor the system closely during the first few days of production use.

For additional support or questions, refer to the project documentation or create an issue in the GitHub repository.

