#!/bin/bash
# Quick Test Script for Egyptian Arabic TTS API
# This script tests the main functionality of the enhanced TTS system

API_URL="https://5001-i4bjzbrfzdbf1lnmbrew1-e38eca77.manusvm.computer"

echo "🎙️ Testing Egyptian Arabic TTS Enhanced System"
echo "=============================================="
echo "API URL: $API_URL"
echo ""

# Test 1: Health Check
echo "1️⃣ Testing Health Check..."
curl -s "$API_URL/api/health" | python3 -m json.tool
echo ""

# Test 2: System Info
echo "2️⃣ Testing System Info..."
curl -s "$API_URL/api/system-info" | python3 -m json.tool
echo ""

# Test 3: Voice Presets
echo "3️⃣ Testing Voice Presets..."
curl -s "$API_URL/api/presets" | python3 -m json.tool
echo ""

# Test 4: Speakers
echo "4️⃣ Testing Speakers..."
curl -s "$API_URL/api/speakers" | python3 -m json.tool
echo ""

# Test 5: Generate Voice (Arabic)
echo "5️⃣ Testing Voice Generation (Arabic)..."
curl -s -X POST "$API_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "مرحبا بكم في نظام تحويل النص إلى كلام المصري",
    "speaker_id": "default",
    "voice_preset": "commercial-warm"
  }' | python3 -m json.tool
echo ""

# Test 6: Generate Voice (English)
echo "6️⃣ Testing Voice Generation (English)..."
curl -s -X POST "$API_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the Egyptian Arabic TTS system",
    "speaker_id": "default",
    "voice_preset": "news-professional"
  }' | python3 -m json.tool
echo ""

echo "✅ Test completed!"
echo ""
echo "💡 To download audio files:"
echo "   1. Copy the 'audio_url' from the generation response"
echo "   2. Use: curl -O '$API_URL/api/audio/AUDIO_ID'"
echo ""
echo "🌐 You can also test in your browser:"
echo "   Health: $API_URL/api/health"
echo "   System: $API_URL/api/system-info"

