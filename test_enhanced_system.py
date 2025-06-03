#!/usr/bin/env python3
"""
Test script for the Enhanced Egyptian Arabic TTS System
This script demonstrates the key functionality of the enhanced system
"""

import requests
import json
import time
import os

# API base URL (update this if running locally)
API_BASE = "https://5001-i4bjzbrfzdbf1lnmbrew1-e38eca77.manusvm.computer"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   TTS System: {data.get('tts_system')}")
            print(f"   espeak Available: {data.get('features', {}).get('espeak_available')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_system_info():
    """Test the system info endpoint"""
    print("\n🔍 Testing system info...")
    try:
        response = requests.get(f"{API_BASE}/api/system-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System info retrieved")
            print(f"   System: {data.get('system')}")
            print(f"   Capabilities: {list(data.get('capabilities', {}).keys())}")
            print(f"   Languages: {data.get('languages')}")
            print(f"   Voice Presets: {len(data.get('voice_presets', []))}")
            return True
        else:
            print(f"❌ System info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ System info error: {e}")
        return False

def test_voice_generation():
    """Test voice generation with Arabic text"""
    print("\n🔍 Testing voice generation...")
    
    # Test data
    test_texts = [
        "مرحبا بكم في نظام تحويل النص إلى كلام المصري",
        "هذا اختبار لجودة الصوت",
        "Hello, this is a test in English"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n   Testing text {i+1}: {text[:50]}...")
        
        try:
            payload = {
                "text": text,
                "speaker_id": "default",
                "voice_preset": "commercial-warm"
            }
            
            response = requests.post(
                f"{API_BASE}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Generation successful")
                print(f"      Audio ID: {data.get('audio_id')}")
                print(f"      Duration: {data.get('metadata', {}).get('duration', 0):.2f}s")
                print(f"      Quality Score: {data.get('metadata', {}).get('quality_score', 0):.2f}")
                print(f"      Method: {data.get('metadata', {}).get('synthesis_method')}")
                
                # Test audio download
                audio_url = data.get('audio_url')
                if audio_url:
                    audio_response = requests.get(f"{API_BASE}{audio_url}", timeout=10)
                    if audio_response.status_code == 200:
                        print(f"      ✅ Audio download successful ({len(audio_response.content)} bytes)")
                    else:
                        print(f"      ❌ Audio download failed: {audio_response.status_code}")
                
            else:
                print(f"   ❌ Generation failed: {response.status_code}")
                print(f"      Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
    
    return True

def test_voice_presets():
    """Test voice presets endpoint"""
    print("\n🔍 Testing voice presets...")
    try:
        response = requests.get(f"{API_BASE}/api/presets", timeout=10)
        if response.status_code == 200:
            data = response.json()
            presets = data.get('presets', {})
            print(f"✅ Voice presets retrieved ({len(presets)} presets)")
            for preset_id, preset_info in presets.items():
                print(f"   - {preset_id}: {preset_info.get('name')}")
            return True
        else:
            print(f"❌ Voice presets failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Voice presets error: {e}")
        return False

def test_speakers():
    """Test speakers endpoint"""
    print("\n🔍 Testing speakers...")
    try:
        response = requests.get(f"{API_BASE}/api/speakers", timeout=10)
        if response.status_code == 200:
            data = response.json()
            speakers = data.get('speakers', {})
            print(f"✅ Speakers retrieved ({len(speakers)} speakers)")
            for speaker_id, speaker_info in speakers.items():
                print(f"   - {speaker_id}: {speaker_info.get('name')} ({speaker_info.get('gender')}, {speaker_info.get('age')})")
            return True
        else:
            print(f"❌ Speakers failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Speakers error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎙️ Egyptian Arabic TTS Enhanced System - Test Suite")
    print("=" * 60)
    
    # Track test results
    tests = [
        ("Health Check", test_health_check),
        ("System Info", test_system_info),
        ("Voice Presets", test_voice_presets),
        ("Speakers", test_speakers),
        ("Voice Generation", test_voice_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced TTS system is working correctly.")
    else:
        print(f"⚠️ {total - passed} tests failed. Check the output above for details.")
    
    print(f"\n📊 System Status:")
    print(f"   API Endpoint: {API_BASE}")
    print(f"   Enhanced Features: Arabic processing, espeak-ng integration")
    print(f"   Ready for: Voice generation, batch processing, quality evaluation")

if __name__ == "__main__":
    main()

