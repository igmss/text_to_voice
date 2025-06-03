"""
Testing Framework for Egyptian Arabic TTS Voice Over System

This module provides comprehensive testing capabilities for validating
Egyptian Arabic TTS voice over quality across different scenarios.
"""

import unittest
import numpy as np
import torch
import librosa
import soundfile as sf
import json
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

# Import system components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.egyptian_tts import EgyptianArabicTTS, create_model_config
from preprocessing.text_processor import EgyptianArabicProcessor
from preprocessing.audio_processor import AudioProcessor
from evaluation.metrics import EgyptianTTSEvaluator
from inference.voice_synthesizer import VoiceSynthesizer


class VoiceOverTestSuite:
    """
    Comprehensive test suite for Egyptian Arabic voice over system.
    Tests quality, accuracy, and performance across various scenarios.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize test suite.
        
        Args:
            config_path: Path to test configuration file
        """
        self.config = self.load_test_config(config_path)
        self.results = []
        self.test_data_dir = self.config.get('test_data_dir', './test_data')
        
        # Initialize components
        self.synthesizer = VoiceSynthesizer()
        self.evaluator = EgyptianTTSEvaluator({'sample_rate': 48000})
        self.audio_processor = AudioProcessor(target_sr=48000, target_bit_depth=24)
        
        # Test scenarios
        self.test_scenarios = self.create_test_scenarios()
        
        # Quality thresholds
        self.quality_thresholds = {
            'minimum_acceptable': 0.7,
            'good_quality': 0.8,
            'excellent_quality': 0.9
        }
    
    def load_test_config(self, config_path: str) -> Dict:
        """Load test configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'test_data_dir': './test_data',
            'output_dir': './test_results',
            'voice_over_standards': {
                'sample_rate': 48000,
                'bit_depth': 24,
                'min_snr_db': 40,
                'max_clipping_percent': 0.1,
                'min_dynamic_range_db': 30
            },
            'test_scenarios': [
                'commercial_short',
                'commercial_long',
                'educational',
                'documentary',
                'audiobook',
                'news'
            ]
        }
    
    def create_test_scenarios(self) -> Dict[str, Dict]:
        """Create test scenarios for different voice over use cases."""
        return {
            'commercial_short': {
                'name': 'Short Commercial',
                'texts': [
                    'منتج جديد ومميز في السوق المصري',
                    'اشتري دلوقتي واستفيد من العرض',
                    'جودة عالية وسعر مناسب للجميع'
                ],
                'voice_preset': 'Commercial - Energetic',
                'expected_duration_range': (2, 8),
                'quality_threshold': 0.85
            },
            'commercial_long': {
                'name': 'Long Commercial',
                'texts': [
                    'في عالم مليان بالتحديات والفرص الجديدة، احنا بنقدملك الحل الأمثل اللي هيغير حياتك للأحسن. منتجنا الجديد مش بس هيوفرلك الوقت والجهد، ده كمان هيديك النتايج اللي كنت بتحلم بيها من زمان.',
                ],
                'voice_preset': 'Commercial - Warm',
                'expected_duration_range': (15, 30),
                'quality_threshold': 0.8
            },
            'educational': {
                'name': 'Educational Content',
                'texts': [
                    'في الدرس ده هنتعلم مع بعض أساسيات اللغة العربية',
                    'المعلومة دي مهمة جداً وهتساعدكم في فهم الموضوع',
                    'خلونا نراجع النقاط المهمة اللي اتكلمنا عنها'
                ],
                'voice_preset': 'Educational - Clear',
                'expected_duration_range': (5, 15),
                'quality_threshold': 0.85
            },
            'documentary': {
                'name': 'Documentary Narration',
                'texts': [
                    'في قلب الصحراء المصرية، تكمن أسرار حضارة عريقة امتدت لآلاف السنين',
                    'الاكتشافات الأثرية الحديثة تكشف لنا جوانب جديدة من تاريخنا العظيم',
                ],
                'voice_preset': 'Documentary - Authoritative',
                'expected_duration_range': (8, 20),
                'quality_threshold': 0.8
            },
            'audiobook': {
                'name': 'Audiobook Narration',
                'texts': [
                    'كان يا ما كان في قديم الزمان، في قرية صغيرة على ضفاف النيل',
                    'البطل بتاعنا كان شاب طموح بيحلم بمستقبل أفضل',
                ],
                'voice_preset': 'Audiobook - Natural',
                'expected_duration_range': (10, 25),
                'quality_threshold': 0.8
            },
            'news': {
                'name': 'News Broadcasting',
                'texts': [
                    'أهم الأخبار في نشرة اليوم',
                    'الرئيس المصري يستقبل وفداً رفيع المستوى',
                    'الطقس غداً معتدل في معظم المحافظات'
                ],
                'voice_preset': 'News - Professional',
                'expected_duration_range': (3, 10),
                'quality_threshold': 0.85
            }
        }
    
    def run_comprehensive_tests(self) -> Dict[str, any]:
        """Run comprehensive test suite."""
        print("🧪 Starting Egyptian Arabic TTS Voice Over Test Suite")
        print("=" * 60)
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'scenario_results': {},
            'overall_metrics': {},
            'passed_tests': 0,
            'total_tests': 0,
            'quality_summary': {}
        }
        
        # Run scenario tests
        for scenario_id, scenario in self.test_scenarios.items():
            print(f"\n📋 Testing Scenario: {scenario['name']}")
            scenario_result = self.test_scenario(scenario_id, scenario)
            test_results['scenario_results'][scenario_id] = scenario_result
            
            # Update counters
            test_results['total_tests'] += scenario_result['total_tests']
            test_results['passed_tests'] += scenario_result['passed_tests']
        
        # Calculate overall metrics
        test_results['overall_metrics'] = self.calculate_overall_metrics(test_results)
        
        # Generate quality summary
        test_results['quality_summary'] = self.generate_quality_summary(test_results)
        
        # Save results
        self.save_test_results(test_results)
        
        # Print summary
        self.print_test_summary(test_results)
        
        return test_results
    
    def test_scenario(self, scenario_id: str, scenario: Dict) -> Dict:
        """Test a specific voice over scenario."""
        results = {
            'scenario_id': scenario_id,
            'scenario_name': scenario['name'],
            'text_results': [],
            'passed_tests': 0,
            'total_tests': 0,
            'average_quality': 0,
            'quality_threshold': scenario['quality_threshold']
        }
        
        for i, text in enumerate(scenario['texts']):
            print(f"  🎯 Testing text {i+1}/{len(scenario['texts'])}")
            
            # Generate voice over
            try:
                audio, sr, metadata = self.synthesizer.synthesize_voice_over(
                    text=text,
                    voice_preset=scenario['voice_preset']
                )
                
                # Test the generated audio
                text_result = self.test_generated_audio(
                    audio, sr, text, scenario, metadata
                )
                
                results['text_results'].append(text_result)
                results['total_tests'] += text_result['total_tests']
                results['passed_tests'] += text_result['passed_tests']
                
            except Exception as e:
                print(f"    ❌ Error generating audio: {e}")
                error_result = {
                    'text': text,
                    'error': str(e),
                    'passed_tests': 0,
                    'total_tests': 1,
                    'quality_score': 0
                }
                results['text_results'].append(error_result)
                results['total_tests'] += 1
        
        # Calculate average quality
        quality_scores = [r.get('quality_score', 0) for r in results['text_results']]
        results['average_quality'] = np.mean(quality_scores) if quality_scores else 0
        
        return results
    
    def test_generated_audio(self, audio: np.ndarray, sr: int, 
                           text: str, scenario: Dict, metadata: Dict) -> Dict:
        """Test generated audio against quality criteria."""
        result = {
            'text': text,
            'audio_length': len(audio) / sr,
            'tests': {},
            'passed_tests': 0,
            'total_tests': 0,
            'quality_score': 0
        }
        
        # Test 1: Duration check
        duration = len(audio) / sr
        expected_range = scenario.get('expected_duration_range', (1, 60))
        duration_ok = expected_range[0] <= duration <= expected_range[1]
        
        result['tests']['duration_check'] = {
            'passed': duration_ok,
            'actual': duration,
            'expected_range': expected_range
        }
        result['total_tests'] += 1
        if duration_ok:
            result['passed_tests'] += 1
        
        # Test 2: Audio quality standards
        audio_quality = self.audio_processor.assess_quality(audio, sr)
        quality_ok = audio_quality.get('meets_vo_standards', False)
        
        result['tests']['audio_quality'] = {
            'passed': quality_ok,
            'details': audio_quality
        }
        result['total_tests'] += 1
        if quality_ok:
            result['passed_tests'] += 1
        
        # Test 3: Voice over specific quality
        vo_quality = self.evaluator.voice_over_evaluator.evaluate(audio)
        vo_quality_ok = vo_quality >= scenario['quality_threshold']
        
        result['tests']['voice_over_quality'] = {
            'passed': vo_quality_ok,
            'score': vo_quality,
            'threshold': scenario['quality_threshold']
        }
        result['total_tests'] += 1
        if vo_quality_ok:
            result['passed_tests'] += 1
        
        # Test 4: No clipping
        clipping_percent = np.mean(np.abs(audio) >= 0.99) * 100
        no_clipping = clipping_percent < self.config['voice_over_standards']['max_clipping_percent']
        
        result['tests']['no_clipping'] = {
            'passed': no_clipping,
            'clipping_percent': clipping_percent,
            'threshold': self.config['voice_over_standards']['max_clipping_percent']
        }
        result['total_tests'] += 1
        if no_clipping:
            result['passed_tests'] += 1
        
        # Test 5: Dynamic range
        dynamic_range = self.calculate_dynamic_range(audio)
        good_dynamic_range = dynamic_range >= self.config['voice_over_standards']['min_dynamic_range_db']
        
        result['tests']['dynamic_range'] = {
            'passed': good_dynamic_range,
            'dynamic_range_db': dynamic_range,
            'threshold': self.config['voice_over_standards']['min_dynamic_range_db']
        }
        result['total_tests'] += 1
        if good_dynamic_range:
            result['passed_tests'] += 1
        
        # Test 6: Egyptian dialect characteristics
        dialect_score = self.test_egyptian_dialect_features(audio, sr, text)
        dialect_ok = dialect_score >= 0.7
        
        result['tests']['dialect_accuracy'] = {
            'passed': dialect_ok,
            'score': dialect_score,
            'threshold': 0.7
        }
        result['total_tests'] += 1
        if dialect_ok:
            result['passed_tests'] += 1
        
        # Calculate overall quality score
        result['quality_score'] = metadata.get('quality_metrics', {}).get('voice_over_quality', vo_quality)
        
        return result
    
    def calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range of audio."""
        peak_level = np.max(np.abs(audio))
        noise_floor = np.percentile(np.abs(audio), 5)
        
        if noise_floor > 0:
            dynamic_range = 20 * np.log10(peak_level / noise_floor)
        else:
            dynamic_range = 60.0  # Very high dynamic range
        
        return dynamic_range
    
    def test_egyptian_dialect_features(self, audio: np.ndarray, sr: int, text: str) -> float:
        """Test for Egyptian Arabic dialect characteristics."""
        # Simplified dialect testing - in practice would use phonetic analysis
        
        # Check for common Egyptian Arabic markers in text
        egyptian_markers = ['ده', 'دي', 'إيه', 'ازاي', 'فين', 'امتى', 'ليه']
        text_score = sum(1 for marker in egyptian_markers if marker in text) / len(egyptian_markers)
        
        # Audio-based features (simplified)
        # In practice, would analyze phonetic features specific to Egyptian Arabic
        audio_score = 0.8  # Placeholder
        
        return (text_score + audio_score) / 2
    
    def calculate_overall_metrics(self, test_results: Dict) -> Dict:
        """Calculate overall test metrics."""
        total_tests = test_results['total_tests']
        passed_tests = test_results['passed_tests']
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Calculate average quality across all scenarios
        quality_scores = []
        for scenario_result in test_results['scenario_results'].values():
            quality_scores.append(scenario_result['average_quality'])
        
        average_quality = np.mean(quality_scores) if quality_scores else 0
        
        return {
            'success_rate': success_rate,
            'average_quality': average_quality,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests
        }
    
    def generate_quality_summary(self, test_results: Dict) -> Dict:
        """Generate quality summary."""
        overall_metrics = test_results['overall_metrics']
        
        # Determine overall grade
        avg_quality = overall_metrics['average_quality']
        success_rate = overall_metrics['success_rate']
        
        if avg_quality >= 0.9 and success_rate >= 0.9:
            grade = 'Excellent'
        elif avg_quality >= 0.8 and success_rate >= 0.8:
            grade = 'Good'
        elif avg_quality >= 0.7 and success_rate >= 0.7:
            grade = 'Fair'
        else:
            grade = 'Poor'
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for scenario_id, scenario_result in test_results['scenario_results'].items():
            if scenario_result['average_quality'] >= 0.85:
                strengths.append(scenario_result['scenario_name'])
            elif scenario_result['average_quality'] < 0.7:
                weaknesses.append(scenario_result['scenario_name'])
        
        return {
            'overall_grade': grade,
            'average_quality': avg_quality,
            'success_rate': success_rate,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': self.generate_recommendations(test_results)
        }
    
    def generate_recommendations(self, test_results: Dict) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        overall_metrics = test_results['overall_metrics']
        
        if overall_metrics['average_quality'] < 0.8:
            recommendations.append("Improve overall audio quality through better training data")
        
        if overall_metrics['success_rate'] < 0.8:
            recommendations.append("Address failing test cases to improve reliability")
        
        # Scenario-specific recommendations
        for scenario_id, scenario_result in test_results['scenario_results'].items():
            if scenario_result['average_quality'] < 0.7:
                recommendations.append(f"Focus on improving {scenario_result['scenario_name']} quality")
        
        # Test-specific recommendations
        common_failures = self.analyze_common_failures(test_results)
        for failure_type, count in common_failures.items():
            if count > len(test_results['scenario_results']) / 2:
                recommendations.append(f"Address {failure_type} issues across scenarios")
        
        return recommendations
    
    def analyze_common_failures(self, test_results: Dict) -> Dict[str, int]:
        """Analyze common failure patterns."""
        failure_counts = {}
        
        for scenario_result in test_results['scenario_results'].values():
            for text_result in scenario_result['text_results']:
                if 'tests' in text_result:
                    for test_name, test_result in text_result['tests'].items():
                        if not test_result['passed']:
                            failure_counts[test_name] = failure_counts.get(test_name, 0) + 1
        
        return failure_counts
    
    def save_test_results(self, test_results: Dict):
        """Save test results to file."""
        output_dir = self.config.get('output_dir', './test_results')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'test_results_{timestamp}.json')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Test results saved to: {results_file}")
    
    def print_test_summary(self, test_results: Dict):
        """Print test summary to console."""
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        
        overall = test_results['overall_metrics']
        quality_summary = test_results['quality_summary']
        
        print(f"Overall Grade: {quality_summary['overall_grade']}")
        print(f"Success Rate: {overall['success_rate']:.1%}")
        print(f"Average Quality: {overall['average_quality']:.3f}")
        print(f"Tests Passed: {overall['passed_tests']}/{overall['total_tests']}")
        
        print(f"\n📊 Scenario Performance:")
        for scenario_id, result in test_results['scenario_results'].items():
            status = "✅" if result['average_quality'] >= result['quality_threshold'] else "❌"
            print(f"  {status} {result['scenario_name']}: {result['average_quality']:.3f}")
        
        if quality_summary['strengths']:
            print(f"\n💪 Strengths: {', '.join(quality_summary['strengths'])}")
        
        if quality_summary['weaknesses']:
            print(f"\n⚠️  Weaknesses: {', '.join(quality_summary['weaknesses'])}")
        
        if quality_summary['recommendations']:
            print(f"\n🔧 Recommendations:")
            for rec in quality_summary['recommendations']:
                print(f"  • {rec}")


class UnitTests(unittest.TestCase):
    """Unit tests for individual components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synthesizer = VoiceSynthesizer()
        self.evaluator = EgyptianTTSEvaluator({'sample_rate': 48000})
        self.audio_processor = AudioProcessor(target_sr=48000, target_bit_depth=24)
    
    def test_text_processing(self):
        """Test Egyptian Arabic text processing."""
        text = "مرحبا بكم في برنامجنا الجديد"
        
        # Test text processor
        processed = self.synthesizer.text_processor.process_for_tts(text)
        
        self.assertIsInstance(processed, dict)
        self.assertIn('text', processed)
        self.assertIn('phonemes', processed)
        self.assertIn('metadata', processed)
    
    def test_audio_generation(self):
        """Test audio generation."""
        text = "هذا اختبار للصوت المصري"
        
        # Generate audio
        audio, sr, metadata = self.synthesizer.synthesize_voice_over(text)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(sr, 48000)
        self.assertGreater(len(audio), 0)
        self.assertIsInstance(metadata, dict)
    
    def test_quality_evaluation(self):
        """Test quality evaluation."""
        # Generate test audio
        duration = 2.0
        sr = 48000
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 150 * t)
        
        # Evaluate quality
        quality = self.audio_processor.assess_quality(audio, sr)
        
        self.assertIsInstance(quality, dict)
        self.assertIn('snr_db', quality)
        self.assertIn('dynamic_range_db', quality)
    
    def test_voice_presets(self):
        """Test voice preset functionality."""
        text = "اختبار الأصوات المختلفة"
        
        for preset_name in self.synthesizer.voice_presets.keys():
            with self.subTest(preset=preset_name):
                audio, sr, metadata = self.synthesizer.synthesize_voice_over(
                    text, voice_preset=preset_name
                )
                
                self.assertIsInstance(audio, np.ndarray)
                self.assertEqual(sr, 48000)
                self.assertGreater(len(audio), 0)


def create_test_config() -> Dict:
    """Create default test configuration."""
    return {
        'test_data_dir': './test_data',
        'output_dir': './test_results',
        'voice_over_standards': {
            'sample_rate': 48000,
            'bit_depth': 24,
            'min_snr_db': 40,
            'max_clipping_percent': 0.1,
            'min_dynamic_range_db': 30
        },
        'quality_thresholds': {
            'minimum_acceptable': 0.7,
            'good_quality': 0.8,
            'excellent_quality': 0.9
        }
    }


def main():
    """Main function to run tests."""
    print("🧪 Egyptian Arabic TTS Voice Over Testing Framework")
    print("=" * 60)
    
    # Create test suite
    test_suite = VoiceOverTestSuite()
    
    # Run comprehensive tests
    results = test_suite.run_comprehensive_tests()
    
    # Run unit tests
    print("\n🔬 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    return results


if __name__ == "__main__":
    main()

