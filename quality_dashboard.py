"""
Quality Evaluation Dashboard for Egyptian Arabic TTS

This module provides a comprehensive dashboard for evaluating and monitoring
the quality of Egyptian Arabic voice over synthesis in real-time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime, timedelta
import tempfile

# Import evaluation components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import EgyptianTTSEvaluator
from preprocessing.audio_processor import AudioProcessor


class QualityDashboard:
    """
    Interactive quality evaluation dashboard for Egyptian Arabic TTS.
    Provides real-time quality monitoring and analysis tools.
    """
    
    def __init__(self):
        """Initialize quality dashboard."""
        self.evaluator = EgyptianTTSEvaluator({'sample_rate': 48000})
        self.audio_processor = AudioProcessor(target_sr=48000, target_bit_depth=24)
        
        # Quality thresholds for voice over standards
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'fair': 0.7,
            'poor': 0.6
        }
        
        # Initialize session state
        if 'evaluation_history' not in st.session_state:
            st.session_state.evaluation_history = []
        
        if 'quality_metrics' not in st.session_state:
            st.session_state.quality_metrics = {}
    
    def create_dashboard(self):
        """Create the main quality evaluation dashboard."""
        st.set_page_config(
            page_title="Egyptian Arabic TTS Quality Dashboard",
            page_icon="🎙️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .quality-excellent { border-left-color: #2ca02c; }
        .quality-good { border-left-color: #ff7f0e; }
        .quality-fair { border-left-color: #d62728; }
        .quality-poor { border-left-color: #8c564b; }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("🎙️ Egyptian Arabic TTS Quality Dashboard")
        st.markdown("Real-time quality evaluation and monitoring for voice over synthesis")
        
        # Sidebar
        self.create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Real-time Evaluation", 
            "📈 Quality Analytics", 
            "🔍 Detailed Analysis",
            "📋 Quality Reports"
        ])
        
        with tab1:
            self.create_realtime_evaluation()
        
        with tab2:
            self.create_quality_analytics()
        
        with tab3:
            self.create_detailed_analysis()
        
        with tab4:
            self.create_quality_reports()
    
    def create_sidebar(self):
        """Create sidebar with controls and settings."""
        st.sidebar.header("⚙️ Evaluation Settings")
        
        # Quality standards selection
        st.sidebar.subheader("Quality Standards")
        voice_over_standard = st.sidebar.selectbox(
            "Voice Over Standard",
            ["Professional", "Broadcast", "Commercial", "Educational"],
            index=0
        )
        
        # Evaluation metrics selection
        st.sidebar.subheader("Metrics to Evaluate")
        eval_phoneme = st.sidebar.checkbox("Phoneme Accuracy", value=True)
        eval_prosody = st.sidebar.checkbox("Prosody Quality", value=True)
        eval_audio = st.sidebar.checkbox("Audio Quality", value=True)
        eval_dialect = st.sidebar.checkbox("Dialect Accuracy", value=True)
        eval_naturalness = st.sidebar.checkbox("Naturalness", value=True)
        
        # Store settings in session state
        st.session_state.eval_settings = {
            'standard': voice_over_standard,
            'metrics': {
                'phoneme': eval_phoneme,
                'prosody': eval_prosody,
                'audio': eval_audio,
                'dialect': eval_dialect,
                'naturalness': eval_naturalness
            }
        }
        
        # Quality thresholds
        st.sidebar.subheader("Quality Thresholds")
        excellent_threshold = st.sidebar.slider("Excellent", 0.8, 1.0, 0.9, 0.01)
        good_threshold = st.sidebar.slider("Good", 0.6, 0.9, 0.8, 0.01)
        fair_threshold = st.sidebar.slider("Fair", 0.4, 0.8, 0.7, 0.01)
        
        self.quality_thresholds = {
            'excellent': excellent_threshold,
            'good': good_threshold,
            'fair': fair_threshold,
            'poor': 0.0
        }
    
    def create_realtime_evaluation(self):
        """Create real-time evaluation interface."""
        st.header("📊 Real-time Quality Evaluation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Audio Upload & Analysis")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Generated Audio",
                type=['wav', 'mp3', 'flac'],
                help="Upload audio file generated by Egyptian Arabic TTS"
            )
            
            reference_file = st.file_uploader(
                "Upload Reference Audio (Optional)",
                type=['wav', 'mp3', 'flac'],
                help="Upload reference audio for comparison"
            )
            
            # Text input for context
            text_input = st.text_area(
                "Original Text (Egyptian Arabic)",
                placeholder="أدخل النص الأصلي هنا...",
                help="Enter the original text that was synthesized"
            )
            
            if uploaded_file is not None:
                # Load and display audio
                audio_data, sample_rate = self.load_audio_file(uploaded_file)
                
                st.audio(uploaded_file, format='audio/wav')
                
                # Evaluate quality
                if st.button("🔍 Evaluate Quality", type="primary"):
                    with st.spinner("Evaluating audio quality..."):
                        quality_results = self.evaluate_audio_quality(
                            audio_data, sample_rate, text_input, reference_file
                        )
                        
                        # Store results
                        st.session_state.quality_metrics = quality_results
                        st.session_state.evaluation_history.append({
                            'timestamp': datetime.now(),
                            'filename': uploaded_file.name,
                            'metrics': quality_results
                        })
                        
                        st.success("✅ Quality evaluation completed!")
        
        with col2:
            st.subheader("Quick Quality Overview")
            
            if st.session_state.quality_metrics:
                self.display_quality_overview(st.session_state.quality_metrics)
            else:
                st.info("Upload and evaluate an audio file to see quality metrics")
        
        # Detailed results
        if st.session_state.quality_metrics:
            st.subheader("📋 Detailed Quality Metrics")
            self.display_detailed_metrics(st.session_state.quality_metrics)
    
    def create_quality_analytics(self):
        """Create quality analytics dashboard."""
        st.header("📈 Quality Analytics")
        
        if not st.session_state.evaluation_history:
            st.info("No evaluation history available. Evaluate some audio files first!")
            return
        
        # Convert history to DataFrame
        df = self.create_analytics_dataframe()
        
        # Time series analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quality Trends Over Time")
            self.plot_quality_trends(df)
        
        with col2:
            st.subheader("Quality Distribution")
            self.plot_quality_distribution(df)
        
        # Metric correlations
        st.subheader("Metric Correlations")
        self.plot_metric_correlations(df)
        
        # Performance summary
        st.subheader("Performance Summary")
        self.display_performance_summary(df)
    
    def create_detailed_analysis(self):
        """Create detailed analysis interface."""
        st.header("🔍 Detailed Quality Analysis")
        
        if not st.session_state.quality_metrics:
            st.info("No current evaluation results. Please evaluate an audio file first.")
            return
        
        metrics = st.session_state.quality_metrics
        
        # Audio waveform analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Audio Waveform Analysis")
            if 'audio_data' in metrics:
                self.plot_waveform_analysis(metrics['audio_data'], metrics['sample_rate'])
        
        with col2:
            st.subheader("Spectral Analysis")
            if 'audio_data' in metrics:
                self.plot_spectral_analysis(metrics['audio_data'], metrics['sample_rate'])
        
        # Prosody analysis
        st.subheader("Prosodic Features Analysis")
        if 'prosody_features' in metrics:
            self.plot_prosody_analysis(metrics['prosody_features'])
        
        # Quality breakdown
        st.subheader("Quality Score Breakdown")
        self.plot_quality_breakdown(metrics)
    
    def create_quality_reports(self):
        """Create quality reports interface."""
        st.header("📋 Quality Reports")
        
        if not st.session_state.evaluation_history:
            st.info("No evaluation history available for reporting.")
            return
        
        # Report generation options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Generate Quality Report")
            
            report_type = st.selectbox(
                "Report Type",
                ["Summary Report", "Detailed Analysis", "Comparative Analysis", "Trend Report"]
            )
            
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=7), datetime.now()),
                help="Select date range for report"
            )
            
            if st.button("📄 Generate Report"):
                report_content = self.generate_quality_report(report_type, date_range)
                st.markdown(report_content)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report_content,
                    file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col2:
            st.subheader("Report Statistics")
            df = self.create_analytics_dataframe()
            
            total_evaluations = len(df)
            avg_quality = df['overall_quality'].mean() if 'overall_quality' in df.columns else 0
            
            st.metric("Total Evaluations", total_evaluations)
            st.metric("Average Quality", f"{avg_quality:.3f}")
            
            # Quality grade distribution
            if 'quality_grade' in df.columns:
                grade_counts = df['quality_grade'].value_counts()
                st.bar_chart(grade_counts)
    
    def load_audio_file(self, uploaded_file) -> Tuple[np.ndarray, int]:
        """Load audio file from upload."""
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load audio
        audio_data, sample_rate = self.audio_processor.load_audio(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return audio_data, sample_rate
    
    def evaluate_audio_quality(self, audio_data: np.ndarray, sample_rate: int,
                              text: str = None, reference_file=None) -> Dict:
        """Evaluate audio quality comprehensively."""
        results = {
            'audio_data': audio_data,
            'sample_rate': sample_rate,
            'timestamp': datetime.now()
        }
        
        # Basic audio quality assessment
        audio_quality = self.audio_processor.assess_quality(audio_data, sample_rate)
        results['audio_quality'] = audio_quality
        
        # Voice over specific quality
        vo_quality = self.evaluator.voice_over_evaluator.evaluate(audio_data)
        results['voice_over_quality'] = vo_quality
        
        # Naturalness assessment
        if reference_file is not None:
            ref_audio, ref_sr = self.load_audio_file(reference_file)
            naturalness = self.evaluator.evaluate_naturalness(audio_data, ref_audio)
            intelligibility = self.evaluator.evaluate_intelligibility(audio_data, ref_audio)
            results['naturalness'] = naturalness
            results['intelligibility'] = intelligibility
        
        # Prosodic analysis
        prosody_features = self.extract_prosodic_features(audio_data, sample_rate)
        results['prosody_features'] = prosody_features
        
        # Overall quality score
        overall_quality = self.calculate_overall_quality(results)
        results['overall_quality'] = overall_quality
        results['quality_grade'] = self.get_quality_grade(overall_quality)
        
        return results
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract prosodic features for analysis."""
        features = {}
        
        # Fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        features['f0'] = f0
        features['voiced_flag'] = voiced_flag
        
        # Energy
        rms = librosa.feature.rms(y=audio)[0]
        features['energy'] = rms
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid'] = spectral_centroids
        
        return features
    
    def calculate_overall_quality(self, results: Dict) -> float:
        """Calculate overall quality score."""
        scores = []
        
        # Audio quality components
        if 'audio_quality' in results:
            aq = results['audio_quality']
            if aq.get('meets_vo_standards', False):
                scores.append(0.9)
            else:
                scores.append(0.6)
        
        # Voice over quality
        if 'voice_over_quality' in results:
            scores.append(results['voice_over_quality'])
        
        # Naturalness and intelligibility
        if 'naturalness' in results:
            scores.append(results['naturalness'])
        
        if 'intelligibility' in results:
            scores.append(results['intelligibility'])
        
        return np.mean(scores) if scores else 0.5
    
    def get_quality_grade(self, score: float) -> str:
        """Get quality grade based on score."""
        if score >= self.quality_thresholds['excellent']:
            return "Excellent"
        elif score >= self.quality_thresholds['good']:
            return "Good"
        elif score >= self.quality_thresholds['fair']:
            return "Fair"
        else:
            return "Poor"
    
    def display_quality_overview(self, metrics: Dict):
        """Display quick quality overview."""
        overall_quality = metrics.get('overall_quality', 0)
        quality_grade = metrics.get('quality_grade', 'Unknown')
        
        # Color coding based on quality
        color_map = {
            'Excellent': '#2ca02c',
            'Good': '#ff7f0e', 
            'Fair': '#d62728',
            'Poor': '#8c564b'
        }
        
        color = color_map.get(quality_grade, '#1f77b4')
        
        # Display metrics
        st.markdown(f"""
        <div class="metric-card quality-{quality_grade.lower()}">
            <h3 style="color: {color};">Overall Quality: {quality_grade}</h3>
            <h2 style="color: {color};">{overall_quality:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual metrics
        if 'voice_over_quality' in metrics:
            st.metric("Voice Over Quality", f"{metrics['voice_over_quality']:.3f}")
        
        if 'naturalness' in metrics:
            st.metric("Naturalness", f"{metrics['naturalness']:.3f}")
        
        if 'intelligibility' in metrics:
            st.metric("Intelligibility", f"{metrics['intelligibility']:.3f}")
    
    def display_detailed_metrics(self, metrics: Dict):
        """Display detailed quality metrics."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Audio Quality")
            if 'audio_quality' in metrics:
                aq = metrics['audio_quality']
                st.metric("SNR (dB)", f"{aq.get('snr_db', 0):.1f}")
                st.metric("Dynamic Range (dB)", f"{aq.get('dynamic_range_db', 0):.1f}")
                st.metric("Peak Level (dB)", f"{aq.get('peak_level_db', 0):.1f}")
        
        with col2:
            st.subheader("Voice Over Standards")
            if 'audio_quality' in metrics:
                aq = metrics['audio_quality']
                meets_standards = aq.get('meets_vo_standards', False)
                st.metric("Meets VO Standards", "✅ Yes" if meets_standards else "❌ No")
                st.metric("Clipping %", f"{aq.get('clipping_percentage', 0):.2f}%")
                st.metric("Silence Ratio", f"{aq.get('silence_ratio', 0):.3f}")
        
        with col3:
            st.subheader("Prosodic Quality")
            if 'prosody_features' in metrics:
                pf = metrics['prosody_features']
                f0_values = pf.get('f0', [])
                if len(f0_values) > 0:
                    f0_clean = f0_values[~np.isnan(f0_values)]
                    if len(f0_clean) > 0:
                        st.metric("Mean F0 (Hz)", f"{np.mean(f0_clean):.1f}")
                        st.metric("F0 Std (Hz)", f"{np.std(f0_clean):.1f}")
    
    def plot_waveform_analysis(self, audio_data: np.ndarray, sample_rate: int):
        """Plot waveform analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Time axis
        time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        # Waveform
        ax1.plot(time, audio_data)
        ax1.set_title("Audio Waveform")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        
        # RMS energy
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)
        
        ax2.plot(rms_time, rms)
        ax2.set_title("RMS Energy")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("RMS")
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def plot_spectral_analysis(self, audio_data: np.ndarray, sample_rate: int):
        """Plot spectral analysis."""
        # Compute spectrogram
        D = librosa.stft(audio_data)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        img = librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title("Spectrogram")
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        st.pyplot(fig)
    
    def plot_prosody_analysis(self, prosody_features: Dict):
        """Plot prosodic features analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # F0 contour
        if 'f0' in prosody_features:
            f0 = prosody_features['f0']
            axes[0, 0].plot(f0)
            axes[0, 0].set_title("Fundamental Frequency (F0)")
            axes[0, 0].set_ylabel("Frequency (Hz)")
            axes[0, 0].grid(True)
        
        # Energy contour
        if 'energy' in prosody_features:
            energy = prosody_features['energy']
            axes[0, 1].plot(energy)
            axes[0, 1].set_title("Energy Contour")
            axes[0, 1].set_ylabel("RMS Energy")
            axes[0, 1].grid(True)
        
        # Spectral centroid
        if 'spectral_centroid' in prosody_features:
            sc = prosody_features['spectral_centroid']
            axes[1, 0].plot(sc)
            axes[1, 0].set_title("Spectral Centroid")
            axes[1, 0].set_ylabel("Frequency (Hz)")
            axes[1, 0].grid(True)
        
        # Voice activity
        if 'voiced_flag' in prosody_features:
            voiced = prosody_features['voiced_flag'].astype(float)
            axes[1, 1].plot(voiced)
            axes[1, 1].set_title("Voice Activity")
            axes[1, 1].set_ylabel("Voiced (1) / Unvoiced (0)")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def plot_quality_breakdown(self, metrics: Dict):
        """Plot quality score breakdown."""
        quality_components = {}
        
        if 'voice_over_quality' in metrics:
            quality_components['Voice Over'] = metrics['voice_over_quality']
        
        if 'naturalness' in metrics:
            quality_components['Naturalness'] = metrics['naturalness']
        
        if 'intelligibility' in metrics:
            quality_components['Intelligibility'] = metrics['intelligibility']
        
        if 'audio_quality' in metrics and metrics['audio_quality'].get('meets_vo_standards'):
            quality_components['Audio Standards'] = 0.9
        else:
            quality_components['Audio Standards'] = 0.6
        
        if quality_components:
            fig = px.bar(
                x=list(quality_components.keys()),
                y=list(quality_components.values()),
                title="Quality Score Breakdown",
                labels={'x': 'Quality Component', 'y': 'Score'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def create_analytics_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from evaluation history."""
        data = []
        
        for entry in st.session_state.evaluation_history:
            row = {
                'timestamp': entry['timestamp'],
                'filename': entry['filename'],
                'overall_quality': entry['metrics'].get('overall_quality', 0),
                'quality_grade': entry['metrics'].get('quality_grade', 'Unknown'),
                'voice_over_quality': entry['metrics'].get('voice_over_quality', 0),
                'naturalness': entry['metrics'].get('naturalness', 0),
                'intelligibility': entry['metrics'].get('intelligibility', 0)
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_quality_trends(self, df: pd.DataFrame):
        """Plot quality trends over time."""
        if df.empty:
            return
        
        fig = px.line(
            df, 
            x='timestamp', 
            y='overall_quality',
            title="Quality Trends Over Time",
            labels={'overall_quality': 'Overall Quality Score', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_quality_distribution(self, df: pd.DataFrame):
        """Plot quality distribution."""
        if df.empty:
            return
        
        fig = px.histogram(
            df,
            x='quality_grade',
            title="Quality Grade Distribution",
            labels={'quality_grade': 'Quality Grade', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_metric_correlations(self, df: pd.DataFrame):
        """Plot metric correlations."""
        if df.empty or len(df) < 2:
            return
        
        numeric_cols = ['overall_quality', 'voice_over_quality', 'naturalness', 'intelligibility']
        available_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().sum() > 1]
        
        if len(available_cols) < 2:
            return
        
        corr_matrix = df[available_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Metric Correlations",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_performance_summary(self, df: pd.DataFrame):
        """Display performance summary statistics."""
        if df.empty:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_quality = df['overall_quality'].mean()
            st.metric("Average Quality", f"{avg_quality:.3f}")
        
        with col2:
            excellent_count = (df['quality_grade'] == 'Excellent').sum()
            excellent_pct = excellent_count / len(df) * 100
            st.metric("Excellent Rate", f"{excellent_pct:.1f}%")
        
        with col3:
            if 'voice_over_quality' in df.columns:
                avg_vo_quality = df['voice_over_quality'].mean()
                st.metric("Avg VO Quality", f"{avg_vo_quality:.3f}")
        
        with col4:
            total_evaluations = len(df)
            st.metric("Total Evaluations", total_evaluations)
    
    def generate_quality_report(self, report_type: str, date_range: Tuple) -> str:
        """Generate quality report."""
        df = self.create_analytics_dataframe()
        
        # Filter by date range
        if not df.empty:
            df = df[(df['timestamp'].dt.date >= date_range[0]) & 
                   (df['timestamp'].dt.date <= date_range[1])]
        
        report = f"# Egyptian Arabic TTS Quality Report\n\n"
        report += f"**Report Type:** {report_type}\n"
        report += f"**Date Range:** {date_range[0]} to {date_range[1]}\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if df.empty:
            report += "No data available for the selected date range.\n"
            return report
        
        # Summary statistics
        report += "## Summary Statistics\n\n"
        report += f"- **Total Evaluations:** {len(df)}\n"
        report += f"- **Average Quality Score:** {df['overall_quality'].mean():.3f}\n"
        report += f"- **Quality Standard Deviation:** {df['overall_quality'].std():.3f}\n"
        
        # Quality grade distribution
        grade_counts = df['quality_grade'].value_counts()
        report += f"\n### Quality Grade Distribution\n\n"
        for grade, count in grade_counts.items():
            percentage = count / len(df) * 100
            report += f"- **{grade}:** {count} ({percentage:.1f}%)\n"
        
        # Recommendations
        report += f"\n## Recommendations\n\n"
        avg_quality = df['overall_quality'].mean()
        
        if avg_quality >= 0.9:
            report += "- ✅ Excellent overall quality. Continue current practices.\n"
        elif avg_quality >= 0.8:
            report += "- 🟡 Good quality with room for improvement in specific areas.\n"
        elif avg_quality >= 0.7:
            report += "- 🟠 Fair quality. Consider reviewing training data and model parameters.\n"
        else:
            report += "- 🔴 Poor quality. Significant improvements needed in model and data.\n"
        
        return report


def main():
    """Main function to run the quality dashboard."""
    dashboard = QualityDashboard()
    dashboard.create_dashboard()


if __name__ == "__main__":
    main()

