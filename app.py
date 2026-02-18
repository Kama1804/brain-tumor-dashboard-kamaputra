import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import os
import gdown
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional

# ==================== CRITICAL: Force CPU ====================

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="NeuroScan AI | Medical Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
MODEL_PATH = 'best.pt'
WAREHOUSE_FILE = 'tumor_warehouse.csv'
PATIENT_DATA_FILE = 'patient_records.csv'

# Class name mapping for better display
CLASS_DISPLAY_NAMES = {
    'glioma': 'Glioma (Malignant)',
    'meningioma': 'Meningioma (Benign)',
    'pituitary': 'Pituitary Tumor',
    'no_tumor': 'No Tumor Detected'
}

# Severity thresholds
SEVERITY_THRESHOLDS = {
    'critical': 0.85,
    'urgent': 0.50,
    'routine': 0.0
}

# ==================== CUSTOM CSS STYLING ====================
def load_custom_css():
    """Apply custom CSS for better UI/UX"""
    st.markdown("""
    <style>
        /* Main container improvements */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Metric cards enhancement */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
        }

        /* Success/Warning/Error boxes */
        .stAlert {
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
        }

        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        /* Dataframe styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 12px 24px;
            font-weight: 600;
        }

        /* Custom card styling */
        .custom-card {
            background: black;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }

        /* Insight card */
        .insight-card {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            padding: 1.2rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }

        /* Risk bar styling */
        .risk-high { color: #ef4444; font-weight: 700; }
        .risk-medium { color: #f59e0b; font-weight: 700; }
        .risk-low { color: #10b981; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ==================== MODEL MANAGEMENT ====================
@st.cache_resource
def load_model() -> Optional[YOLO]:
    """
    Load YOLO model with error handling
    Returns: YOLO model or None if loading fails
    """
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file '{MODEL_PATH}' not found!")
            st.info("""
            **Setup Instructions:**
            1. Complete YOLOv8 model training
            2. Copy 'best.pt' to the application directory
            3. Restart the application
            """)
            return None

        model = YOLO(MODEL_PATH)
        st.sidebar.success("✅ AI Model Loaded Successfully")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ==================== DATA WAREHOUSE MANAGEMENT ====================
class DataWarehouse:
    """Manages all data warehouse operations with Star Schema"""

    @staticmethod
    def init_warehouse() -> None:
        """Initialize CSV warehouse with Star Schema structure"""
        columns = [
            'Detection_ID',
            'Scan_Date',
            'Scan_Time',
            'Image_Name',
            'Tumor_Class',
            'Confidence',
            'Num_Detections',
            'Box_Coordinates',
            'Severity_Level',
            'Patient_ID'
        ]

        try:
            if not os.path.exists(WAREHOUSE_FILE):
                df = pd.DataFrame(columns=columns)
                df.to_csv(WAREHOUSE_FILE, index=False)
            else:
                df = pd.read_csv(WAREHOUSE_FILE)
                missing_cols = set(columns) - set(df.columns)
                if missing_cols:
                    for col in missing_cols:
                        df[col] = None
                    df.to_csv(WAREHOUSE_FILE, index=False)
        except Exception as e:
            st.error(f"Error initializing warehouse: {str(e)}")
            df = pd.DataFrame(columns=columns)
            df.to_csv(WAREHOUSE_FILE, index=False)

    @staticmethod
    def get_severity_level(confidence: float) -> str:
        """Determine severity level based on confidence score"""
        if confidence >= SEVERITY_THRESHOLDS['critical']:
            return 'Critical'
        elif confidence >= SEVERITY_THRESHOLDS['urgent']:
            return 'Urgent'
        else:
            return 'Routine'

    @staticmethod
    def update_warehouse(image_name: str, results, patient_id: str = None) -> Dict:
        """
        ETL Process: Extract detection -> Transform -> Load to CSV
        Returns: Dictionary with processing statistics
        """
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y-%m-%d")
        current_time = current_datetime.strftime("%H:%M:%S")
        timestamp_id = int(current_datetime.timestamp() * 1000)

        records_added = 0
        detections_count = len(results[0].boxes) if results and len(results) > 0 else 0

        try:
            if detections_count == 0:
                new_record = {
                    'Detection_ID': timestamp_id,
                    'Scan_Date': current_date,
                    'Scan_Time': current_time,
                    'Image_Name': image_name,
                    'Tumor_Class': 'No Detection',
                    'Confidence': 0.0,
                    'Num_Detections': 0,
                    'Box_Coordinates': '',
                    'Severity_Level': 'None',
                    'Patient_ID': patient_id or ''
                }
                pd.DataFrame([new_record]).to_csv(
                    WAREHOUSE_FILE, mode='a', header=False, index=False
                )
                records_added = 1
            else:
                for i, box in enumerate(results[0].boxes):
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]

                    coords = box.xyxy[0].cpu().numpy()
                    box_coords = f"{coords[0]:.1f},{coords[1]:.1f},{coords[2]:.1f},{coords[3]:.1f}"

                    new_record = {
                        'Detection_ID': f"{timestamp_id}_{i}",
                        'Scan_Date': current_date,
                        'Scan_Time': current_time,
                        'Image_Name': image_name,
                        'Tumor_Class': class_name,
                        'Confidence': round(confidence, 4),
                        'Num_Detections': 1,
                        'Box_Coordinates': box_coords,
                        'Severity_Level': DataWarehouse.get_severity_level(confidence),
                        'Patient_ID': patient_id or ''
                    }
                    pd.DataFrame([new_record]).to_csv(
                        WAREHOUSE_FILE, mode='a', header=False, index=False
                    )
                    records_added += 1

            return {
                'success': True,
                'records_added': records_added,
                'detections': detections_count,
                'timestamp': current_datetime
            }
        except Exception as e:
            st.error(f"Error updating warehouse: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'records_added': 0
            }

    @staticmethod
    def load_warehouse() -> Optional[pd.DataFrame]:
        """Load warehouse data with error handling"""
        try:
            if os.path.exists(WAREHOUSE_FILE):
                df = pd.read_csv(WAREHOUSE_FILE)
                if not df.empty:
                    df['Scan_Date'] = pd.to_datetime(df['Scan_Date'], errors='coerce')
                    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
                    df['Num_Detections'] = pd.to_numeric(df['Num_Detections'], errors='coerce')
                    return df
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading warehouse: {str(e)}")
            return pd.DataFrame()


# ==================== POPULATION INSIGHTS ====================
class PopulationInsights:
    """Loads and analyses patient_records.csv for population-level statistics"""

    @staticmethod
    @st.cache_data(ttl=300)
    def load_insights() -> Dict:
        """
        Read patient_records.csv and compute:
        - Average age overall and by tumor type
        - Risk factor prevalence
        - Radiation exposure distribution
        - Genetic risk summary
        Returns a dictionary of insights.
        """
        result = {
            'loaded': False,
            'total_patients': 0,
            'tumor_patients': 0,
            'overall_avg_age': None,
            'avg_age_by_type': {},           # {tumor_type: avg_age}
            'risk_factor_pcts': {},          # {factor_label: pct}
            'radiation_distribution': {},    # {level: count}
            'genetic_mean': None,
            'genetic_median': None,
            'top_risk_factor': None,
            'top_risk_pct': None,
            'age_range_tumor': None,         # (min, max)
            'gender_distribution': {},       # {gender: count}
            'country_top5': [],              # [(country, count), ...]
        }

        if not os.path.exists(PATIENT_DATA_FILE):
            return result

        try:
            # Read file in chunks for large datasets
            chunk_size = 50000
            chunks = []
            for chunk in pd.read_csv(PATIENT_DATA_FILE, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)

            result['total_patients'] = len(df)

            # Identify tumor patients
            if 'Brain_Tumor_Present' in df.columns:
                tumor_mask = df['Brain_Tumor_Present'].astype(str).str.strip().str.lower() == 'yes'
            elif 'Tumor_Type' in df.columns:
                tumor_mask = df['Tumor_Type'].notna() & \
                             (~df['Tumor_Type'].astype(str).str.lower().isin(['no', 'none', 'nan', '']))
            else:
                tumor_mask = pd.Series([True] * len(df))

            tumor_df = df[tumor_mask].copy()
            result['tumor_patients'] = len(tumor_df)

            # ---- Age analysis ----
            if 'Age' in df.columns:
                tumor_df['Age'] = pd.to_numeric(tumor_df['Age'], errors='coerce')
                ages = tumor_df['Age'].dropna()
                if len(ages):
                    result['overall_avg_age'] = round(float(ages.mean()), 1)
                    result['age_range_tumor'] = (int(ages.min()), int(ages.max()))

            # Average age by tumor type
            if 'Tumor_Type' in tumor_df.columns and 'Age' in tumor_df.columns:
                tumor_df['Age'] = pd.to_numeric(tumor_df['Age'], errors='coerce')
                age_by_type = (
                    tumor_df.groupby('Tumor_Type')['Age']
                    .mean()
                    .dropna()
                    .round(1)
                    .to_dict()
                )
                result['avg_age_by_type'] = age_by_type

            # ---- Risk factors ----
            binary_factors = {
                'Smoking History':       'Smoking_History',
                'Alcohol Consumption':   'Alcohol_Consumption',
                'Head Injury History':   'Head_Injury_History',
                'Chronic Illness':       'Chronic_Illness',
                'Family History':        'Family_History',
            }
            factor_pcts = {}
            for label, col in binary_factors.items():
                if col in tumor_df.columns:
                    pct = (tumor_df[col].astype(str).str.strip().str.lower() == 'yes').mean() * 100
                    factor_pcts[label] = round(float(pct), 1)

            result['risk_factor_pcts'] = dict(
                sorted(factor_pcts.items(), key=lambda x: x[1], reverse=True)
            )

            if factor_pcts:
                top = max(factor_pcts, key=factor_pcts.get)
                result['top_risk_factor'] = top
                result['top_risk_pct'] = factor_pcts[top]

            # ---- Radiation exposure ----
            if 'Radiation_Exposure' in tumor_df.columns:
                rad = (
                    tumor_df['Radiation_Exposure']
                    .astype(str).str.strip()
                    .value_counts(dropna=True)
                    .to_dict()
                )
                result['radiation_distribution'] = {str(k): int(v) for k, v in rad.items()}

            # ---- Genetic risk ----
            if 'Genetic_Risk' in tumor_df.columns:
                gr = pd.to_numeric(tumor_df['Genetic_Risk'], errors='coerce').dropna()
                if len(gr):
                    result['genetic_mean'] = round(float(gr.mean()), 1)
                    result['genetic_median'] = round(float(gr.median()), 1)

            # ---- Gender distribution ----
            if 'Gender' in df.columns:
                result['gender_distribution'] = (
                    tumor_df['Gender'].astype(str).str.strip()
                    .value_counts(dropna=True)
                    .head(5)
                    .to_dict()
                )

            # ---- Top 5 countries ----
            if 'Country' in df.columns:
                result['country_top5'] = (
                    tumor_df['Country'].astype(str).str.strip()
                    .value_counts(dropna=True)
                    .head(5)
                    .items()
                )
                result['country_top5'] = [
                    (k, int(v)) for k, v in
                    tumor_df['Country'].astype(str).str.strip()
                    .value_counts(dropna=True)
                    .head(5)
                    .items()
                ]

            result['loaded'] = True

        except Exception as e:
            st.warning(f"Could not fully load population insights: {str(e)}")

        return result


# ==================== CLINICAL DECISION SUPPORT ====================
class ClinicalSupport:
    """Provides clinical insights and recommendations"""

    @staticmethod
    def get_recommendation(tumor_class: str, confidence: float) -> Dict:
        """
        Generate clinical recommendations based on detection results
        """
        recommendation = {
            "action": "Consult Specialist",
            "treatment": "Further Evaluation Required",
            "urgency": "🟡 ROUTINE",
            "urgency_level": "Routine",
            "similar_cases": 0,
            "csv_type": "Unknown",
            "severity": "Unknown",
            "next_steps": [],
            "survival_rate": None,
            "avg_tumor_size": None,
            "common_location": None
        }

        if confidence >= SEVERITY_THRESHOLDS['critical']:
            severity_filter = 'Severe'
            urgency_level = "🚨 CRITICAL"
            urgency_text = "Critical"
        elif confidence >= SEVERITY_THRESHOLDS['urgent']:
            severity_filter = 'Moderate'
            urgency_level = "⚠️ URGENT"
            urgency_text = "Urgent"
        else:
            severity_filter = 'Mild'
            urgency_level = "🟡 ROUTINE"
            urgency_text = "Routine"

        recommendation.update({
            "urgency": urgency_level,
            "urgency_level": urgency_text,
            "severity": severity_filter
        })

        tumor_lower = tumor_class.lower()
        if 'glioma' in tumor_lower:
            csv_type = 'Malignant'
            recommendation["next_steps"] = [
                "Immediate oncology consultation",
                "MRI with contrast enhancement",
                "Biopsy for histological confirmation",
                "Begin pre-operative planning"
            ]
        elif 'meningioma' in tumor_lower or 'pituitary' in tumor_lower:
            csv_type = 'Benign'
            recommendation["next_steps"] = [
                "Schedule follow-up MRI in 3-6 months",
                "Neurosurgery consultation if symptomatic",
                "Monitor for symptom progression",
                "Consider treatment if tumor grows"
            ]
        else:
            csv_type = 'Unknown'
            recommendation["next_steps"] = [
                "Additional imaging recommended",
                "Consult with radiologist",
                "Consider follow-up scan"
            ]

        recommendation["csv_type"] = csv_type

        if os.path.exists(PATIENT_DATA_FILE):
            try:
                cols_to_read = ['Tumor_Type', 'Symptom_Severity', 'Treatment_Received',
                                'Survival_Rate(%)', 'Tumor_Size', 'Tumor_Location']
                chunk_size = 50000
                matching_rows = []

                for chunk in pd.read_csv(PATIENT_DATA_FILE, usecols=cols_to_read, chunksize=chunk_size):
                    subset = chunk[
                        (chunk['Tumor_Type'] == csv_type) &
                        (chunk['Symptom_Severity'] == severity_filter)
                    ]
                    if not subset.empty:
                        matching_rows.append(subset)

                if matching_rows:
                    combined_subset = pd.concat(matching_rows, ignore_index=True)
                    total_matches = len(combined_subset)

                    common_treatment = combined_subset['Treatment_Received'].mode()
                    if len(common_treatment) > 0:
                        treatment = common_treatment[0]
                        if pd.isna(treatment) or str(treatment).lower() in ["none", "nan", ""]:
                            treatment = "Observation & Monitoring"
                    else:
                        treatment = "Standard Protocol"

                    avg_survival = combined_subset['Survival_Rate(%)'].mean()
                    avg_size = combined_subset['Tumor_Size'].mean()
                    common_location = combined_subset['Tumor_Location'].mode()
                    location_str = common_location[0] if len(common_location) > 0 else "Various"

                    recommendation.update({
                        "action": f"Initiate {treatment}",
                        "treatment": treatment,
                        "similar_cases": total_matches,
                        "survival_rate": round(avg_survival, 1) if pd.notna(avg_survival) else None,
                        "avg_tumor_size": round(avg_size, 2) if pd.notna(avg_size) else None,
                        "common_location": location_str
                    })

            except Exception:
                pass

        return recommendation


# ==================== ANALYTICS & VISUALIZATION ====================
class Analytics:
    """Handles all analytics and visualization"""

    @staticmethod
    def create_prevalence_chart(df: pd.DataFrame) -> go.Figure:
        if df.empty or 'Tumor_Class' not in df.columns:
            return go.Figure()

        counts = df['Tumor_Class'].value_counts().reset_index()
        counts.columns = ['Tumor Type', 'Count']
        counts['Display Name'] = counts['Tumor Type'].apply(
            lambda x: CLASS_DISPLAY_NAMES.get(x.lower(), x)
        )

        colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6']

        fig = go.Figure(data=[go.Pie(
            labels=counts['Display Name'],
            values=counts['Count'],
            hole=0.5,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig.update_layout(
            title="Disease Distribution Across All Scans",
            showlegend=True,
            height=400,
            annotations=[dict(text=f'Total<br>{counts["Count"].sum()}', x=0.5, y=0.5,
                              font_size=20, showarrow=False)]
        )
        return fig

    @staticmethod
    def create_confidence_boxplot(df: pd.DataFrame) -> go.Figure:
        tumor_only = df[df['Confidence'] > 0].copy()
        if tumor_only.empty:
            return go.Figure()

        tumor_only['Display Name'] = tumor_only['Tumor_Class'].apply(
            lambda x: CLASS_DISPLAY_NAMES.get(x.lower(), x)
        )

        fig = px.box(
            tumor_only,
            x='Display Name',
            y='Confidence',
            color='Display Name',
            title="AI Confidence Distribution by Tumor Type",
            labels={'Display Name': 'Tumor Type', 'Confidence': 'Confidence Score'}
        )
        fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
        return fig

    @staticmethod
    def create_timeline_chart(df: pd.DataFrame) -> go.Figure:
        if df.empty or 'Scan_Date' not in df.columns:
            return go.Figure()

        df_copy = df.copy()
        df_copy['Scan_Date'] = pd.to_datetime(df_copy['Scan_Date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['Scan_Date'])

        daily_trend = df_copy.groupby(df_copy['Scan_Date'].dt.date).size().reset_index(name='Scans')
        daily_trend.columns = ['Date', 'Scans']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_trend['Date'],
            y=daily_trend['Scans'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8),
            name='Daily Scans',
            hovertemplate='<b>Date:</b> %{x}<br><b>Scans:</b> %{y}<extra></extra>'
        ))
        fig.update_layout(
            title="Diagnostic Volume Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Scans",
            height=400,
            hovermode='x unified'
        )
        return fig

    @staticmethod
    def create_stacked_bar(df: pd.DataFrame) -> go.Figure:
        if df.empty:
            return go.Figure()

        df_copy = df.copy()
        df_copy['Scan_Date'] = pd.to_datetime(df_copy['Scan_Date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['Scan_Date'])

        daily_class = df_copy.groupby([
            df_copy['Scan_Date'].dt.date, 'Tumor_Class'
        ]).size().reset_index(name='Count')
        daily_class.columns = ['Date', 'Tumor_Class', 'Count']

        fig = px.bar(
            daily_class,
            x='Date',
            y='Count',
            color='Tumor_Class',
            barmode='stack',
            title="Daily Tumor Detection Breakdown"
        )
        fig.update_layout(height=400)
        return fig

    @staticmethod
    def create_confidence_histogram(df: pd.DataFrame) -> go.Figure:
        detections = df[df['Confidence'] > 0].copy()
        if detections.empty:
            return go.Figure()

        fig = px.histogram(
            detections,
            x="Confidence",
            nbins=20,
            color="Tumor_Class",
            title="Model Confidence Distribution",
            labels={'Confidence': 'Confidence Score', 'count': 'Frequency'}
        )
        fig.update_layout(height=400, bargap=0.1)
        return fig

    # ---- NEW: Population insight charts ----
    @staticmethod
    def create_age_by_tumor_chart(avg_age_by_type: Dict) -> go.Figure:
        """Bar chart of average patient age grouped by tumor type"""
        if not avg_age_by_type:
            return go.Figure()

        labels = list(avg_age_by_type.keys())
        values = list(avg_age_by_type.values())

        color_map = {
            'Benign':    '#10b981',
            'Malignant': '#ef4444',
        }
        colors = [color_map.get(l, '#3b82f6') for l in labels]

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f} yrs" for v in values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Average Age: %{y:.1f} years<extra></extra>'
        ))
        fig.update_layout(
            title="Average Patient Age by Tumor Type",
            xaxis_title="Tumor Type",
            yaxis_title="Average Age (Years)",
            height=380,
            yaxis=dict(range=[0, max(values) * 1.2 if values else 80])
        )
        return fig

    @staticmethod
    def create_risk_factor_chart(risk_factor_pcts: Dict) -> go.Figure:
        """Horizontal bar chart of risk factor prevalence"""
        if not risk_factor_pcts:
            return go.Figure()

        labels = list(risk_factor_pcts.keys())
        values = list(risk_factor_pcts.values())

        colors = [
            '#ef4444' if v >= 50 else '#f59e0b' if v >= 30 else '#10b981'
            for v in values
        ]

        fig = go.Figure(go.Bar(
            y=labels,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f"{v}%" for v in values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Prevalence: %{x:.1f}%<extra></extra>'
        ))
        fig.update_layout(
            title="Risk Factor Prevalence Among Tumor Patients",
            xaxis_title="Prevalence (%)",
            xaxis=dict(range=[0, 110]),
            height=380,
            margin=dict(l=160)
        )
        return fig

    @staticmethod
    def create_radiation_chart(radiation_distribution: Dict) -> go.Figure:
        """Donut chart of radiation exposure levels"""
        if not radiation_distribution:
            return go.Figure()

        labels = list(radiation_distribution.keys())
        values = list(radiation_distribution.values())

        color_map = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'}
        colors = [color_map.get(l, '#3b82f6') for l in labels]

        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
        ))
        fig.update_layout(
            title="Radiation Exposure Distribution (Tumor Patients)",
            height=380
        )
        return fig

    @staticmethod
    def create_gender_chart(gender_distribution: Dict) -> go.Figure:
        """Donut chart of gender split among tumor patients"""
        if not gender_distribution:
            return go.Figure()

        labels = list(gender_distribution.keys())
        values = list(gender_distribution.values())

        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=['#3b82f6', '#ec4899', '#8b5cf6']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
        ))
        fig.update_layout(
            title="Gender Distribution Among Tumor Patients",
            height=380
        )
        return fig


# ==================== INITIALIZE SYSTEM ====================
DataWarehouse.init_warehouse()
model = load_model()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("⚙️ System Controls")

    st.markdown("---")
    st.markdown("### 🎯 AI Configuration")

    conf_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower values detect more tumors but may increase false positives"
    )

    st.progress(conf_threshold)
    st.caption(f"Current: {conf_threshold:.0%} confidence required")

    st.markdown("---")
    st.markdown("### 📊 System Architecture")

    with st.expander("View Technical Details", expanded=False):
        st.markdown("""
        **Data Pipeline:**
        - 🔄 **OLTP**: Real-time transaction processing
        - ⭐ **Star Schema**: Dimensional modeling
        - 📈 **OLAP**: Multi-dimensional analysis
        - 🤖 **ML Integration**: YOLOv8 detection

        **Data Warehouse Layers:**
        - Fact Table: Detection metrics
        - Time Dimension: Temporal analysis
        - Tumor Dimension: Classification
        - Patient Dimension: Demographics
        """)

    st.markdown("---")
    st.markdown("### 🗄️ Data Management")

    if st.button("📊 View Warehouse Stats", use_container_width=True):
        df = DataWarehouse.load_warehouse()
        if not df.empty:
            st.metric("Total Records", len(df))
            st.metric("Total Scans", df['Image_Name'].nunique())
            st.metric("Positive Detections", len(df[df['Confidence'] > 0]))
        else:
            st.info("No data yet")

    if st.button("🗑️ Reset Warehouse", use_container_width=True, type="secondary"):
        if os.path.exists(WAREHOUSE_FILE):
            os.remove(WAREHOUSE_FILE)
            DataWarehouse.init_warehouse()
            st.success("✅ Warehouse reset successfully!")
            st.rerun()

    st.markdown("---")
    st.caption("NeuroScan AI v2.0")
    st.caption("Powered by YOLOv8 & Streamlit")

# ==================== MAIN APPLICATION ====================
st.title("🧠 NeuroScan AI - Medical Intelligence Platform")
st.markdown("### Advanced Brain Tumor Detection & Clinical Decision Support System")

if model is None:
    st.error("⚠️ AI model not loaded. Please check model file and restart.")
    st.stop()

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 AI Diagnosis (OLTP)",
    "🗄️ Data Warehouse (Star Schema)",
    "📈 Analytics Dashboard (OLAP)",
    "👥 Population Insights",
    "📋 Export & Reports"
])

# ==================== TAB 1: DIAGNOSIS ====================
with tab1:
    st.header("Clinical Diagnosis Workflow")

    col_upload, col_info = st.columns([1, 1])

    with col_info:
        st.info("""
        **Diagnostic Protocol:**
        1. Upload patient MRI scan (JPG/PNG)
        2. AI processes image in real-time
        3. Results stored in data warehouse
        4. Clinical recommendations generated
        5. Optional: Link to patient ID
        """)

        patient_id = st.text_input(
            "Patient ID (Optional)",
            placeholder="e.g., PT-2024-001",
            help="Link this scan to a patient record"
        )

    with col_upload:
        st.markdown("#### 📤 Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose an MRI image file",
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, PNG, JPEG"
        )

    if uploaded_file is not None:
        col_input, col_output = st.columns(2)

        with col_input:
            st.markdown("#### 🖼️ Input Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f'Source: {uploaded_file.name}', use_container_width=True)

            st.markdown(f"""
            **Image Details:**
            - **File**: {uploaded_file.name}
            - **Size**: {image.size[0]} x {image.size[1]} pixels
            - **Format**: {image.format}
            """)

        with col_output:
            st.markdown("#### 🤖 AI Analysis")

            if st.button("⚡ Run AI Diagnosis", type="primary", use_container_width=True):
                with st.spinner('🔄 Processing with AI model...'):
                    try:
                        results = model.predict(
                            image,
                            conf=conf_threshold,
                            augment=True,
                            verbose=False
                        )

                        res_plotted = results[0].plot()
                        st.image(
                            res_plotted,
                            caption='AI Detection Result',
                            use_container_width=True
                        )

                        etl_result = DataWarehouse.update_warehouse(
                            uploaded_file.name,
                            results,
                            patient_id if patient_id else None
                        )

                        if etl_result['success']:
                            st.success(f"✅ Scan processed | {etl_result['records_added']} record(s) saved")

                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.stop()

        st.markdown("---")

        if 'results' in locals():
            detections = len(results[0].boxes)

            if detections > 0:
                st.markdown("### 🏥 Clinical Assessment")

                # Load population insights once for context
                pop = PopulationInsights.load_insights()

                for idx, box in enumerate(results[0].boxes):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    recommendation = ClinicalSupport.get_recommendation(class_name, confidence)

                    if "CRITICAL" in recommendation['urgency']:
                        st.error(f"🚨 **CRITICAL FINDING**: {CLASS_DISPLAY_NAMES.get(class_name.lower(), class_name).upper()}")
                    elif "URGENT" in recommendation['urgency']:
                        st.warning(f"⚠️ **URGENT ATTENTION**: {CLASS_DISPLAY_NAMES.get(class_name.lower(), class_name).upper()}")
                    else:
                        st.success(f"✅ **Detection #{idx+1}**: {CLASS_DISPLAY_NAMES.get(class_name.lower(), class_name)}")

                    met1, met2, met3, met4 = st.columns(4)
                    met1.metric("Confidence", f"{confidence:.1%}")
                    met2.metric("Urgency", recommendation['urgency_level'])
                    met3.metric("Severity", recommendation['severity'])
                    met4.metric("Similar Cases", f"{recommendation['similar_cases']:,}")

                    # Historical DB statistics
                    if recommendation['similar_cases'] > 0 and recommendation['survival_rate'] is not None:
                        st.markdown("#### 📊 Historical Data Insights (from 250K+ patient records)")
                        db_col1, db_col2, db_col3 = st.columns(3)

                        db_col1.metric(
                            "Avg Survival Rate",
                            f"{recommendation['survival_rate']:.1f}%"
                        )
                        if recommendation['avg_tumor_size'] is not None:
                            db_col2.metric(
                                "Avg Tumor Size",
                                f"{recommendation['avg_tumor_size']:.2f} cm"
                            )
                        if recommendation['common_location']:
                            db_col3.metric(
                                "Common Location",
                                recommendation['common_location']
                            )

                    # ---- NEW: Population context for this tumor class ----
                    if pop['loaded']:
                        csv_type = recommendation['csv_type']
                        st.markdown("#### 👥 Population Context")
                        pc1, pc2, pc3 = st.columns(3)

                        # Average age for this tumor type
                        age_for_type = pop['avg_age_by_type'].get(csv_type)
                        overall_age = pop['overall_avg_age']
                        pc1.metric(
                            f"Avg Age ({csv_type} Tumors)",
                            f"{age_for_type} yrs" if age_for_type else "N/A",
                            delta=f"Overall avg: {overall_age} yrs" if overall_age else None,
                            delta_color="off"
                        )

                        # Top risk factor
                        if pop['top_risk_factor']:
                            pc2.metric(
                                "Top Risk Factor",
                                pop['top_risk_factor'],
                                delta=f"{pop['top_risk_pct']}% prevalence",
                                delta_color="off"
                            )

                        # Genetic risk
                        if pop['genetic_mean'] is not None:
                            pc3.metric(
                                "Avg Genetic Risk Score",
                                f"{pop['genetic_mean']}",
                                delta=f"Median: {pop['genetic_median']}"
                            )

                        st.markdown("---")

                    st.markdown(f"""
                    <div class="custom-card">
                        <h4>📋 Clinical Recommendation</h4>
                        <p><strong>Primary Action:</strong> {recommendation['action']}</p>
                        <p><strong>Treatment Protocol:</strong> {recommendation['treatment']}</p>
                        <p><strong>Classification:</strong> {recommendation['csv_type']} Tumor</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if recommendation['next_steps']:
                        with st.expander("📝 Recommended Next Steps", expanded=False):
                            for step in recommendation['next_steps']:
                                st.markdown(f"- {step}")

                    with st.expander("🔬 View Clinical Reasoning", expanded=False):
                        st.markdown(f"""
                        **Analysis Summary:**

                        1. **Detection Trigger**: AI identified **{class_name.upper()}** with **{confidence:.1%}** confidence
                        2. **Severity Assessment**: Classified as **{recommendation['severity']}** based on confidence threshold
                        3. **Historical Match**: {recommendation['similar_cases']:,} similar cases found in 250,000+ patient database
                        4. **Tumor Classification**: Categorized as **{recommendation['csv_type']}** tumor type
                        5. **Protocol Selection**: **{recommendation['treatment']}** is standard care for this profile
                        """)

                        if recommendation['similar_cases'] > 0:
                            st.markdown(f"""
                        **Database Insights:**
                        - **Survival Rate**: Average {recommendation['survival_rate']:.1f}% in similar cases
                        - **Tumor Size**: Average {recommendation['avg_tumor_size']:.2f} cm in matched patients
                        - **Common Location**: {recommendation['common_location']} (most frequent in this profile)
                        """)

                        st.markdown("""
                        **Risk Stratification:**
                        - Confidence ≥85%: Critical (immediate action)
                        - Confidence 50-85%: Urgent (priority scheduling)
                        - Confidence <50%: Routine (standard follow-up)
                        """)

                    st.markdown("---")
            else:
                st.success("✅ **No Abnormalities Detected**")
                st.info("""
                The AI model did not detect any tumors above the current confidence threshold.

                **Recommendations:**
                - Consider routine follow-up screening
                - If symptoms persist, consult with neurologist
                - Lower detection threshold if high suspicion
                """)

# ==================== TAB 2: DATA WAREHOUSE ====================
with tab2:
    st.header("📊 Data Warehouse Architecture (Star Schema)")

    df = DataWarehouse.load_warehouse()

    if not df.empty:
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Total Records", f"{len(df):,}")
        col_stat2.metric("Unique Scans", f"{df['Image_Name'].nunique():,}")
        col_stat3.metric("Positive Cases", f"{len(df[df['Confidence'] > 0]):,}")
        col_stat4.metric("Date Range", f"{df['Scan_Date'].nunique()} days")

        st.markdown("---")

        st.subheader("1️⃣ Fact Table: `Fact_Tumor_Detections`")
        st.caption("Central fact table containing all detection measurements")

        display_df = df.copy()
        if 'Confidence' in display_df.columns:
            display_df['Confidence'] = display_df['Confidence'].apply(
                lambda x: f"{x:.2%}" if pd.notnull(x) and x > 0 else "N/A"
            )

        st.dataframe(display_df, use_container_width=True, height=400)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Fact Table (CSV)",
            data=csv,
            file_name=f"fact_table_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        st.markdown("---")

        dim_col1, dim_col2, dim_col3 = st.columns(3)

        with dim_col1:
            st.subheader("2️⃣ Dimension: `Dim_Time`")
            st.caption("Temporal hierarchy for time-based analysis")

            if 'Scan_Date' in df.columns:
                dim_time = df[['Scan_Date']].copy()
                dim_time['Scan_Date'] = pd.to_datetime(dim_time['Scan_Date'], errors='coerce')
                dim_time = dim_time.dropna()

                if not dim_time.empty:
                    dim_time['Year'] = dim_time['Scan_Date'].dt.year
                    dim_time['Month'] = dim_time['Scan_Date'].dt.month
                    dim_time['Month_Name'] = dim_time['Scan_Date'].dt.strftime('%B')
                    dim_time['Day'] = dim_time['Scan_Date'].dt.day
                    dim_time['Day_of_Week'] = dim_time['Scan_Date'].dt.strftime('%A')

                    st.dataframe(
                        dim_time.drop_duplicates().head(10),
                        use_container_width=True,
                        height=300
                    )

        with dim_col2:
            st.subheader("3️⃣ Dimension: `Dim_Tumor`")
            st.caption("Tumor classification hierarchy")

            if 'Tumor_Class' in df.columns:
                dim_tumor = df[['Tumor_Class']].drop_duplicates().reset_index(drop=True)
                dim_tumor['Tumor_ID'] = range(1, len(dim_tumor) + 1)
                dim_tumor['Category'] = dim_tumor['Tumor_Class'].apply(
                    lambda x: 'Malignant' if 'glioma' in x.lower()
                    else 'Benign' if x != 'No Detection'
                    else 'Healthy'
                )

                st.dataframe(
                    dim_tumor[['Tumor_ID', 'Tumor_Class', 'Category']],
                    use_container_width=True,
                    height=300
                )

        with dim_col3:
            st.subheader("4️⃣ Dimension: `Dim_Severity`")
            st.caption("Clinical severity classification")

            if 'Severity_Level' in df.columns:
                dim_severity = df[['Severity_Level']].drop_duplicates().reset_index(drop=True)
                dim_severity['Severity_ID'] = range(1, len(dim_severity) + 1)
                dim_severity['Priority'] = dim_severity['Severity_Level'].map({
                    'Critical': 1, 'Urgent': 2, 'Routine': 3, 'None': 4
                })

                st.dataframe(
                    dim_severity.sort_values('Priority'),
                    use_container_width=True,
                    height=300
                )
    else:
        st.info("📭 **Data warehouse is empty**")
        st.markdown("""
        No records found in the warehouse. To populate:
        1. Navigate to **AI Diagnosis** tab
        2. Upload and process MRI scans
        3. Return here to view stored data
        """)

# ==================== TAB 3: ANALYTICS ====================
with tab3:
    st.header("📈 Strategic Analytics Dashboard (OLAP)")

    df = DataWarehouse.load_warehouse()

    if not df.empty:
        st.subheader("🎯 Key Performance Indicators")

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        total_scans = len(df)
        positive_cases = len(df[df['Confidence'] > 0])
        detection_rate = (positive_cases / total_scans * 100) if total_scans > 0 else 0
        avg_confidence = df[df['Confidence'] > 0]['Confidence'].mean() if positive_cases > 0 else 0
        healthy_scans = len(df[df['Tumor_Class'] == 'No Detection'])

        kpi1.metric("Total Scans", f"{total_scans:,}")
        kpi2.metric("Positive Detections", f"{positive_cases:,}")
        kpi3.metric("Detection Rate", f"{detection_rate:.1f}%")
        kpi4.metric("Avg Confidence", f"{avg_confidence:.1%}")
        kpi5.metric("Healthy Scans", f"{healthy_scans:,}")

        st.markdown("---")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.subheader("📊 Disease Prevalence Analysis")
            fig_prevalence = Analytics.create_prevalence_chart(df)
            st.plotly_chart(fig_prevalence, use_container_width=True, key="prevalence")

        with viz_col2:
            st.subheader("🎯 AI Confidence Distribution")
            fig_confidence = Analytics.create_confidence_boxplot(df)
            st.plotly_chart(fig_confidence, use_container_width=True, key="confidence_box")

        st.markdown("---")

        st.subheader("📅 Temporal Analysis")
        fig_timeline = Analytics.create_timeline_chart(df)
        st.plotly_chart(fig_timeline, use_container_width=True, key="timeline")

        st.markdown("---")

        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            st.subheader("📈 Daily Tumor Composition")
            fig_stacked = Analytics.create_stacked_bar(df)
            st.plotly_chart(fig_stacked, use_container_width=True, key="stacked")

        with viz_col4:
            st.subheader("🔍 Model Calibration Analysis")
            fig_histogram = Analytics.create_confidence_histogram(df)
            st.plotly_chart(fig_histogram, use_container_width=True, key="histogram")

        st.markdown("---")

        st.subheader("📊 Statistical Summary")

        if positive_cases > 0:
            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.markdown("**Tumor Type Distribution**")
                tumor_dist = df[df['Confidence'] > 0]['Tumor_Class'].value_counts()
                st.dataframe(
                    tumor_dist.reset_index().rename(
                        columns={'index': 'Tumor Type', 'Tumor_Class': 'Count'}
                    ),
                    use_container_width=True
                )

            with summary_col2:
                st.markdown("**Severity Level Breakdown**")
                if 'Severity_Level' in df.columns:
                    severity_dist = df[df['Confidence'] > 0]['Severity_Level'].value_counts()
                    st.dataframe(
                        severity_dist.reset_index().rename(
                            columns={'index': 'Severity', 'Severity_Level': 'Count'}
                        ),
                        use_container_width=True
                    )
    else:
        st.info("📊 **No data available for analysis**")
        st.markdown("""
        Analytics dashboard requires processed scans. Get started:
        - Process MRI scans in the Diagnosis tab
        - Data will automatically appear here
        - Visualizations update in real-time
        """)

# ==================== TAB 4: POPULATION INSIGHTS (NEW) ====================
with tab4:
    st.header("👥 Population Insights from Historical Patient Records")
    st.markdown(
        "Analysis of **250,000+ historical patient records** to understand tumor demographics, "
        "common causes, and risk profiles."
    )

    with st.spinner("Loading population data..."):
        pop = PopulationInsights.load_insights()

    if not pop['loaded']:
        st.warning("""
        ⚠️ **patient_records.csv not found.**

        Please ensure `patient_records.csv` is in the same directory as `app.py`.
        """)
    else:
        # ---- Top-level KPIs ----
        st.subheader("📊 Dataset Overview")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Patients", f"{pop['total_patients']:,}")
        k2.metric("Tumor Patients", f"{pop['tumor_patients']:,}")
        k3.metric(
            "Overall Avg Age",
            f"{pop['overall_avg_age']} yrs" if pop['overall_avg_age'] else "N/A"
        )
        age_range = pop.get('age_range_tumor')
        k4.metric(
            "Age Range (Tumor)",
            f"{age_range[0]}–{age_range[1]} yrs" if age_range else "N/A"
        )

        st.markdown("---")

        # ---- Average Age by Tumor Type ----
        st.subheader("🎂 Average Patient Age by Tumor Type")

        age_col1, age_col2 = st.columns([2, 1])

        with age_col1:
            fig_age = Analytics.create_age_by_tumor_chart(pop['avg_age_by_type'])
            st.plotly_chart(fig_age, use_container_width=True, key="age_tumor")

        with age_col2:
            st.markdown("**Age Breakdown**")
            if pop['avg_age_by_type']:
                for tumor_type, avg_age in pop['avg_age_by_type'].items():
                    icon = "🔴" if tumor_type == "Malignant" else "🟢"
                    st.markdown(
                        f"{icon} **{tumor_type}**: {avg_age:.1f} years average"
                    )
                st.markdown("---")
            if pop['overall_avg_age']:
                st.info(
                    f"**Overall average age** of patients diagnosed with brain tumors: "
                    f"**{pop['overall_avg_age']} years**"
                )
            if age_range:
                st.info(
                    f"**Age range** spans from **{age_range[0]}** to **{age_range[1]}** years, "
                    f"indicating tumors can affect patients across a wide age spectrum."
                )

        st.markdown("---")

        # ---- Risk Factors (Causes) ----
        st.subheader("⚠️ Common Risk Factors & Causes")
        st.caption(
            "Prevalence of known risk factors among patients diagnosed with brain tumors. "
            "Higher prevalence suggests stronger association."
        )

        rf_col1, rf_col2 = st.columns([2, 1])

        with rf_col1:
            fig_risk = Analytics.create_risk_factor_chart(pop['risk_factor_pcts'])
            st.plotly_chart(fig_risk, use_container_width=True, key="risk_factors")

        with rf_col2:
            st.markdown("**Risk Factor Summary**")
            for factor, pct in pop['risk_factor_pcts'].items():
                if pct >= 50:
                    level_label = "🔴 High"
                elif pct >= 30:
                    level_label = "🟡 Moderate"
                else:
                    level_label = "🟢 Low"
                st.markdown(f"{level_label} — **{factor}**: {pct}%")

            st.markdown("---")
            if pop['top_risk_factor']:
                st.warning(
                    f"**Most Common Risk Factor:** {pop['top_risk_factor']} "
                    f"({pop['top_risk_pct']}% of tumor patients)"
                )

        st.markdown("---")

        # ---- Radiation & Genetic Risk ----
        rad_col1, rad_col2 = st.columns(2)

        with rad_col1:
            st.subheader("☢️ Radiation Exposure Distribution")
            if pop['radiation_distribution']:
                fig_rad = Analytics.create_radiation_chart(pop['radiation_distribution'])
                st.plotly_chart(fig_rad, use_container_width=True, key="radiation")

                total_rad = sum(pop['radiation_distribution'].values())
                high_rad = pop['radiation_distribution'].get('High', 0)
                if total_rad > 0:
                    high_pct = high_rad / total_rad * 100
                    st.caption(
                        f"**{high_pct:.1f}%** of tumor patients had high radiation exposure."
                    )
            else:
                st.info("Radiation exposure data not available.")

        with rad_col2:
            st.subheader("🧬 Genetic Risk Profile")
            if pop['genetic_mean'] is not None:
                g1, g2 = st.columns(2)
                g1.metric("Mean Genetic Risk Score", f"{pop['genetic_mean']}")
                g2.metric("Median Genetic Risk Score", f"{pop['genetic_median']}")

                st.markdown("""
                **Interpretation:**
                - Score 0–33: Low genetic predisposition
                - Score 34–66: Moderate genetic predisposition
                - Score 67–100: High genetic predisposition
                """)

                risk_level = (
                    "🔴 **High**" if pop['genetic_mean'] > 66
                    else "🟡 **Moderate**" if pop['genetic_mean'] > 33
                    else "🟢 **Low**"
                )
                st.info(
                    f"The average tumor patient has {risk_level} genetic predisposition "
                    f"(mean score: {pop['genetic_mean']})."
                )
            else:
                st.info("Genetic risk data not available.")

        st.markdown("---")

        # ---- Gender Distribution ----
        if pop['gender_distribution']:
            st.subheader("👤 Gender Distribution Among Tumor Patients")
            gen_col1, gen_col2 = st.columns([2, 1])

            with gen_col1:
                fig_gender = Analytics.create_gender_chart(pop['gender_distribution'])
                st.plotly_chart(fig_gender, use_container_width=True, key="gender")

            with gen_col2:
                st.markdown("**Gender Breakdown**")
                total_gen = sum(pop['gender_distribution'].values())
                for gender, count in pop['gender_distribution'].items():
                    pct = count / total_gen * 100 if total_gen > 0 else 0
                    st.markdown(f"**{gender}**: {count:,} ({pct:.1f}%)")

        st.markdown("---")

        # ---- Top Countries ----
        if pop['country_top5']:
            st.subheader("🌍 Top 5 Countries by Tumor Cases")
            country_df = pd.DataFrame(pop['country_top5'], columns=['Country', 'Cases'])
            fig_country = px.bar(
                country_df,
                x='Country',
                y='Cases',
                color='Cases',
                color_continuous_scale='blues',
                title="Countries with Highest Reported Brain Tumor Cases"
            )
            fig_country.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_country, use_container_width=True, key="countries")

        st.markdown("---")

        # ---- Key Takeaways ----
        st.subheader("💡 Key Insights Summary")

        insights_text = []
        if pop['overall_avg_age']:
            insights_text.append(
                f"🎂 The average age of brain tumor patients in this dataset is "
                f"**{pop['overall_avg_age']} years**."
            )
        if pop['avg_age_by_type']:
            for ttype, age in pop['avg_age_by_type'].items():
                insights_text.append(f"🔬 **{ttype}** tumor patients average **{age} years** old.")
        if pop['top_risk_factor']:
            insights_text.append(
                f"⚠️ The most prevalent risk factor is **{pop['top_risk_factor']}**, "
                f"affecting **{pop['top_risk_pct']}%** of tumor patients."
            )
        if pop['genetic_mean']:
            insights_text.append(
                f"🧬 The average genetic risk score among tumor patients is **{pop['genetic_mean']}** "
                f"(scale 0–100), indicating "
                f"{'high' if pop['genetic_mean'] > 66 else 'moderate' if pop['genetic_mean'] > 33 else 'low'} "
                f"hereditary predisposition."
            )
        if pop['radiation_distribution']:
            total_r = sum(pop['radiation_distribution'].values())
            high_r = pop['radiation_distribution'].get('High', 0)
            if total_r > 0:
                insights_text.append(
                    f"☢️ **{high_r/total_r*100:.1f}%** of tumor patients had high radiation exposure."
                )

        for insight in insights_text:
            st.markdown(f"- {insight}")

# ==================== TAB 5: EXPORT & REPORTS ====================
with tab5:
    st.header("📋 Export & Reporting")

    df = DataWarehouse.load_warehouse()

    if not df.empty:
        st.subheader("📥 Download Options")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.markdown("#### Complete Dataset")

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"neuroscan_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            try:
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Detections', index=False)

                    summary_data = {
                        'Metric': ['Total Scans', 'Positive Detections', 'Avg Confidence'],
                        'Value': [
                            len(df),
                            len(df[df['Confidence'] > 0]),
                            f"{df[df['Confidence'] > 0]['Confidence'].mean():.2%}"
                            if len(df[df['Confidence'] > 0]) > 0 else 'N/A'
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                st.download_button(
                    label="📊 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"neuroscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.caption("Excel export requires openpyxl package")

        with export_col2:
            st.markdown("#### Filtered Export")

            if 'Scan_Date' in df.columns:
                df['Scan_Date'] = pd.to_datetime(df['Scan_Date'], errors='coerce')
                df_filtered = df.dropna(subset=['Scan_Date'])

                if not df_filtered.empty:
                    min_date = df_filtered['Scan_Date'].min().date()
                    max_date = df_filtered['Scan_Date'].max().date()

                    date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )

                    if len(date_range) == 2:
                        mask = (
                            (df_filtered['Scan_Date'].dt.date >= date_range[0]) &
                            (df_filtered['Scan_Date'].dt.date <= date_range[1])
                        )
                        filtered_data = df_filtered[mask]

                        st.info(f"📊 {len(filtered_data)} records in selected range")

                        csv_filtered = filtered_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📄 Download Filtered Data",
                            data=csv_filtered,
                            file_name=f"neuroscan_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        st.markdown("---")

        st.subheader("📊 Generate Summary Report")

        if st.button("🔄 Generate Comprehensive Report", type="primary"):
            with st.spinner("Generating report..."):
                report_date = datetime.now().strftime("%B %d, %Y %I:%M %p")
                pop = PopulationInsights.load_insights()

                st.markdown(f"""
                ## NeuroScan AI - Diagnostic Summary Report
                **Generated:** {report_date}

                ---

                ### Executive Summary

                **Reporting Period:** {df['Scan_Date'].min()} to {df['Scan_Date'].max()}

                #### Key Metrics:
                - **Total Scans Processed:** {len(df):,}
                - **Positive Detections:** {len(df[df['Confidence'] > 0]):,}
                - **Detection Rate:** {(len(df[df['Confidence'] > 0]) / len(df) * 100):.2f}%
                - **Average AI Confidence:** {df[df['Confidence'] > 0]['Confidence'].mean():.2%}

                ---

                ### Tumor Type Distribution
                """)

                tumor_summary = df[df['Confidence'] > 0]['Tumor_Class'].value_counts()
                st.dataframe(tumor_summary, use_container_width=True)

                st.markdown("---\n### Severity Classification")

                if 'Severity_Level' in df.columns:
                    severity_summary = df['Severity_Level'].value_counts()
                    st.dataframe(severity_summary, use_container_width=True)

                # Population insights in report
                if pop['loaded']:
                    st.markdown("---\n### Population Insights (Historical Data)")
                    st.markdown(f"""
                    - **Total historical patients analysed:** {pop['total_patients']:,}
                    - **Confirmed tumor cases:** {pop['tumor_patients']:,}
                    - **Overall average age at diagnosis:** {pop['overall_avg_age']} years
                    """)

                    if pop['avg_age_by_type']:
                        st.markdown("**Average Age by Tumor Type:**")
                        for ttype, age in pop['avg_age_by_type'].items():
                            st.markdown(f"  - {ttype}: {age:.1f} years")

                    if pop['top_risk_factor']:
                        st.markdown(
                            f"\n**Most Common Risk Factor:** {pop['top_risk_factor']} "
                            f"({pop['top_risk_pct']}% prevalence)"
                        )

                    if pop['genetic_mean']:
                        st.markdown(
                            f"**Average Genetic Risk Score:** {pop['genetic_mean']} "
                            f"(median: {pop['genetic_median']})"
                        )

                st.success("✅ Report generated successfully!")
    else:
        st.info("📭 No data available for export")
        st.markdown("Process some scans first to generate reports.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p><strong>NeuroScan AI v2.0</strong> | Medical Intelligence Platform</p>
    <p>Powered by YOLOv8 Deep Learning • Built with Streamlit</p>
    <p style='font-size: 0.8rem;'>⚕️ For research and educational purposes • Consult qualified medical professionals for diagnosis</p>
</div>
""", unsafe_allow_html=True)

