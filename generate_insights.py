#!/usr/bin/env python3
"""
UIDAI Hackathon Dashboard - High-Performance Backend Script
============================================================
Multi-Part CSV Loading, Ensemble Anomaly Detection, and Ticket Generation.

Author: Principal Data Engineer & Lead Data Scientist
"""

import glob
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = Path(__file__).parent / "Data"

DATA_CONFIG = {
    'enrolment': {
        'folder': 'api_data_aadhar_enrolment',
        'pattern': '*_enrolment_*.csv',
        'columns': ['date', 'state', 'district', 'pincode', 'age_0_5', 'age_5_17', 'age_18_greater']
    },
    'biometric': {
        'folder': 'api_data_aadhar_biometric',
        'pattern': '*_biometric_*.csv',
        'columns': ['date', 'state', 'district', 'pincode', 'bio_age_5_17', 'bio_age_17_']
    },
    'demographic': {
        'folder': 'api_data_aadhar_demographic',
        'pattern': '*_demographic_*.csv',
        'columns': ['date', 'state', 'district', 'pincode', 'demo_age_5_17', 'demo_age_17_']
    }
}

# Anomaly detection configuration
ANOMALY_CONFIG = {
    'contamination': 0.1,
    'random_state': 42,
    'autoencoder_threshold_percentile': 90,
    'mbu_deficit_threshold': 500,
    'fraud_vote_threshold': 3
}


# =============================================================================
# PART 1: DATA LOADING & PREPROCESSING
# =============================================================================

def load_csv_chunks(folder: Path, pattern: str, columns: List[str]) -> pd.DataFrame:
    """
    Load all CSV chunks matching the pattern from the specified folder.
    Verifies file integrity and concatenates into a single DataFrame.
    """
    folder_path = BASE_PATH / folder
    files = glob.glob(str(folder_path / pattern))
    
    if not files:
        print(f"  ⚠️  No files found matching {pattern} in {folder_path}")
        return pd.DataFrame(columns=columns)
    
    print(f"  📂 Found {len(files)} file(s) in {folder}")
    
    dfs = []
    for file_path in sorted(files):
        try:
            df = pd.read_csv(file_path, usecols=lambda c: c in columns, low_memory=False)
            # Verify all required columns exist
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                print(f"    ⚠️  Missing columns in {Path(file_path).name}: {missing_cols}")
                continue
            dfs.append(df)
            print(f"    ✓ Loaded {Path(file_path).name}: {len(df):,} rows")
        except Exception as e:
            print(f"    ✗ Error loading {Path(file_path).name}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=columns)
    
    return pd.concat(dfs, ignore_index=True)


def clean_pincode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows where pincode is numeric and exactly 6 digits.
    """
    # Convert pincode to string and strip whitespace
    df['pincode'] = df['pincode'].astype(str).str.strip()
    
    # Filter: numeric and length == 6
    mask = df['pincode'].str.match(r'^\d{6}$', na=False)
    filtered_df = df[mask].copy()
    
    removed = len(df) - len(filtered_df)
    if removed > 0:
        print(f"    🔍 Filtered out {removed:,} rows with invalid pincodes")
    
    return filtered_df


def preprocess_dataframe(
    df: pd.DataFrame, 
    name: str, 
    numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Preprocess DataFrame: convert date, clean pincode, aggregate by location.
    """
    print(f"\n  🔧 Preprocessing {name}...")
    
    if df.empty:
        print(f"    ⚠️  Empty DataFrame, skipping preprocessing")
        return df
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    # Clean pincodes
    df = clean_pincode(df)
    
    if df.empty:
        return df
    
    # Ensure numeric columns are numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Aggregate by pincode, district, state (SUM all numeric columns)
    agg_dict = {col: 'sum' for col in numeric_cols if col in df.columns}
    
    df_agg = df.groupby(['pincode', 'district', 'state'], as_index=False).agg(agg_dict)
    
    print(f"    ✓ Aggregated to {len(df_agg):,} unique pincodes")
    
    return df_agg


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to load all three datasets.
    """
    print("\n" + "="*60)
    print("📊 PART 1: LOADING & PREPROCESSING DATA")
    print("="*60)
    
    # Load Enrolment Data
    print("\n📁 Loading Enrolment Data...")
    df_enrol = load_csv_chunks(
        DATA_CONFIG['enrolment']['folder'],
        DATA_CONFIG['enrolment']['pattern'],
        DATA_CONFIG['enrolment']['columns']
    )
    df_enrol = preprocess_dataframe(
        df_enrol, 
        'enrolment',
        ['age_0_5', 'age_5_17', 'age_18_greater']
    )
    
    # Load Biometric Data
    print("\n📁 Loading Biometric Data...")
    df_bio = load_csv_chunks(
        DATA_CONFIG['biometric']['folder'],
        DATA_CONFIG['biometric']['pattern'],
        DATA_CONFIG['biometric']['columns']
    )
    df_bio = preprocess_dataframe(
        df_bio,
        'biometric',
        ['bio_age_5_17', 'bio_age_17_']
    )
    
    # Load Demographic Data
    print("\n📁 Loading Demographic Data...")
    df_demo = load_csv_chunks(
        DATA_CONFIG['demographic']['folder'],
        DATA_CONFIG['demographic']['pattern'],
        DATA_CONFIG['demographic']['columns']
    )
    df_demo = preprocess_dataframe(
        df_demo,
        'demographic',
        ['demo_age_5_17', 'demo_age_17_']
    )
    
    print(f"\n✅ Data Loading Complete!")
    print(f"   Enrolment: {len(df_enrol):,} pincodes")
    print(f"   Biometric: {len(df_bio):,} pincodes")
    print(f"   Demographic: {len(df_demo):,} pincodes")
    
    return df_enrol, df_bio, df_demo


# =============================================================================
# PART 2: CONSENSUS VOTING ANOMALY ENGINE
# =============================================================================

def prepare_features(
    df_enrol: pd.DataFrame, 
    df_bio: pd.DataFrame, 
    df_demo: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Create feature DataFrame indexed by pincode for anomaly detection.
    Returns: df_features, scaled_features, scaler
    """
    print("\n" + "="*60)
    print("🔬 PART 2: ANOMALY DETECTION ENGINE")
    print("="*60)
    
    print("\n  📐 Preparing features for ensemble models...")
    
    # Create feature DataFrame
    # Start with enrolment (adult enrolments is our target)
    df_features = df_enrol[['pincode', 'district', 'state', 'age_18_greater']].copy()
    df_features = df_features.set_index('pincode')
    
    # Merge biometric data
    if not df_bio.empty:
        df_bio_indexed = df_bio[['pincode', 'bio_age_5_17', 'bio_age_17_']].set_index('pincode')
        df_features = df_features.join(df_bio_indexed, how='left')
    else:
        df_features['bio_age_5_17'] = 0
        df_features['bio_age_17_'] = 0
    
    # Merge demographic data
    if not df_demo.empty:
        df_demo_indexed = df_demo[['pincode', 'demo_age_5_17', 'demo_age_17_']].set_index('pincode')
        df_features = df_features.join(df_demo_indexed, how='left')
    else:
        df_features['demo_age_5_17'] = 0
        df_features['demo_age_17_'] = 0
    
    # Fill NaN values with 0
    df_features = df_features.fillna(0)
    
    # Feature columns for anomaly detection
    feature_cols = ['age_18_greater', 'bio_age_17_', 'demo_age_17_']
    X = df_features[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"    ✓ Created feature matrix: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    
    return df_features, X_scaled, scaler


def run_isolation_forest(X_scaled: np.ndarray) -> np.ndarray:
    """
    Isolation Forest anomaly detection.
    Returns: binary array (1 = anomaly, 0 = normal)
    """
    print("\n  🌲 Running Isolation Forest...")
    
    clf = IsolationForest(
        contamination=ANOMALY_CONFIG['contamination'],
        random_state=ANOMALY_CONFIG['random_state'],
        n_jobs=-1
    )
    predictions = clf.fit_predict(X_scaled)
    
    # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
    anomalies = (predictions == -1).astype(int)
    
    print(f"    ✓ Detected {anomalies.sum():,} anomalies ({100*anomalies.mean():.1f}%)")
    
    return anomalies


def run_one_class_svm(X_scaled: np.ndarray) -> np.ndarray:
    """
    One-Class SVM anomaly detection.
    Returns: binary array (1 = anomaly, 0 = normal)
    """
    print("\n  🎯 Running One-Class SVM...")
    
    clf = OneClassSVM(
        nu=ANOMALY_CONFIG['contamination'],
        kernel='rbf',
        gamma='scale'
    )
    predictions = clf.fit_predict(X_scaled)
    
    # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
    anomalies = (predictions == -1).astype(int)
    
    print(f"    ✓ Detected {anomalies.sum():,} anomalies ({100*anomalies.mean():.1f}%)")
    
    return anomalies


def run_local_outlier_factor(X_scaled: np.ndarray) -> np.ndarray:
    """
    Local Outlier Factor anomaly detection.
    Returns: binary array (1 = anomaly, 0 = normal)
    """
    print("\n  📍 Running Local Outlier Factor...")
    
    clf = LocalOutlierFactor(
        n_neighbors=20,
        contamination=ANOMALY_CONFIG['contamination'],
        n_jobs=-1
    )
    predictions = clf.fit_predict(X_scaled)
    
    # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
    anomalies = (predictions == -1).astype(int)
    
    print(f"    ✓ Detected {anomalies.sum():,} anomalies ({100*anomalies.mean():.1f}%)")
    
    return anomalies


def run_autoencoder(X_scaled: np.ndarray) -> Optional[np.ndarray]:
    """
    Autoencoder-based anomaly detection using TensorFlow/Keras.
    Returns: binary array (1 = anomaly, 0 = normal) or None if TF unavailable
    """
    print("\n  🧠 Running Autoencoder (Deep Learning)...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Suppress TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        
        # Build Autoencoder
        input_dim = X_scaled.shape[1]
        
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(8, activation='relu')(inputs)
        x = layers.Dense(4, activation='relu')(x)  # Bottleneck (Code layer)
        
        # Decoder
        x = layers.Dense(8, activation='relu')(x)
        outputs = layers.Dense(input_dim, activation='linear')(x)
        
        autoencoder = keras.Model(inputs, outputs)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train (silently)
        autoencoder.fit(
            X_scaled, X_scaled,
            epochs=50,
            batch_size=256,
            shuffle=True,
            verbose=0,
            validation_split=0.1
        )
        
        # Calculate reconstruction error (MSE per sample)
        reconstructions = autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        
        # Threshold = 90th percentile
        threshold = np.percentile(mse, ANOMALY_CONFIG['autoencoder_threshold_percentile'])
        anomalies = (mse > threshold).astype(int)
        
        print(f"    ✓ Detected {anomalies.sum():,} anomalies ({100*anomalies.mean():.1f}%)")
        print(f"    ✓ MSE Threshold: {threshold:.4f}")
        
        return anomalies
        
    except ImportError:
        print("    ⚠️  TensorFlow not available. Falling back to 3-model voting.")
        return None
    except Exception as e:
        print(f"    ⚠️  Autoencoder failed: {e}. Falling back to 3-model voting.")
        return None


def run_ensemble_voting(
    df_features: pd.DataFrame, 
    X_scaled: np.ndarray
) -> pd.DataFrame:
    """
    Run all 4 models and perform consensus voting.
    Returns: DataFrame with votes and confidence scores.
    """
    # Run all models
    votes_if = run_isolation_forest(X_scaled)
    votes_svm = run_one_class_svm(X_scaled)
    votes_lof = run_local_outlier_factor(X_scaled)
    votes_ae = run_autoencoder(X_scaled)
    
    # Combine votes
    if votes_ae is not None:
        total_votes = votes_if + votes_svm + votes_lof + votes_ae
        total_models = 4
        print(f"\n  📊 Ensemble: Using 4 models (including Autoencoder)")
    else:
        total_votes = votes_if + votes_svm + votes_lof
        total_models = 3
        print(f"\n  📊 Ensemble: Using 3 models (Autoencoder unavailable)")
    
    # Add results to DataFrame
    df_results = df_features.copy()
    df_results['votes'] = total_votes
    df_results['total_models'] = total_models
    df_results['confidence_score'] = (total_votes / total_models) * 100
    df_results['is_high_risk'] = total_votes >= ANOMALY_CONFIG['fraud_vote_threshold']
    
    high_risk_count = df_results['is_high_risk'].sum()
    print(f"\n  🚨 HIGH RISK AREAS (votes >= {ANOMALY_CONFIG['fraud_vote_threshold']}): {high_risk_count:,}")
    
    return df_results


# =============================================================================
# PART 3: BUSINESS LOGIC - TICKET GENERATION
# =============================================================================

def generate_mbu_compliance_tickets(
    df_enrol: pd.DataFrame, 
    df_bio: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Generate MBU (Mobile Biometric Update) compliance tickets.
    Deficit = Enrolment[age_5_17] - Biometric[bio_age_5_17]
    """
    print("\n  📋 Generating MBU Compliance Tickets...")
    
    tickets = []
    
    # Merge enrolment and biometric on pincode
    df_merged = df_enrol[['pincode', 'district', 'state', 'age_5_17']].merge(
        df_bio[['pincode', 'bio_age_5_17']],
        on='pincode',
        how='left'
    )
    df_merged['bio_age_5_17'] = df_merged['bio_age_5_17'].fillna(0)
    
    # Calculate deficit
    df_merged['deficit'] = df_merged['age_5_17'] - df_merged['bio_age_5_17']
    
    # Filter where deficit > threshold
    threshold = ANOMALY_CONFIG['mbu_deficit_threshold']
    df_high_deficit = df_merged[df_merged['deficit'] > threshold]
    
    for _, row in df_high_deficit.iterrows():
        ticket = {
            'pincode': row['pincode'],
            'district': row['district'],
            'state': row['state'],
            'priority': 'High',
            'task': 'MBU Camp',
            'venue': 'Schools',
            'details': f"{int(row['deficit'])} kids pending biometric update",
            'deficit': int(row['deficit']),
            'whatsapp_msg': f"🏫 MBU CAMP NEEDED\n📍 {row['district']}, {row['state']}\n📮 Pincode: {row['pincode']}\n👦 {int(row['deficit'])} children pending\n⚠️ Priority: HIGH"
        }
        tickets.append(ticket)
    
    print(f"    ✓ Generated {len(tickets):,} MBU compliance tickets")
    
    return tickets


def generate_fraud_alert_tickets(df_results: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate Fraud Alert tickets from ensemble anomaly detection.
    """
    print("\n  🚨 Generating Fraud Alert Tickets...")
    
    tickets = []
    
    df_high_risk = df_results[df_results['is_high_risk']].reset_index()
    
    for _, row in df_high_risk.iterrows():
        ticket = {
            'pincode': row['pincode'],
            'district': row['district'],
            'state': row['state'],
            'priority': 'Critical',
            'task': 'Fraud Audit',
            'venue': 'CSC Center',
            'details': f"Suspicious Adult Enrolments (Confidence: {row['confidence_score']:.0f}%)",
            'votes': int(row['votes']),
            'confidence': row['confidence_score'],
            'whatsapp_msg': f"🚨 FRAUD ALERT\n📍 {row['district']}, {row['state']}\n📮 Pincode: {row['pincode']}\n🎯 Confidence: {row['confidence_score']:.0f}%\n⚠️ Priority: CRITICAL\n📋 Action: Immediate Audit Required"
        }
        tickets.append(ticket)
    
    print(f"    ✓ Generated {len(tickets):,} fraud alert tickets")
    
    return tickets


def generate_migration_hotspot_tickets(
    df_enrol: pd.DataFrame,
    df_demo: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Generate Migration Hotspot tickets.
    Condition: demo_age_17_ in top 10% AND age_0_5 in bottom 20%
    """
    print("\n  🏙️  Generating Migration Hotspot Tickets...")
    
    tickets = []
    
    # Merge data
    df_merged = df_enrol[['pincode', 'district', 'state', 'age_0_5']].merge(
        df_demo[['pincode', 'demo_age_17_']],
        on='pincode',
        how='left'
    )
    df_merged['demo_age_17_'] = df_merged['demo_age_17_'].fillna(0)
    
    # Calculate thresholds
    demo_top_10_threshold = df_merged['demo_age_17_'].quantile(0.90)
    age_0_5_bottom_20_threshold = df_merged['age_0_5'].quantile(0.20)
    
    # Filter hotspots
    mask = (
        (df_merged['demo_age_17_'] >= demo_top_10_threshold) & 
        (df_merged['age_0_5'] <= age_0_5_bottom_20_threshold)
    )
    df_hotspots = df_merged[mask]
    
    for _, row in df_hotspots.iterrows():
        ticket = {
            'pincode': row['pincode'],
            'district': row['district'],
            'state': row['state'],
            'priority': 'Medium',
            'task': 'Urban Planning Survey',
            'venue': 'Municipal Ward',
            'details': 'High In-Migration Detected',
            'demo_adults': int(row['demo_age_17_']),
            'young_children': int(row['age_0_5']),
            'whatsapp_msg': f"📊 MIGRATION ALERT\n📍 {row['district']}, {row['state']}\n📮 Pincode: {row['pincode']}\n👥 High Adult Migration: {int(row['demo_age_17_'])}\n👶 Low Young Children: {int(row['age_0_5'])}\n📋 Action: Urban Planning Survey"
        }
        tickets.append(ticket)
    
    print(f"    ✓ Generated {len(tickets):,} migration hotspot tickets")
    
    return tickets


def generate_all_tickets(
    df_enrol: pd.DataFrame,
    df_bio: pd.DataFrame,
    df_demo: pd.DataFrame,
    df_results: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Generate all action tickets.
    """
    print("\n" + "="*60)
    print("📝 PART 3: GENERATING ACTION TICKETS")
    print("="*60)
    
    all_tickets = []
    
    # MBU Compliance tickets
    mbu_tickets = generate_mbu_compliance_tickets(df_enrol, df_bio)
    all_tickets.extend(mbu_tickets)
    
    # Fraud Alert tickets
    fraud_tickets = generate_fraud_alert_tickets(df_results)
    all_tickets.extend(fraud_tickets)
    
    # Migration Hotspot tickets
    migration_tickets = generate_migration_hotspot_tickets(df_enrol, df_demo)
    all_tickets.extend(migration_tickets)
    
    print(f"\n  ✅ Total tickets generated: {len(all_tickets):,}")
    
    return all_tickets


# =============================================================================
# PART 4: OUTPUT GENERATION
# =============================================================================

def generate_compliance_map_data(
    df_enrol: pd.DataFrame,
    df_bio: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Generate compliance map data showing deficit per pincode.
    """
    df_merged = df_enrol[['pincode', 'district', 'state', 'age_5_17']].merge(
        df_bio[['pincode', 'bio_age_5_17']],
        on='pincode',
        how='left'
    )
    df_merged['bio_age_5_17'] = df_merged['bio_age_5_17'].fillna(0)
    df_merged['deficit'] = df_merged['age_5_17'] - df_merged['bio_age_5_17']
    
    # Filter only positive deficits
    df_positive = df_merged[df_merged['deficit'] > 0]
    
    compliance_data = []
    for _, row in df_positive.iterrows():
        compliance_data.append({
            'pincode': row['pincode'],
            'district': row['district'],
            'state': row['state'],
            'deficit': int(row['deficit'])
        })
    
    return compliance_data


def generate_output(
    df_enrol: pd.DataFrame,
    df_bio: pd.DataFrame,
    df_results: pd.DataFrame,
    all_tickets: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate the final output JSON structure.
    """
    print("\n" + "="*60)
    print("💾 PART 4: GENERATING OUTPUT")
    print("="*60)
    
    # Calculate summary statistics
    total_pincodes = len(df_enrol)
    critical_alerts = sum(1 for t in all_tickets if t['priority'] == 'Critical')
    
    # Calculate MBU backlog total
    df_merged = df_enrol[['pincode', 'age_5_17']].merge(
        df_bio[['pincode', 'bio_age_5_17']],
        on='pincode',
        how='left'
    )
    df_merged['bio_age_5_17'] = df_merged['bio_age_5_17'].fillna(0)
    df_merged['deficit'] = df_merged['age_5_17'] - df_merged['bio_age_5_17']
    mbu_backlog_total = int(df_merged[df_merged['deficit'] > 0]['deficit'].sum())
    
    # Generate compliance map data
    compliance_map_data = generate_compliance_map_data(df_enrol, df_bio)
    
    # Prepare action tickets (simplified for output)
    action_tickets = []
    for ticket in all_tickets:
        action_tickets.append({
            'pincode': ticket['pincode'],
            'priority': ticket['priority'],
            'task': ticket['task'],
            'venue': ticket['venue'],
            'details': ticket['details'],
            'whatsapp_msg': ticket['whatsapp_msg']
        })
    
    # Sort by priority (Critical > High > Medium)
    priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    action_tickets.sort(key=lambda x: priority_order.get(x['priority'], 99))
    
    output = {
        'summary': {
            'total_pincodes': total_pincodes,
            'critical_alerts': critical_alerts,
            'mbu_backlog_total': mbu_backlog_total,
            'high_priority_tickets': sum(1 for t in all_tickets if t['priority'] in ['Critical', 'High']),
            'total_tickets': len(all_tickets)
        },
        'compliance_map_data': compliance_map_data,
        'action_tickets': action_tickets
    }
    
    # Add anomaly detection summary
    if 'is_high_risk' in df_results.columns:
        output['anomaly_summary'] = {
            'total_analyzed': len(df_results),
            'high_risk_count': int(df_results['is_high_risk'].sum()),
            'average_confidence': float(df_results[df_results['is_high_risk']]['confidence_score'].mean()) if df_results['is_high_risk'].any() else 0
        }
    
    return output


def save_output(output: Dict[str, Any], output_path: Path) -> None:
    """
    Save output to JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✅ Output saved to: {output_path}")
    print(f"     📊 Total Pincodes: {output['summary']['total_pincodes']:,}")
    print(f"     🚨 Critical Alerts: {output['summary']['critical_alerts']:,}")
    print(f"     📋 MBU Backlog: {output['summary']['mbu_backlog_total']:,}")
    print(f"     🎫 Total Tickets: {output['summary']['total_tickets']:,}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("🚀 UIDAI HACKATHON DASHBOARD - INSIGHT GENERATOR")
    print("="*60)
    print("   High-Performance Backend Script")
    print("   4-Model Ensemble Anomaly Detection")
    print("="*60)
    
    # Part 1: Load and preprocess data
    df_enrol, df_bio, df_demo = load_all_data()
    
    if df_enrol.empty:
        print("\n❌ Error: No enrolment data loaded. Exiting.")
        return
    
    # Part 2: Run anomaly detection
    df_features, X_scaled, scaler = prepare_features(df_enrol, df_bio, df_demo)
    df_results = run_ensemble_voting(df_features, X_scaled)
    
    # Part 3: Generate tickets
    all_tickets = generate_all_tickets(df_enrol, df_bio, df_demo, df_results)
    
    # Part 4: Generate and save output
    output = generate_output(df_enrol, df_bio, df_results, all_tickets)
    
    output_path = Path(__file__).parent / "app_data.json"
    save_output(output, output_path)
    
    print("\n" + "="*60)
    print("✅ INSIGHT GENERATION COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
