"""
Data Processing Module

Handles all data loading, preprocessing, and feature engineering for telecom KPI data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin, log_function_call
from .config import TelecomConfig


class TelecomDataProcessor(LoggerMixin):
    """
    Comprehensive data processor for telecom KPI data.
    
    Handles data loading, cleaning, feature engineering, and preprocessing
    for anomaly detection models.
    """
    
    def __init__(self, config: TelecomConfig):
        """
        Initialize data processor with configuration.
        
        Args:
            config: TelecomConfig instance with data processing parameters
        """
        self.config = config
        self.scaler = None
        self.feature_columns = []
        self.original_data = None
        self.processed_data = None
        
    @log_function_call
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load telecom KPI data from CSV file.
        
        Args:
            data_path: Path to CSV file. If None, uses config default.
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        if data_path is None:
            data_path = self.config.data_dir / self.config.data.data_file
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.logger.info(f"Loading data from {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            self.logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            
            # Basic validation
            if df.empty:
                raise pd.errors.EmptyDataError("Loaded data is empty")
            
            self.original_data = df.copy()
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    @log_function_call
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate telecom data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Convert Date column to datetime
        if 'Date' in cleaned_df.columns:
            cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'], errors='coerce')
            invalid_dates = cleaned_df['Date'].isna().sum()
            if invalid_dates > 0:
                self.logger.warning(f"Found {invalid_dates} invalid dates, filling with interpolation")
                cleaned_df['Date'] = cleaned_df['Date'].fillna(method='ffill')
        
        # Handle missing values in KPI columns
        kpi_columns = [col for col in self.config.data.kpi_columns if col in cleaned_df.columns]
        
        for col in kpi_columns:
            missing_count = cleaned_df[col].isna().sum()
            if missing_count > 0:
                self.logger.info(f"Column {col}: {missing_count} missing values")
                
                # Use interpolation for numerical columns
                if cleaned_df[col].dtype in ['int64', 'float64']:
                    cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
                    # Fill remaining NaNs with median
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Remove outliers using IQR method for each KPI
        for col in kpi_columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # More conservative than typical 1.5*IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers_count = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    self.logger.info(f"Capping {outliers_count} outliers in {col}")
                    cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
        
        self.logger.info(f"Data cleaning completed. Shape: {cleaned_df.shape}")
        return cleaned_df
    
    @log_function_call
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for improved anomaly detection.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering")
        
        # Make a copy
        feature_df = df.copy()
        
        # Ensure Date column is sorted
        if 'Date' in feature_df.columns:
            feature_df = feature_df.sort_values('Date').reset_index(drop=True)
        
        # Get KPI columns for feature engineering
        kpi_columns = [col for col in self.config.data.kpi_columns if col in feature_df.columns]
        
        # Rolling statistics
        window = self.config.data.rolling_window
        for col in kpi_columns:
            # Rolling mean and std
            feature_df[f'{col}_rolling_mean_{window}d'] = (
                feature_df[col].rolling(window=window, min_periods=1).mean()
            )
            feature_df[f'{col}_rolling_std_{window}d'] = (
                feature_df[col].rolling(window=window, min_periods=1).std()
            )
            
            # Rolling min/max
            feature_df[f'{col}_rolling_min_{window}d'] = (
                feature_df[col].rolling(window=window, min_periods=1).min()
            )
            feature_df[f'{col}_rolling_max_{window}d'] = (
                feature_df[col].rolling(window=window, min_periods=1).max()
            )
        
        # Lag features
        for lag in self.config.data.lag_periods:
            for col in kpi_columns:
                feature_df[f'{col}_lag_{lag}'] = feature_df[col].shift(lag)
        
        # Time-based features
        if 'Date' in feature_df.columns:
            feature_df['hour'] = feature_df['Date'].dt.hour
            feature_df['day_of_week'] = feature_df['Date'].dt.dayofweek
            feature_df['is_weekend'] = (feature_df['day_of_week'] >= 5).astype(int)
            feature_df['is_business_hour'] = ((feature_df['hour'] >= 9) & (feature_df['hour'] <= 17)).astype(int)
        
        # Rate of change features
        for col in kpi_columns:
            feature_df[f'{col}_diff'] = feature_df[col].diff()
            feature_df[f'{col}_pct_change'] = feature_df[col].pct_change()
        
        # Fill NaN values created by lag and diff operations
        feature_df = feature_df.fillna(method='bfill').fillna(0)
        
        self.logger.info(f"Feature engineering completed. New shape: {feature_df.shape}")
        return feature_df
    
    @log_function_call
    def scale_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale numerical features for model training.
        
        Args:
            df: DataFrame with engineered features
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        self.logger.info("Scaling features")
        
        # Select numerical columns for scaling
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        non_scale_cols = ['Date'] if 'Date' in df.columns else []
        
        scale_cols = [col for col in numerical_cols if col not in non_scale_cols]
        
        if fit_scaler:
            # Initialize scaler based on config
            if self.config.data.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.config.data.scaling_method == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.config.data.scaling_method}")
            
            # Fit and transform
            scaled_values = self.scaler.fit_transform(df[scale_cols])
            self.feature_columns = scale_cols
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            scaled_values = self.scaler.transform(df[scale_cols])
        
        # Create scaled DataFrame
        scaled_df = df.copy()
        scaled_df[scale_cols] = scaled_values
        
        self.logger.info(f"Scaled {len(scale_cols)} numerical features")
        return scaled_df
    
    @log_function_call
    def prepare_anomaly_data(self, site_id: Optional[str] = None, kpi_name: Optional[str] = None) -> Dict:
        """
        Prepare data specifically for anomaly detection.
        
        Args:
            site_id: Optional site ID to filter data
            kpi_name: Optional KPI name to focus on
            
        Returns:
            Dictionary with prepared data and metadata
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run full preprocessing pipeline first.")
        
        # Filter data if requested
        data = self.processed_data.copy()
        
        if site_id and 'Site_ID' in data.columns:
            data = data[data['Site_ID'] == site_id]
            self.logger.info(f"Filtered to site {site_id}: {len(data)} rows")
        
        if kpi_name and kpi_name in data.columns:
            # For single KPI analysis, include related features
            kpi_features = [col for col in data.columns if kpi_name in col]
            keep_cols = ['Date', 'Site_ID', 'Sector_ID'] + kpi_features
            keep_cols = [col for col in keep_cols if col in data.columns]
            data = data[keep_cols]
            self.logger.info(f"Focused on KPI {kpi_name}: {len(keep_cols)} features")
        
        # Prepare feature matrix (numerical columns only)
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        X = data[feature_cols].values
        
        return {
            'data': data,
            'features': X,
            'feature_names': feature_cols,
            'metadata': {
                'site_id': site_id,
                'kpi_name': kpi_name,
                'n_samples': len(data),
                'n_features': len(feature_cols)
            }
        }
    
    @log_function_call
    def process_pipeline(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            Fully processed DataFrame ready for model training
        """
        self.logger.info("Starting complete data processing pipeline")
        
        # Load data
        df = self.load_data(data_path)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Scale features
        df = self.scale_features(df, fit_scaler=True)
        
        # Store processed data
        self.processed_data = df
        
        self.logger.info("Data processing pipeline completed successfully")
        return df
    
    @log_function_call
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run data processing pipeline on an existing DataFrame.
        
        Args:
            df: Input DataFrame to process
            
        Returns:
            Fully processed DataFrame ready for model training
        """
        self.logger.info("Starting data processing pipeline on existing DataFrame")
        
        # Store original data
        self.original_data = df.copy()
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Scale features
        df = self.scale_features(df, fit_scaler=True)
        
        # Store processed data
        self.processed_data = df
        
        self.logger.info("DataFrame processing pipeline completed successfully")
        return df
    
    def get_kpi_summary(self) -> Dict:
        """Get summary statistics for all KPIs"""
        if self.original_data is None:
            raise ValueError("No data loaded")
        
        kpi_columns = [col for col in self.config.data.kpi_columns if col in self.original_data.columns]
        summary = {}
        
        for kpi in kpi_columns:
            if kpi in self.original_data.columns:
                summary[kpi] = {
                    'mean': float(self.original_data[kpi].mean()),
                    'std': float(self.original_data[kpi].std()),
                    'min': float(self.original_data[kpi].min()),
                    'max': float(self.original_data[kpi].max()),
                    'missing_count': int(self.original_data[kpi].isna().sum())
                }
        
        return summary
    
    def save_processed_data(self, output_path: Optional[str] = None):
        """Save processed data to file"""
        if self.processed_data is None:
            raise ValueError("No processed data to save")
        
        if output_path is None:
            output_path = self.config.data_dir / "processed_kpi_data.csv"
        
        self.processed_data.to_csv(output_path, index=False)
        self.logger.info(f"Processed data saved to {output_path}")
