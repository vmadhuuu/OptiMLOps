from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

class AlzheimerDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> tuple:
        """
        Preprocess the Alzheimer's dataset.
        
        Args:
            data: Raw dataframe
            is_training: Whether this is training data or inference data
        
        Returns:
            Preprocessed features and labels (if training)
        """
        df = data.copy()
        df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)
        categorical_features = ['Gender', 'Ethnicity', 'EducationLevel']
    
        if is_training:
            for cat_feature in categorical_features:
                self.label_encoders[cat_feature] = LabelEncoder()
                df[cat_feature] = self.label_encoders[cat_feature].fit_transform(df[cat_feature])
        else:
            for cat_feature in categorical_features:
                df[cat_feature] = self.label_encoders[cat_feature].transform(df[cat_feature])
        
        # Binary features (already 0/1)
        binary_features = [
            'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
            'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
            'MemoryComplaints', 'BehavioralProblems', 'Confusion',
            'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
            'Forgetfulness'
        ]
        
        # Numerical features to scale
        numerical_features = [
            'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
            'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
            'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
            'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL'
        ]
        
        # Scale numerical features
        if is_training:
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        else:
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        if is_training:
            X = df.drop('Diagnosis', axis=1)
            y = df['Diagnosis']
            return X, y
        else:
            return df
        
    def prepare_training_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Load and prepare data for training.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        data_path = os.path.join('tests', 'data', 'alzheimers_disease_data.csv')
        df = pd.read_csv(data_path)
        X, y = self.preprocess_data(df, is_training=True)
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> list:
        """Get list of feature names after preprocessing."""
        return (
            ['Gender', 'Ethnicity', 'EducationLevel'] +  # Categorical
            ['Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
             'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
             'MemoryComplaints', 'BehavioralProblems', 'Confusion',
             'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
             'Forgetfulness'] +  # Binary
            ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
             'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
             'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
             'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL']  # Numerical
        )