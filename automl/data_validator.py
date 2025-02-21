import great_expectations as ge
from typing import List, Dict
import pandas as pd
from typing import Any

class DataValidator:
    def __init__(self, dataset_name: str):
        """Initialize data validator with dataset name."""
        self.dataset_name = dataset_name
        self.context = ge.get_context()
        
    def create_expectations(self, data: pd.DataFrame) -> None:
        """Create data quality expectations based on dataset."""
        suite = self.context.create_expectation_suite(
            f"{self.dataset_name}_suite"
        )
        
        validator = ge.dataset.PandasDataset(data, 
                                           expectation_suite=suite)
        
        # Add basic expectations
        validator.expect_table_row_count_to_be_between(min_value=100)
        validator.expect_table_columns_to_match_ordered_list(
            list(data.columns)
        )
        
        # Add column-specific expectations
        for column in data.columns:
            validator.expect_column_values_to_not_be_null(column)
            if data[column].dtype in ['int64', 'float64']:
                validator.expect_column_values_to_be_between(
                    column,
                    min_value=data[column].min(),
                    max_value=data[column].max()
                )
        
        # Save suite
        self.context.save_expectation_suite(suite)
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data against saved expectations."""
        validator = ge.dataset.PandasDataset(
            data,
            expectation_suite_name=f"{self.dataset_name}_suite"
        )
        results = validator.validate()
        return results.to_json_dict()