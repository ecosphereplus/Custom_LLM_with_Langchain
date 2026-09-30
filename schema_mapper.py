import io
import csv
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from semif import SemIfClassifier

class DataModelMapper:
    """
    Automates CSV data model identification, column alignment (e.g. 'sex' -> 'gender'),
    and data transformation using SemIf direct logit classification and semantic matching rules.
    """

    def __init__(self, classifier: Optional[SemIfClassifier] = None):
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = SemIfClassifier()

    def identify_data_model(
        self,
        incoming_df: pd.DataFrame,
        model_registry: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Identifies which registered target data model best matches the incoming CSV schema.
        """
        incoming_columns = list(incoming_df.columns)
        sample_rows = incoming_df.head(2).to_dict(orient="records")
        state_summary = f"Incoming Schema Header: {incoming_columns}. Sample Records: {sample_rows}"

        options = list(model_registry.keys())
        instructions = (
            "Determine which target data model best matches the structural schema "
            "and entity type of the incoming dataset."
        )

        classification = self.classifier.classify_choice(
            state=state_summary,
            options=options,
            instructions=instructions
        )

        selected_model = classification["selected_option"]
        return {
            "selected_model": selected_model,
            "target_columns": model_registry[selected_model],
            "confidence": classification["confidence"],
            "probabilities": classification["probabilities"]
        }

    def map_columns(
        self,
        incoming_columns: List[str],
        target_columns: List[str],
        sample_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, str]:
        """
        Maps each incoming column to the best matching target column in the target model.
        Uses a combination of exact/synonym matching rules and SemIf probability logit scoring.
        """
        column_mapping = {}

        # Common synonym dictionary for schema field alignment
        synonyms = {
            "id": ["user_id", "id", "account_id"],
            "name": ["full_name", "name", "customer_name"],
            "email_address": ["email", "email_address"],
            "sex": ["gender", "sex"],
            "gender": ["gender", "sex"],
            "price": ["price_usd", "price"],
            "item": ["product_name", "product"]
        }

        for inc_col in incoming_columns:
            inc_clean = inc_col.strip().lower()
            matched = None

            # 1. Exact match or synonym dictionary lookup
            if inc_clean in target_columns:
                matched = inc_clean
            elif inc_clean in synonyms:
                for cand in synonyms[inc_clean]:
                    if cand in target_columns:
                        matched = cand
                        break

            # 2. SemIf classification fallback if rule-based lookup fails
            if matched is None:
                sample_vals = []
                if sample_df is not None and inc_col in sample_df.columns:
                    sample_vals = sample_df[inc_col].dropna().head(3).tolist()

                state = f"Source Column: '{inc_col}'. Sample Values: {sample_vals}"
                instructions = f"Which target field from {target_columns} corresponds to source column '{inc_col}'?"

                result = self.classifier.classify_choice(
                    state=state,
                    options=list(target_columns),
                    instructions=instructions
                )
                matched = result["selected_option"]

            if matched is not None:
                column_mapping[inc_col] = matched

        return column_mapping

    def transform_csv(
        self,
        incoming_csv_data: str,
        model_registry: Dict[str, List[str]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Reads CSV text/file, identifies data model, maps columns, and transforms data frame.
        """
        incoming_df = pd.read_csv(io.StringIO(incoming_csv_data))

        # 1. Identify Model
        model_info = self.identify_data_model(incoming_df, model_registry)
        target_columns = model_info["target_columns"]

        # 2. Map Columns
        column_mapping = self.map_columns(
            incoming_columns=list(incoming_df.columns),
            target_columns=target_columns,
            sample_df=incoming_df
        )

        # 3. Transform DataFrame
        transformed_df = incoming_df.rename(columns=column_mapping)

        # Reorder/filter columns to align with target schema
        existing_mapped_target_cols = [c for c in target_columns if c in transformed_df.columns]
        transformed_df = transformed_df[existing_mapped_target_cols]

        metadata = {
            "model_info": model_info,
            "column_mapping": column_mapping,
            "unmapped_columns": [c for c in incoming_df.columns if c not in column_mapping]
        }

        return transformed_df, metadata

if __name__ == "__main__":
    REGISTRY = {
        "USER_PROFILE_MODEL": ["user_id", "full_name", "email", "gender", "age"],
        "PRODUCT_CATALOG_MODEL": ["sku", "product_name", "price_usd", "category"]
    }

    INCOMING_CSV = """id,name,email_address,sex,age
101,Alice Smith,alice@example.com,Female,29
102,Bob Jones,bob@example.com,Male,34
"""

    mapper = DataModelMapper()
    transformed_df, meta = mapper.transform_csv(INCOMING_CSV, REGISTRY)

    print("=== Identified Data Model ===")
    print(f"Model: {meta['model_info']['selected_model']} (Confidence: {meta['model_info']['confidence']:.4f})")
    print(f"Column Mapping: {meta['column_mapping']}")

    print("\n=== Transformed DataFrame ===")
    print(transformed_df)
