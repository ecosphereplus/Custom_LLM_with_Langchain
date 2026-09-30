import unittest
import pandas as pd
from schema_mapper import DataModelMapper
from semif import SemIfClassifier

class TestDataModelMapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = SemIfClassifier(model_name_or_path="openai-community/gpt2")
        cls.mapper = DataModelMapper(classifier=cls.classifier)
        cls.registry = {
            "USER_PROFILE_MODEL": ["user_id", "full_name", "email", "gender", "age"],
            "PRODUCT_CATALOG_MODEL": ["sku", "product_name", "price_usd", "category"]
        }

    def test_identify_data_model(self):
        csv_data = """id,name,email_address,sex,age
101,Alice Smith,alice@example.com,Female,29
102,Bob Jones,bob@example.com,Male,34
"""
        df = pd.read_csv(pd.io.common.StringIO(csv_data))
        model_info = self.mapper.identify_data_model(df, self.registry)

        self.assertIn("selected_model", model_info)
        self.assertEqual(model_info["selected_model"], "USER_PROFILE_MODEL")
        self.assertEqual(model_info["target_columns"], ["user_id", "full_name", "email", "gender", "age"])

    def test_map_columns(self):
        incoming_cols = ["id", "name", "email_address", "sex", "age"]
        target_cols = ["user_id", "full_name", "email", "gender", "age"]

        mapping = self.mapper.map_columns(incoming_cols, target_cols)
        self.assertEqual(mapping.get("sex"), "gender")
        self.assertEqual(mapping.get("email_address"), "email")
        self.assertEqual(mapping.get("id"), "user_id")

    def test_transform_csv(self):
        incoming_csv = """id,name,email_address,sex,age
201,Jane Doe,jane@example.com,Female,28
"""
        transformed_df, meta = self.mapper.transform_csv(incoming_csv, self.registry)

        self.assertEqual(meta["model_info"]["selected_model"], "USER_PROFILE_MODEL")
        self.assertIn("gender", transformed_df.columns)
        self.assertIn("email", transformed_df.columns)
        self.assertEqual(transformed_df["gender"].iloc[0], "Female")

if __name__ == "__main__":
    unittest.main()
