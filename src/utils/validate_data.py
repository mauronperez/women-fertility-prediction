import great_expectations as ge
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    
    print("Starting data validation with Great Expectations...")

    
    context = ge.get_context()
    
    # Add a Pandas Data Source
    data_source = context.data_sources.add_pandas(name="ine_data")
    
    # Add a Data Asset to the Data Source
    data_asset = data_source.add_dataframe_asset(name="ine_asset")
    
    # Add the Batch Definition
    batch_definition = data_asset.add_batch_definition_whole_dataframe("ine_batch")
    
    # Retrieve the Batch
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    # Create Expectation Suite
    suite = ge.ExpectationSuite(name="ine_suite")
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column="EDAD"))
    suite.add_expectation(ge.expectations.ExpectColumnValuesToNotBeNull(column="EC"))
    
    
    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeInSet(
        column="EC", value_set=[1,2,3,4,5]
    ))
    
    
    
    
    # Add the Expectation Suite to the Context
    context.suites.add(suite)
    
    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = batch.validate(suite)
    print(results)
    
    # === PROCESS RESULTS ===
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_context"]
            failed_expectations.append(expectation_type)
    
    # Print validation summary

    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations