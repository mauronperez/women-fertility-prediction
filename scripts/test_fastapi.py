import requests

url = "http://127.0.0.1:8000/predict"

sample_data = {
    'age': "Less than 30",
    'married': "Yes",
    'housing_payment': "Less than 20%",  # should be a percentage range
    'housing_income': "Low",  # should be "Low", "Middle", or "High"
    'job': "Yes",  # should be binary, she works or not
    'level_studies': "Post-Graduate",  # education level 
    'income': "High",
    'number_living': "Less than 4",
    'house_income': "High"  # should be "Low", "Middle", or "High"
}

response = requests.post(url, json=sample_data)
print("Status Code:", response.status_code)
print("Response:", response.json())
