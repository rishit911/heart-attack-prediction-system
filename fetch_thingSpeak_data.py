import requests
import pandas as pd

# ThingSpeak Channel Details
CHANNEL_ID = "2819490"
URL = f"https://api.thingspeak.com/channels/2819490/feeds.json"

def fetch_data():
    """Fetch latest heart rate, SpO₂, and temperature from ThingSpeak."""
    response = requests.get(URL)
    
    if response.status_code == 200:
        data = response.json()
        feeds = data["feeds"]

        # Extract latest values
        latest_entry = feeds[-1]
        heart_rate = latest_entry["field1"]
        spo2 = latest_entry["field2"]
        temperature = latest_entry["field3"]
        timestamp = latest_entry["created_at"]

        print(f"Timestamp: {timestamp}")
        print(f"Heart Rate: {heart_rate} bpm")
        print(f"SpO₂: {spo2} %")
        print(f"Temperature: {temperature} °C")

        return timestamp, heart_rate, spo2, temperature
    else:
        print("Failed to fetch data from ThingSpeak")
        return None

def save_to_csv():
    """Fetch data and save it to a CSV file."""
    data = fetch_data()
    if data:
        df = pd.DataFrame([data], columns=["Timestamp", "Heart Rate", "SpO₂", "Temperature"])
        df.to_csv("health_data.csv", mode="a", header=False, index=False)
        print("Data saved to health_data.csv")

# Run the script
fetch_data()
save_to_csv()
