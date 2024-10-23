import requests
import json
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

test_data = [
    {
        "text": "Donald Trump won the election",
        "prediction": "negative"
    },
    {
        "text": "World War III is coming",
        "prediction": "negative"
    },
    {
        "text": "Obama was a US president",
        "prediction": "postive"
    },
    {
        "text": "The Olympic Games are held every 4 years",
        "prediction": "positive"
    }
]


def send_request(input_text,ground_truth):
    url = "http://servesentimentenv-env.eba-keghuafu.us-east-1.elasticbeanstalk.com/predict"
    data = {"text": input_text}
    response = requests.post(url, json=data)
    print(response.json())
    assert "prediction" in response.json(), "Prediction is missing from the response"
    print(f"Prediction: {response.json()['prediction']}, Ground Truth: {ground_truth}")


def run_functional_tests():
    for i in range(4):
        send_request(test_data[i]["text"],test_data[i]["prediction"])



def  run_performance_tests():
    with open('performance_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['test_case', 'start_time', 'end_time', 'latency', 'input_text', 'prediction', 'ground_truth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(100):
            for j in range(4):
                start_time = datetime.now()
                url = "http://servesentimentenv-env.eba-keghuafu.us-east-1.elasticbeanstalk.com/predict"
                data = {"text": test_data[j]["text"]}
                response = requests.post(url, json=data)
                end_time = datetime.now()
                latency = (end_time - start_time).total_seconds()
                prediction = response.json().get('prediction', 'N/A')
                
                # Write the row to the CSV file
                writer.writerow({
                    'test_case': i,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'latency': latency,
                    'input_text': test_data[j]["text"],
                    'prediction': prediction,
                    'ground_truth': test_data[j]["prediction"]
                })

                print(f"Request {i*4 + j + 1}: Latency = {latency} seconds, Prediction: {prediction}, Ground Truth: {test_data[j]['prediction']}")

def generate_box_plot():
    df = pd.read_csv('performance_results.csv')
    df['latency'] = pd.to_numeric(df['latency'])
    latency_2d = df.pivot(index='test_case', columns='input_text', values='latency').values
    plt.figure(figsize=(10, 6))
    plt.boxplot(latency_2d, vert=False, patch_artist=True)
    plt.title('Latency Distribution')
    plt.xlabel('Latency (seconds)')
    plt.show()


def calculate_average_latencies():
    df = pd.read_csv('performance_results.csv')
    df['latency'] = pd.to_numeric(df['latency'])
    
    overall_avg_latency = df['latency'].mean()
    print(f"Overall average latency: {overall_avg_latency:.6f} seconds")
    
    avg_latency_per_case = df.groupby('input_text')['latency'].mean()
    print("\nAverage latency per test case:")
    for case, avg_latency in avg_latency_per_case.items():
        print(f"{case}: {avg_latency:.6f} seconds")

    return overall_avg_latency, avg_latency_per_case

# def generate_box_plot_per_test_case():
            

if __name__ == "__main__":
    generate_box_plot()
    calculate_average_latencies()