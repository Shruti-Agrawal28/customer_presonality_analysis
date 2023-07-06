from analysis.pipeline.batch_prediction import start_batch_prediction

styles = """
    .container {
        width: 500px;
        margin: auto;
    }
    .input-label {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .input-field {
        font-size: 16px;
        padding: 5px;
        margin-bottom: 10px;
        width: 100%;
    }
    .submit-button {
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
    }
    .result {
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
        white-space: pre-line;
    }
"""


file_path = "marketing_campaign.csv"
print(__name__)
if __name__ == "__main__":
    try:
        # start_training_pipeline()
        output_file = start_batch_prediction(input_file_path=file_path)
        print(output_file)
    except Exception as e:
        print(e)
