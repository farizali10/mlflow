import mlflow

def calculate_sum(x,y):
    return x+y

if __name__ == "__main__":
    # Starting the server of mlflow
    with mlflow.start_run():
        x,y=75,10
        z = calculate_sum(x,y)
        # Tracking the experiment with MLFlow
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("Z",z)
