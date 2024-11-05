import click
from TraderLab import TraderLab

def build_experiment(config_path: str) -> None:
    """
    Build and run a trading experiment using the specified configuration.

    Args:
        config_path (str): The file path to the JSON configuration file.

    The function initializes a TraderLab instance, fetches data, preprocesses it,
    trains the model, evaluates its performance, and performs backtesting.
    """
    # Initialize the TraderLab with the provided configuration path
    exp = TraderLab(config_path)
    
    # Fetch data required for the experiment
    exp.fetch_data()
    
    # Preprocess the data to make it suitable for training
    exp.preprocess_data()
    
    # Train the trading model
    exp.train()
    
    # Evaluate the trained model's performance
    exp.evaluate()
    
    # Conduct backtesting using the trained model
    exp.backtest()


@click.command()
@click.argument("model_name", type=str, required=True)
def main(model_name: str) -> None:
    """
    Main entry point for the CLI that runs the trading experiment.

    Args:
        model_name (str): The name of the model configuration file (without extension).

    This function constructs the configuration path from the provided model name
    and calls the build_experiment function.
    """
    base_path = "experiment_configs/"
    # Construct the full path to the model's configuration file
    config_path = f"{base_path}{model_name}.json"
    
    # Build and run the trading experiment
    build_experiment(config_path)

if __name__ == "__main__":
    main()
