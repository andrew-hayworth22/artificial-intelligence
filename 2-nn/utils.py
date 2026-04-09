from sklearn import datasets, model_selection

def validate_config(config) -> None:
    """Validate the config file."""
    config_dataset = config['dataset']
    is_blobs = config_dataset['type'] == 'blobs'
    config_nn = config['nn']
    if config_dataset['type'] not in ['blobs', 'moons', 'circles']:
        exit(f"Invalid dataset type: {config_dataset['type']}. Must be one of: blobs, moons, circles")
    if is_blobs and config_dataset['clusters'] is not config_nn['output_size']:
        exit(f"Dataset clusters must match output size of NN: {config_nn['output_size']} != {config_dataset['clusters']}")
    if is_blobs and config_dataset['features'] != config_nn['input_size']:
        exit(f"Dataset features must match input size of NN: {config_nn['input_size']} != {config_dataset['features']}")
    if is_blobs and config['render_graph'] and config_dataset['features'] != 2:
        exit(f"Cannot render graph with more than {config_dataset['features']} features. Must be 2")

def generate_dataset(cfg: dict):
    """Generates dataset based on config."""
    seed = None if cfg['seed'] == 0 else cfg['seed']
    data, groups = None, None
    match cfg['type']:
        case 'blobs':
            print(f"Generating blob dataset with {cfg['datapoints']} datapoints with {cfg['clusters']} clusters and {cfg['features']} features...")
            data, groups = datasets.make_blobs(
                n_samples = cfg['datapoints'],
                centers = cfg['clusters'],
                n_features = cfg['features'],
                random_state = seed
            )
        case 'circles':
            print(f"Generating circle dataset with {cfg['datapoints']} datapoints...")
            data, groups = datasets.make_circles(
                n_samples = cfg['datapoints'],
                factor = cfg['factor'],
                noise = cfg['noise'],
                random_state = seed
            )
        case 'moons':
            print(f"Generating moon dataset with {cfg['datapoints']} datapoints...")
            data, groups = datasets.make_moons(
                n_samples = cfg['datapoints'],
                noise = cfg['noise'],
                random_state = seed
            )

    return model_selection.train_test_split(data, groups, test_size=cfg['test_size'], random_state=seed)
