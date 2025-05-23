import os
import yaml
from ultralytics import YOLO
import splitfolders

def validate_dataset_structure(dataset_path):
    """Verify the dataset has correct YOLO format structure"""
    required_folders = ['train', 'val']
    required_subfolders = ['images', 'labels']
    
    for folder in required_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            raise ValueError(f"Missing folder: {folder_path}")
        
        for subfolder in required_subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.exists(subfolder_path):
                raise ValueError(f"Missing subfolder: {subfolder_path}")
    
    print("✅ Dataset structure validated successfully")

def create_yaml_config(dataset_path, output_path):
    """Generate YAML configuration file for YOLO training"""
    # Get class names from the dataset
    classes = sorted([d for d in os.listdir(os.path.join(dataset_path, 'train', 'labels')) 
                     if d.endswith('.txt')])
    
    # Create absolute paths
    abs_dataset_path = os.path.abspath(dataset_path)
    
    config = {
        'path': abs_dataset_path,
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(classes),
        'names': [f'class_{i}' for i in range(len(classes))]  # Update with your actual class names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"✅ YAML config created at: {output_path}")
    return config

def train_yolo_model(config_path, model_name='yolov8n.pt', epochs=50, imgsz=640):
    """Train a YOLOv8 model with the given configuration"""
    # Load model
    model = YOLO(model_name)
    
    # Training parameters
    train_args = {
        'data': config_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'optimizer': 'Adam',
        'lr0': 0.01,
        'patience': 10,
        'save': True,
        'save_period': 5,
        'plots': True,
        'rect': False,
        'augment': True
    }
    
    # Start training
    results = model.train(**train_args)
    
    print("✅ Training completed successfully")
    return results

def main():
    # Configuration
    RAW_DATA_PATH = "uncleaned_file"  # Folder with your raw images and labels
    PROCESSED_PATH = "processed_dataset"  # Where to store split dataset
    YAML_CONFIG_PATH = "dataset_config.yaml"  # Output YAML file
    
    # Step 1: Split dataset (if not already done)
    if not os.path.exists(PROCESSED_PATH):
        print("🔹 Splitting dataset...")
        splitfolders.ratio(
            RAW_DATA_PATH,
            output=PROCESSED_PATH,
            seed=42,
            ratio=(0.8, 0.2),  # Train/Val split
            group_prefix=None,
            move=False
        )
    
    # Step 2: Validate dataset structure
    validate_dataset_structure(PROCESSED_PATH)
    
    # Step 3: Create YAML config
    create_yaml_config(PROCESSED_PATH, YAML_CONFIG_PATH)
    
    # Step 4: Train model
    print("🚀 Starting training...")
    train_yolo_model(YAML_CONFIG_PATH)
    
    print("✨ All done!")

        # Plotting
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label="Train Loss", color='red')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    import torch  # Import here to properly check CUDA availability
    main()