import argparse
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
import glob
import matplotlib.pyplot as plt





def load_predefined_split(split_file_path):
    
    split_dict = {}
    with open(split_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                file_id = parts[0]
                split = parts[1]
                split_dict[file_id] = split
    
    print(f"Loaded predefined split: {len(split_dict)} files")
    train_count = sum(1 for split in split_dict.values() if split == 'train')
    test_count = sum(1 for split in split_dict.values() if split == 'test')
    print(f"Train files: {train_count}, Test files: {test_count}")
    
    return split_dict


def parse_icbhi_annotation_file(file_path):
    
    segments = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    segment = {
                        'start_time': float(parts[0]),
                        'end_time': float(parts[1]),
                        'crackle': int(parts[2]),
                        'wheeze': int(parts[3])
                    }
                    
                    # Determine the class based on crackle and wheeze flags
                    if segment['crackle'] == 0 and segment['wheeze'] == 0:
                        segment['class'] = 'normal'
                    elif segment['crackle'] == 1 and segment['wheeze'] == 0:
                        segment['class'] = 'crackle'
                    elif segment['crackle'] == 0 and segment['wheeze'] == 1:
                        segment['class'] = 'wheeze'
                    else:  # Both are 1
                        segment['class'] = 'both'
                    
                    segments.append(segment)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line in {file_path}: {line.strip()}")
                    print(f"Error details: {e}")
    
    return segments


def extract_file_id_from_path(file_path):
    
    # Extracting base filename without extension
    base_name = os.path.basename(file_path).replace('.wav', '')
    return base_name


def create_icbhi_dataset_with_predefined_split(data_dir, split_file_path, output_csv=None):
    
    # Loading predefined split
    split_dict = load_predefined_split(split_file_path)
    
    data = []
    
    # Find all audio files (WAV)
    audio_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    print(f"Found {len(audio_files)} audio files")
    
    for audio_file in audio_files:
        # Getting the corresponding text file
        txt_file = audio_file.replace(".wav", ".txt")
        
        if not os.path.exists(txt_file):
            print(f"Warning: No annotation file for {audio_file}")
            continue
        
        # Extracting file ID to check in split dictionary
        file_id = extract_file_id_from_path(audio_file)
        
        # Skipping if file is not in predefined split
        if file_id not in split_dict:
            print(f"Warning: File ID {file_id} not found in predefined split")
            continue
        
        # Getting the split (train or test)
        split = split_dict[file_id]
        
        try:
            # Parsing annotation file
            segments = parse_icbhi_annotation_file(txt_file)
            
            if not segments:
                print(f"Warning: No valid segments in {txt_file}")
                continue
            
            # Getting audio information
            audio_info = os.path.basename(audio_file).split('_')
            patient_id = audio_info[0] if len(audio_info) > 0 else "unknown"
            
            # Extracting location if available (e.g., Tc, Ar, Pr, etc.)
            location = None
            for i, part in enumerate(audio_info):
                if part in ["Tc", "Ar", "Pr", "Pl", "Ll", "Lr", "Al"]:
                    location = part
                    break
            
            for i, segment in enumerate(segments):
                entry = {
                    'file_path': audio_file,
                    'file_id': file_id,
                    'patient_id': patient_id,
                    'location': location,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['end_time'] - segment['start_time'],
                    'crackle': segment['crackle'],
                    'wheeze': segment['wheeze'],
                    'label': segment['class'],
                    'segment_id': i,
                    'split': split  # Add the predefined split
                }
                data.append(entry)
                
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        raise ValueError("No valid data found. Check dataset structure.")
    
    print("\nDataset Statistics:")
    print(f"Total segments: {len(df)}")
    print("\nClass distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\nPatient distribution:")
    patient_counts = df['patient_id'].value_counts()
    print(f"  Number of patients: {len(patient_counts)}")
    
    print("\nLocation distribution:")
    print(df['location'].value_counts())
    
    print("\nSplit distribution:")
    split_counts = df['split'].value_counts()
    print(f"  Train: {split_counts.get('train', 0)} segments")
    print(f"  Test: {split_counts.get('test', 0)} segments")
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Dataset saved to {output_csv}")
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    # Verifying patients don't overlap between train and test
    train_patients = set(train_df['patient_id'].unique())
    test_patients = set(test_df['patient_id'].unique())
    overlap = train_patients.intersection(test_patients)
    
    if overlap:
        print(f"\nWARNING: {len(overlap)} patients appear in both train and test sets: {overlap}")
        print("This could lead to data leakage. Consider using patient-based split.")
    
    print("\nTrain class distribution:")
    train_class_counts = train_df['label'].value_counts()
    for label, count in train_class_counts.items():
        print(f"  {label}: {count} ({count/len(train_df)*100:.1f}%)")
    
    print("\nTest class distribution:")
    test_class_counts = test_df['label'].value_counts()
    for label, count in test_class_counts.items():
        print(f"  {label}: {count} ({count/len(test_df)*100:.1f}%)")
    
    return train_df, test_df

def prepare_icbhi_dataset_with_predefined_split(
    data_dir, 
    split_file_path,
    output_dir="./icbhi_data",
    visualize_samples=True
):
    """
    Prepare ICBHI dataset for training using predefined split
    
    Args:
        data_dir: Directory containing ICBHI data
        split_file_path: Path to file containing predefined train/test split
        output_dir: Directory to save processed data
        visualize_samples: Whether to visualize sample segments
        
    Returns:
        train_df, test_df: DataFrames for training and testing
    """
    # Creating output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Creating dataset from ICBHI data with predefined split
    train_df, test_df = create_icbhi_dataset_with_predefined_split(
        data_dir, 
        split_file_path,
        output_csv=os.path.join(output_dir, "icbhi_dataset.csv")
    )
    
    # Saving train and test splits
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    
    
    return train_df, test_df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Prepare ICBHI dataset with predefined split")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with ICBHI dataset")
    parser.add_argument("--split_file", type=str, required=True, help="File with predefined train/test split")
    parser.add_argument("--output_dir", type=str, default="./icbhi_data", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Visualize samples")
    
    args = parser.parse_args()
    
    # Prepare dataset with predefined split
    train_df, test_df = prepare_icbhi_dataset_with_predefined_split(
        args.data_dir,
        args.split_file,
        args.output_dir,
        args.visualize
    )
    
    print("Dataset preparation completed!")