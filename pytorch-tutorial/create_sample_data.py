import os
import random
import string

def generate_sample_text(num_samples=1000, min_length=50, max_length=200):
    """Generate random text samples for training."""
    samples = []
    for _ in range(num_samples):
        # Generate random text with words and punctuation
        length = random.randint(min_length, max_length)
        words = []
        for _ in range(length):
            # Add words of varying lengths
            word_length = random.randint(1, 10)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            
            # Add some punctuation randomly
            if random.random() < 0.1:  # 10% chance of punctuation
                word += random.choice(['.', ',', '!', '?', ';', ':'])
            
            # Add space after word
            word += ' '
            words.append(word)
            
            # Add newlines sometimes
            if random.random() < 0.05:  # 5% chance of newline
                words.append('\n')
        
        samples.append(''.join(words).strip() + '\n')
    
    return samples

def save_samples_to_file(samples, filename):
    """Save samples to a text file."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Write samples to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(samples)
    
    print(f"Generated {len(samples)} samples in {filename}")
    print(f"Total characters: {sum(len(s) for s in samples):,}")

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample training data...")
    samples = generate_sample_text(num_samples=1000)
    
    # Save to file
    data_file = "data/sample_data.txt"
    save_samples_to_file(samples, data_file)
    
    print(f"\nYou can now use this data for training by running:")
    print(f"sbatch submit_gpt.sh {os.path.abspath(data_file)}")
