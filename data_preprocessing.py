"""
Vietnamese Text Preprocessing for Aspect-based Sentiment Analysis
"""

import pandas as pd
import numpy as np
import re
import os
import sys
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Text processing libraries
import emoji
from bs4 import BeautifulSoup
import html2text
import time
import random

# Try to import underthesea with fallback
try:
    from underthesea import word_tokenize, pos_tag
    UNDERTHESEA_AVAILABLE = True
    print("✓ Underthesea imported successfully")
except ImportError as e:
    print(f"⚠ Underthesea import failed: {e}")
    print("Using fallback tokenization...")
    UNDERTHESEA_AVAILABLE = False

    def word_tokenize(text):
        """Fallback tokenization"""
        return text.split()

    def pos_tag(text):
        """Fallback POS tagging"""
        return [(word, 'N') for word in text.split()]

# Try to import translator with fallback
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
    print("✓ Deep translator imported successfully")
except ImportError as e:
    print(f"⚠ Deep translator import failed: {e}")
    print("Translation will be skipped...")
    TRANSLATOR_AVAILABLE = False

# Add vietnamese-preprocess to path if exists
if os.path.exists('vietnamese-preprocess'):
    sys.path.append('vietnamese-preprocess')

class VietnameseTextPreprocessor:
    def __init__(self):
        # Initialize translator if available
        if TRANSLATOR_AVAILABLE:
            try:
                self.translator = GoogleTranslator(source='en', target='vi')
                self.translation_enabled = True
                print("✓ Google Translator initialized")
            except Exception as e:
                print(f"⚠ Translator initialization failed: {e}")
                self.translator = None
                self.translation_enabled = False
        else:
            self.translator = None
            self.translation_enabled = False

        self.html_parser = html2text.HTML2Text()
        self.html_parser.ignore_links = True
        self.html_parser.ignore_images = True
        
        # Common English words that should be translated
        self.common_english_words = {
            'good': 'tốt', 'bad': 'xấu', 'nice': 'đẹp', 'beautiful': 'đẹp',
            'love': 'yêu', 'like': 'thích', 'hate': 'ghét', 'amazing': 'tuyệt vời',
            'perfect': 'hoàn hảo', 'excellent': 'xuất sắc', 'terrible': 'tệ',
            'awesome': 'tuyệt vời', 'great': 'tuyệt', 'wonderful': 'tuyệt vời',
            'fantastic': 'tuyệt vời', 'horrible': 'kinh khủng', 'awful': 'tệ hại',
            'super': 'siêu', 'ok': 'ổn', 'okay': 'ổn', 'fine': 'ổn',
            'cheap': 'rẻ', 'expensive': 'đắt', 'fast': 'nhanh', 'slow': 'chậm',
            'quality': 'chất lượng', 'price': 'giá', 'color': 'màu', 'smell': 'mùi',
            'texture': 'kết cấu', 'shipping': 'vận chuyển', 'package': 'gói hàng'
        }
        
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def remove_html(self, text: str) -> str:
        """Remove HTML tags and convert HTML entities"""
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Convert HTML entities
        text = self.html_parser.handle(text)
        return text.strip()
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text"""
        return emoji.replace_emoji(text, replace='')
    
    def translate_english_words(self, text: str) -> str:
        """Translate English words to Vietnamese"""
        words = text.split()
        translated_words = []

        for word in words:
            word_lower = word.lower().strip('.,!?;:"()[]{}')

            # Check if word is in common dictionary first (faster)
            if word_lower in self.common_english_words:
                translated_words.append(self.common_english_words[word_lower])
            # Only translate if translation is enabled and word looks like English
            elif (self.translation_enabled and
                  re.match(r'^[a-zA-Z]{3,}$', word_lower) and
                  len(word_lower) >= 3 and len(word_lower) <= 15):  # Limit word length
                try:
                    # Add small delay to avoid rate limiting
                    time.sleep(0.05)  # Reduced delay
                    translated = self.translator.translate(word_lower)
                    if translated and translated != word_lower and len(translated) > 0:
                        translated_words.append(translated)
                    else:
                        translated_words.append(word)
                except Exception:
                    translated_words.append(word)
            else:
                translated_words.append(word)

        return ' '.join(translated_words)
    
    def remove_meaningless_words(self, text: str) -> str:
        """Remove meaningless words and repetitive patterns"""
        words = text.split()
        meaningful_words = []

        for word in words:
            # Skip very short words (1-2 characters) that are likely noise
            if len(word) <= 2 and not re.match(r'^[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+$', word):
                continue

            # Remove words that are clearly repetitive patterns
            if self.is_repetitive_pattern(word):
                continue

            # Remove words with too many repeated characters
            if self.has_excessive_repetition(word):
                continue

            # Remove random character sequences (no vowels or all consonants)
            if self.is_random_sequence(word):
                continue

            # Keep words with Vietnamese characters
            if re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', word.lower()):
                meaningful_words.append(word)
                continue

            # For words without Vietnamese characters, be more strict
            if len(word) >= 4:
                # Check if word is meaningful using POS tagging (if available)
                if UNDERTHESEA_AVAILABLE:
                    try:
                        pos_tags = pos_tag(word)
                        if pos_tags and len(pos_tags) > 0:
                            # If POS tagger can identify the word, it's likely meaningful
                            meaningful_words.append(word)
                        else:
                            # Skip if it looks like random text
                            continue
                    except:
                        # If POS tagging fails, use heuristics
                        if self.looks_like_meaningful_word(word):
                            meaningful_words.append(word)
                else:
                    # Without underthesea, use heuristics
                    if self.looks_like_meaningful_word(word):
                        meaningful_words.append(word)
            else:
                # Keep short words that look meaningful
                if self.looks_like_meaningful_word(word):
                    meaningful_words.append(word)

        return ' '.join(meaningful_words)
    
    def is_repetitive_pattern(self, word: str) -> bool:
        """Check if word is a repetitive pattern"""
        if len(word) < 3:
            return False

        # Check for patterns like "haha", "hihi", "keke", "hihihihi"
        if len(set(word)) <= 2 and len(word) >= 4:
            return True

        # Check for repeated substrings
        for i in range(1, len(word) // 2 + 1):
            substring = word[:i]
            if word == substring * (len(word) // i) and len(word) % i == 0:
                return True

        # Check for alternating patterns like "ababab"
        if len(word) >= 6:
            pattern1 = word[:2]
            if word == pattern1 * (len(word) // 2):
                return True

        return False

    def has_excessive_repetition(self, word: str) -> bool:
        """Check if word has too many repeated characters"""
        if len(word) < 4:
            return False

        # Count consecutive repeated characters
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(word)):
            if word[i] == word[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        # If more than 3 consecutive same characters, likely noise
        return max_consecutive > 3

    def is_random_sequence(self, word: str) -> bool:
        """Check if word looks like random character sequence"""
        if len(word) < 5:
            return False

        word_lower = word.lower()

        # Check vowel/consonant ratio
        vowels = 'aeiouàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ'
        vowel_count = sum(1 for c in word_lower if c in vowels)
        consonant_count = sum(1 for c in word_lower if c.isalpha() and c not in vowels)

        # If no vowels or too few vowels, likely random
        if vowel_count == 0 or (consonant_count > 0 and vowel_count / len(word) < 0.1):
            return True

        # Check for too many consonants in a row
        consonant_streak = 0
        max_consonant_streak = 0

        for c in word_lower:
            if c.isalpha() and c not in vowels:
                consonant_streak += 1
                max_consonant_streak = max(max_consonant_streak, consonant_streak)
            else:
                consonant_streak = 0

        # If more than 4 consonants in a row, likely random
        return max_consonant_streak > 4

    def looks_like_meaningful_word(self, word: str) -> bool:
        """Check if word looks meaningful using heuristics"""
        if len(word) < 2:
            return True

        word_lower = word.lower()

        # Check if it's a common pattern
        if word_lower in ['ok', 'ko', 'dc', 'vs', 'cx', 'nx', 'qua', 'roi', 'nha', 'nhe']:
            return True

        # Check vowel presence for longer words
        if len(word) >= 4:
            vowels = 'aeiouàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ'
            has_vowel = any(c in vowels for c in word_lower)
            if not has_vowel:
                return False

        return True
    
    def normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Vietnamese characters
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        
        # Remove extra whitespaces again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str) -> str:
        """Tokenize Vietnamese text using underthesea"""
        if not UNDERTHESEA_AVAILABLE:
            # Simple fallback tokenization
            return text

        try:
            tokens = word_tokenize(text)
            return ' '.join(tokens)
        except Exception as e:
            print(f"Tokenization error: {e}")
            return text
    
    def preprocess_text(self, text: str, translate_english: bool = False) -> str:
        """Complete preprocessing pipeline"""
        if pd.isna(text) or text == '':
            return ''

        # Convert to string
        text = str(text)

        # Step 1: Remove URLs
        text = self.remove_urls(text)

        # Step 2: Remove HTML
        text = self.remove_html(text)

        # Step 3: Remove emojis
        text = self.remove_emojis(text)

        # Step 4: Translate English words (disabled by default for speed)
        if translate_english:
            text = self.translate_english_words(text)

        # Step 5: Remove meaningless words
        text = self.remove_meaningless_words(text)

        # Step 6: Normalize text
        text = self.normalize_text(text)

        # Step 7: Tokenize
        text = self.tokenize_text(text)

        return text

def load_raw_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data files"""
    print("Loading raw data...")
    
    train_df = pd.read_csv(os.path.join(data_path, 'data_train.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'data_val.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'data_test.csv'))
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, val_df, test_df

def convert_labels_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text labels to numeric format"""
    df_converted = df.copy()

    # Define label mapping according to original format
    label_mapping = {
        'positive': 2,    # Positive sentiment
        'neutral': 1,     # Neutral sentiment
        'negative': 0,    # Negative sentiment
        '': -1,           # Empty string means aspect not mentioned
        'nan': -1,        # String 'nan' means aspect not mentioned
        np.nan: -1,       # NaN means aspect not mentioned
        None: -1          # None means aspect not mentioned
    }

    # List of aspect columns
    aspect_columns = ['stayingpower', 'texture', 'smell', 'price', 'others', 'colour', 'shipping', 'packing']

    for col in aspect_columns:
        if col in df_converted.columns:
            # Fill NaN with empty string first
            df_converted[col] = df_converted[col].fillna('')

            # Convert to string and strip whitespace
            df_converted[col] = df_converted[col].astype(str).str.strip().str.lower()

            # Map text labels to numbers
            df_converted[col] = df_converted[col].map(label_mapping).fillna(-1).astype(int)

            # Debug: print label distribution
            print(f"  {col}: {dict(zip(*np.unique(df_converted[col], return_counts=True)))}")

    return df_converted

def preprocess_dataset(df: pd.DataFrame, preprocessor: VietnameseTextPreprocessor,
                      text_column: str = 'data', enable_translation: bool = False) -> pd.DataFrame:
    """Preprocess a dataset"""
    df_processed = df.copy()

    print(f"Preprocessing {len(df)} samples...")
    if enable_translation:
        print("Translation enabled - this will take longer...")
    else:
        print("Translation disabled for faster processing...")

    # Convert labels to numeric format first
    print("Converting labels to numeric format...")
    df_processed = convert_labels_to_numeric(df_processed)

    # Apply preprocessing to text column
    processed_texts = []
    for i, text in enumerate(df[text_column]):
        if i % 500 == 0:  # Reduced frequency of progress updates
            print(f"Processing sample {i+1}/{len(df)}")

        processed_text = preprocessor.preprocess_text(text, translate_english=enable_translation)
        processed_texts.append(processed_text)

    df_processed['processed_data'] = processed_texts

    return df_processed

def main():
    """Main preprocessing function"""
    # Initialize preprocessor
    preprocessor = VietnameseTextPreprocessor()
    
    # Data paths
    raw_data_path = 'Aspect-based-Sentiment-Analysis-for-Vietnamese-Reviews-about-Beauty-Product-on-E-commerce-Websites/data'
    processed_data_path = 'data_pr'
    
    # Create output directory
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Load raw data
    train_df, val_df, test_df = load_raw_data(raw_data_path)
    
    # Preprocess datasets (translation disabled for speed)
    print("\n" + "="*50)
    print("PREPROCESSING TRAINING DATA")
    print("="*50)
    train_processed = preprocess_dataset(train_df, preprocessor, enable_translation=False)

    print("\n" + "="*50)
    print("PREPROCESSING VALIDATION DATA")
    print("="*50)
    val_processed = preprocess_dataset(val_df, preprocessor, enable_translation=False)

    print("\n" + "="*50)
    print("PREPROCESSING TEST DATA")
    print("="*50)
    test_processed = preprocess_dataset(test_df, preprocessor, enable_translation=False)
    
    # Save processed data
    print("\nSaving processed data...")
    train_processed.to_csv(os.path.join(processed_data_path, 'processed_train.csv'), index=False)
    val_processed.to_csv(os.path.join(processed_data_path, 'processed_val.csv'), index=False)
    test_processed.to_csv(os.path.join(processed_data_path, 'processed_test.csv'), index=False)
    
    print("Preprocessing completed!")
    print(f"Processed data saved to: {processed_data_path}")
    
    # Show sample results
    print("\n" + "="*50)
    print("SAMPLE RESULTS")
    print("="*50)
    for i in range(min(3, len(train_processed))):
        print(f"\nSample {i+1}:")
        print(f"Original: {train_df.iloc[i]['data'][:100]}...")
        print(f"Processed: {train_processed.iloc[i]['processed_data'][:100]}...")

if __name__ == "__main__":
    main()
