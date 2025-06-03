"""
Egyptian Arabic Text Preprocessing Module

This module handles text normalization, diacritization, and phonetic conversion
specifically for Egyptian Arabic dialect, preparing text for TTS synthesis.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import pyarabic.araby as araby
import pyarabic.number as number
from arabic_reshaper import reshape
from bidi.algorithm import get_display


class EgyptianArabicProcessor:
    """
    Comprehensive text processor for Egyptian Arabic dialect.
    Handles normalization, diacritization, and phonetic conversion.
    """
    
    def __init__(self):
        self.setup_egyptian_mappings()
        self.setup_phonetic_mappings()
        self.setup_normalization_rules()
    
    def setup_egyptian_mappings(self):
        """Setup Egyptian Arabic specific character and word mappings."""
        # Egyptian Arabic specific phonetic mappings
        self.egyptian_phonetic_map = {
            'ق': 'ء',  # qaf -> glottal stop in Egyptian
            'ث': 'ت',  # tha -> ta in some Egyptian contexts
            'ذ': 'د',  # dhal -> dal in some Egyptian contexts
            'ظ': 'ض',  # zah -> dad in Egyptian
        }
        
        # Common Egyptian Arabic words and their pronunciations
        self.egyptian_lexicon = {
            'إيه': 'ʔeːh',      # what
            'ازاي': 'ʔizzaːj',   # how
            'فين': 'feːn',      # where
            'امتى': 'ʔimta',    # when
            'ليه': 'leːh',      # why
            'كده': 'kida',      # like this
            'اهو': 'ʔaho',      # here it is
            'يلا': 'jallaː',    # let's go
            'معلش': 'maʕleːʃ',  # never mind
            'خلاص': 'xalaːs',   # finished/enough
        }
        
        # Code-switching patterns (English words in Egyptian Arabic)
        self.code_switch_map = {
            'كمبيوتر': 'kombjuːtar',
            'موبايل': 'mobaːjil',
            'ايميل': 'iːmeːl',
            'انترنت': 'ʔintarnet',
            'فيسبوك': 'feːsbuːk',
        }
    
    def setup_phonetic_mappings(self):
        """Setup phonetic mappings for Egyptian Arabic."""
        # Egyptian Arabic phoneme inventory
        self.phoneme_map = {
            # Consonants
            'ب': 'b', 'ت': 't', 'ث': 't', 'ج': 'g',  # Egyptian ج as 'g'
            'ح': 'ħ', 'خ': 'x', 'د': 'd', 'ذ': 'd',
            'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'ʃ',
            'ص': 'sˤ', 'ض': 'dˤ', 'ط': 'tˤ', 'ظ': 'dˤ',
            'ع': 'ʕ', 'غ': 'ɣ', 'ف': 'f', 'ق': 'ʔ',  # Egyptian ق as glottal stop
            'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n',
            'ه': 'h', 'و': 'w', 'ي': 'j',
            
            # Vowels (Egyptian Arabic vowel system)
            'َ': 'a',   # fatha
            'ِ': 'i',   # kasra  
            'ُ': 'u',   # damma
            'ا': 'aː',  # alif (long a)
            'ي': 'iː',  # ya (long i)
            'و': 'uː',  # waw (long u)
            'ة': 'a',   # ta marbuta
            'ى': 'aː',  # alif maksura
        }
        
        # Stress patterns for Egyptian Arabic
        self.stress_patterns = {
            'CVCV': [0, 1],      # stress on second syllable
            'CVCVC': [0, 0, 1],  # stress on final syllable
            'CVCVCV': [0, 1, 0], # stress on second syllable
        }
    
    def setup_normalization_rules(self):
        """Setup text normalization rules for Egyptian Arabic."""
        # Number normalization
        self.number_patterns = {
            r'\d+': self.convert_numbers_to_words,
        }
        
        # Punctuation normalization
        self.punctuation_map = {
            '؟': '?',
            '؛': ';',
            '،': ',',
            '٪': '%',
        }
        
        # Diacritics to remove for normalization
        self.diacritics_to_remove = [
            '\u064B',  # tanween fath
            '\u064C',  # tanween kasra
            '\u064D',  # tanween damma
            '\u064E',  # fatha
            '\u064F',  # damma
            '\u0650',  # kasra
            '\u0651',  # shadda
            '\u0652',  # sukun
            '\u0653',  # maddah
            '\u0654',  # hamza above
            '\u0655',  # hamza below
            '\u0656',  # subscript alef
            '\u0657',  # inverted damma
            '\u0658',  # mark noon ghunna
            '\u0659',  # zwarakay
            '\u065A',  # vowel sign small v above
            '\u065B',  # vowel sign inverted small v above
            '\u065C',  # vowel sign dot below
            '\u065D',  # reversed damma
            '\u065E',  # fatha with two dots
            '\u065F',  # wavy hamza below
            '\u0670',  # superscript alef
        ]
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Egyptian Arabic text for TTS processing.
        
        Args:
            text: Raw Egyptian Arabic text
            
        Returns:
            Normalized text ready for phonetic conversion
        """
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize Arabic characters
        text = self.normalize_arabic_chars(text)
        
        # Handle numbers
        text = self.normalize_numbers(text)
        
        # Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Apply Egyptian-specific normalizations
        text = self.apply_egyptian_normalizations(text)
        
        return text
    
    def normalize_arabic_chars(self, text: str) -> str:
        """Normalize Arabic character variations."""
        # Normalize alef variations
        text = re.sub(r'[آأإ]', 'ا', text)
        
        # Normalize ya variations
        text = re.sub(r'[ىئ]', 'ي', text)
        
        # Normalize ta marbuta
        text = re.sub(r'ة', 'ه', text)  # Convert ta marbuta to ha
        
        # Remove tatweel (kashida)
        text = re.sub(r'ـ', '', text)
        
        return text
    
    def normalize_numbers(self, text: str) -> str:
        """Convert numbers to Arabic words."""
        # Convert Arabic-Indic digits to Arabic words
        def replace_number(match):
            num = match.group()
            try:
                # Convert to integer and then to Arabic words
                arabic_num = number.int2word(int(num), lang='ar')
                return arabic_num
            except:
                return num
        
        # Replace Arabic-Indic digits
        text = re.sub(r'[٠-٩]+', replace_number, text)
        
        # Replace Western digits
        text = re.sub(r'\d+', replace_number, text)
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        for arabic_punct, latin_punct in self.punctuation_map.items():
            text = text.replace(arabic_punct, latin_punct)
        return text
    
    def apply_egyptian_normalizations(self, text: str) -> str:
        """Apply Egyptian Arabic specific normalizations."""
        # Apply Egyptian phonetic changes
        for standard, egyptian in self.egyptian_phonetic_map.items():
            text = text.replace(standard, egyptian)
        
        # Handle common Egyptian words
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Check if word is in Egyptian lexicon
            if word in self.egyptian_lexicon:
                normalized_words.append(word)  # Keep original for now
            elif word in self.code_switch_map:
                normalized_words.append(word)  # Keep original for now
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def convert_numbers_to_words(self, match) -> str:
        """Convert numeric match to Arabic words."""
        num_str = match.group()
        try:
            num = int(num_str)
            return number.int2word(num, lang='ar')
        except:
            return num_str
    
    def add_diacritics(self, text: str) -> str:
        """
        Add diacritics to Egyptian Arabic text for better pronunciation.
        
        Args:
            text: Normalized Egyptian Arabic text
            
        Returns:
            Text with appropriate diacritics
        """
        # This is a simplified diacritization
        # In practice, this would use a trained diacritization model
        
        words = text.split()
        diacritized_words = []
        
        for word in words:
            if word in self.egyptian_lexicon:
                # Use predefined diacritization for known words
                diacritized_words.append(self.get_diacritized_form(word))
            else:
                # Apply rule-based diacritization
                diacritized_words.append(self.apply_diacritization_rules(word))
        
        return ' '.join(diacritized_words)
    
    def get_diacritized_form(self, word: str) -> str:
        """Get diacritized form of known Egyptian words."""
        # Simplified mapping - in practice this would be more comprehensive
        diacritized_map = {
            'إيه': 'إِيه',
            'ازاي': 'إِزَّاي',
            'فين': 'فِين',
            'امتى': 'إِمتَى',
            'ليه': 'لِيه',
            'كده': 'كِده',
            'اهو': 'أَهُو',
            'يلا': 'يَلَّا',
            'معلش': 'مَعلِيش',
            'خلاص': 'خَلاص',
        }
        return diacritized_map.get(word, word)
    
    def apply_diacritization_rules(self, word: str) -> str:
        """Apply rule-based diacritization for unknown words."""
        # Simplified rule-based approach
        # In practice, this would use statistical or neural models
        
        # Add basic vowels based on Egyptian Arabic patterns
        diacritized = word
        
        # Add fatha to consonant clusters
        diacritized = re.sub(r'([بتثجحخدذرزسشصضطظعغفقكلمنهوي])([بتثجحخدذرزسشصضطظعغفقكلمنهوي])', 
                           r'\1َ\2', diacritized)
        
        return diacritized
    
    def to_phonemes(self, text: str) -> List[str]:
        """
        Convert Egyptian Arabic text to phoneme sequence.
        
        Args:
            text: Diacritized Egyptian Arabic text
            
        Returns:
            List of phonemes
        """
        phonemes = []
        
        for char in text:
            if char in self.phoneme_map:
                phonemes.append(self.phoneme_map[char])
            elif char == ' ':
                phonemes.append('_')  # Word boundary marker
            elif char in '.,!?;:':
                phonemes.append('|')  # Pause marker
            # Skip unknown characters
        
        return phonemes
    
    def process_for_tts(self, text: str) -> Dict[str, any]:
        """
        Complete processing pipeline for TTS synthesis.
        
        Args:
            text: Raw Egyptian Arabic text
            
        Returns:
            Dictionary containing processed text, phonemes, and metadata
        """
        # Step 1: Normalize text
        normalized = self.normalize_text(text)
        
        # Step 2: Add diacritics
        diacritized = self.add_diacritics(normalized)
        
        # Step 3: Convert to phonemes
        phonemes = self.to_phonemes(diacritized)
        
        # Step 4: Generate metadata
        metadata = {
            'original_text': text,
            'normalized_text': normalized,
            'diacritized_text': diacritized,
            'phoneme_count': len(phonemes),
            'word_count': len(normalized.split()),
            'has_code_switching': any(word in self.code_switch_map for word in normalized.split()),
            'dialect': 'egyptian_arabic'
        }
        
        return {
            'text': diacritized,
            'phonemes': phonemes,
            'metadata': metadata
        }


def main():
    """Test the Egyptian Arabic processor."""
    processor = EgyptianArabicProcessor()
    
    # Test cases
    test_texts = [
        "مرحبا، إزيك النهارده؟",
        "أنا بشتغل في الكمبيوتر",
        "يلا نروح السينما كده",
        "معلش، خلاص كده",
        "فين المحطة من فضلك؟"
    ]
    
    for text in test_texts:
        print(f"Original: {text}")
        result = processor.process_for_tts(text)
        print(f"Processed: {result['text']}")
        print(f"Phonemes: {' '.join(result['phonemes'])}")
        print(f"Metadata: {result['metadata']}")
        print("-" * 50)


if __name__ == "__main__":
    main()

