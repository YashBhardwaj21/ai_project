import cv2
import re
from paddleocr import PaddleOCR
import numpy as np
from datetime import datetime
import os

def initialize_ocr():
    """Initialize PaddleOCR with correct parameters"""
    try:
        # Use the correct parameter name and initialization
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return ocr
    except Exception as e:
        print(f"Error initializing OCR: {e}")
        return None

def preprocess_image(image_path):
    """Preprocess image to improve OCR accuracy"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not load image")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return cv2.imread(image_path, 0)  # Fallback to simple grayscale

def extract_text_with_paddleocr(image_path):
    """Extract text using PaddleOCR with proper error handling"""
    try:
        # Initialize OCR
        ocr = initialize_ocr()
        if ocr is None:
            return None, "Failed to initialize OCR"
        

        result = ocr.predict(image_path)
        return result, None
    except Exception as e:
        return None, str(e)

def parse_ocr_result(result):
    """Parse OCR result and extract text with confidence scores"""
    all_texts = []
    
    if not result:
        print("No result returned from OCR")
        return all_texts
    
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
    
    # Handle different possible result structures
    try:
        # Check if it's the new dictionary format
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            # New format: result[0] is a dictionary with keys like 'rec_texts', 'rec_scores', 'rec_polys'
            data_dict = result[0]
            
            if 'rec_texts' in data_dict and 'rec_scores' in data_dict:
                texts = data_dict['rec_texts']
                scores = data_dict['rec_scores']
                polys = data_dict.get('rec_polys', data_dict.get('dt_polys', []))
                
                print(f"Processing {len(texts)} detected items")
                
                for i, (text, score) in enumerate(zip(texts, scores)):
                    confidence = float(score) * 100  # Convert to percentage
                    bbox = polys[i] if i < len(polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                    
                    print(f"Item {i}: '{text}' (confidence: {confidence:.1f}%)")
                    
                    all_texts.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            else:
                print("Dictionary format detected but missing expected keys")
                
        elif isinstance(result, list) and len(result) > 0:
            # Try old format fallback
            if isinstance(result[0], list):
                # Old format: result[0] is list of detections
                data = result[0]
            else:
                # New format: result is directly the list of detections
                data = result
                
            print(f"Processing {len(data)} detected items (old format)")
            
            for i, line in enumerate(data):
                if line and len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]
                    
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = float(text_info[1]) * 100  # Convert to percentage
                    else:
                        continue
                    
                    print(f"Item {i}: '{text}' (confidence: {confidence:.1f}%)")
                    
                    all_texts.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    
    except Exception as e:
        print(f"Error parsing OCR result: {e}")
        print(f"Result structure keys: {list(result[0].keys()) if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) else 'N/A'}")
    
    print(f"Successfully parsed {len(all_texts)} text items")
    return all_texts

def extract_and_format_text(image_path):
    """Extract text from image and format it nicely"""
    try:
        print("Preprocessing image...")
        processed_img = preprocess_image(image_path)
        
        if processed_img is None:
            print("Failed to process image, trying original image directly...")
            # Try with original image directly
            result, error = extract_text_with_paddleocr(image_path)
        else:
            # Save processed image temporarily for OCR
            temp_path = "temp_processed.jpg"
            cv2.imwrite(temp_path, processed_img)
            
            print("Extracting text from processed image...")
            result, error = extract_text_with_paddleocr(temp_path)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if error:
            print(f"OCR Error: {error}")
            return None, None
            
        if not result:
            print("No result from OCR")
            return None, None

        # Parse the OCR result
        all_texts = parse_ocr_result(result)
        
        if not all_texts:
            print("No text extracted from result")
            return None, None
            
        print(f"Successfully extracted {len(all_texts)} text items\n")
        
        # Sort text by position (top to bottom, left to right)
        sorted_texts = sorted(all_texts, key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))
        
        # Extract just the text for formatting
        texts = [item['text'] for item in sorted_texts]
        confidences = [item['confidence'] for item in sorted_texts]
        
        formatted_text = format_extracted_text(texts, confidences, sorted_texts)
        
        save_to_file(formatted_text, "extracted_report.txt")
        
        return texts, formatted_text
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def group_text_by_lines(text_items, y_threshold=20):
    """Group text items that are on the same line"""
    lines = []
    current_line = []
    current_y = None
    
    for item in text_items:
        y_pos = item['bbox'][0][1]  # Top y coordinate
        
        if current_y is None:
            current_y = y_pos
            current_line.append(item)
        elif abs(y_pos - current_y) < y_threshold:
            current_line.append(item)
        else:
            # Sort items in the line by x position
            current_line.sort(key=lambda x: x['bbox'][0][0])
            lines.append(current_line)
            current_line = [item]
            current_y = y_pos
    
    if current_line:
        current_line.sort(key=lambda x: x['bbox'][0][0])
        lines.append(current_line)
    
    return lines

def format_extracted_text(texts, scores, text_items):
    """Format the extracted text to make it more readable"""
    
    # Group text by lines
    lines = group_text_by_lines(text_items)
    
    formatted_lines = []
    current_section = ""
    
    print("EXTRACTED TEXT (Formatted)")
    print("=" * 80)
    
    for line in lines:
        # Combine text in the same line
        line_text = " ".join([item['text'] for item in line])
        avg_confidence = sum([item['confidence'] for item in line]) / len(line)
        
        # Skip very low confidence or empty text
        if avg_confidence < 50 or not line_text.strip():
            continue
            
        clean_text = line_text.strip()
        
        if is_header(clean_text):
            print(f"\n{clean_text.upper()}")
            print("-" * len(clean_text))
            current_section = "header"
            formatted_lines.append(f"\n{clean_text.upper()}")
            formatted_lines.append("-" * len(clean_text))
            
        elif is_patient_info(clean_text):
            if current_section != "patient":
                print(f"\nPATIENT INFORMATION:")
                print("-" * 20)
                formatted_lines.append(f"\nPATIENT INFORMATION:")
                formatted_lines.append("-" * 20)
                current_section = "patient"
            print(f"  {clean_text}")
            formatted_lines.append(f"  {clean_text}")
            
        elif is_test_result(clean_text):
            if current_section != "results":
                print(f"\nTEST RESULTS:")
                print("-" * 13)
                formatted_lines.append(f"\nTEST RESULTS:")
                formatted_lines.append("-" * 13)
                current_section = "results"
            
            # Special formatting for test results
            if any(char.isdigit() for char in clean_text) and \
               (':' in clean_text or any(unit in clean_text.lower() for unit in ['g/dl', 'mg/dl', '%', 'fl', 'pg'])):
                # This looks like a test result with value
                parts = re.split(r'[:]', clean_text, maxsplit=1)
                if len(parts) == 2:
                    test_name, test_value = parts
                    print(f"  {test_name.strip():<30} {test_value.strip()}")
                    formatted_lines.append(f"  {test_name.strip():<30} {test_value.strip()}")
                else:
                    print(f"  {clean_text}")
                    formatted_lines.append(f"  {clean_text}")
            else:
                print(f"  {clean_text}")
                formatted_lines.append(f"  {clean_text}")
            
        elif is_clinical_note(clean_text):
            if current_section != "notes":
                print(f"\nCLINICAL NOTES:")
                print("-" * 15)
                formatted_lines.append(f"\nCLINICAL NOTES:")
                formatted_lines.append("-" * 15)
                current_section = "notes"
            print(f"  {clean_text}")
            formatted_lines.append(f"  {clean_text}")
            
        elif is_doctor_info(clean_text):
            if current_section != "doctor":
                print(f"\nMEDICAL STAFF:")
                print("-" * 13)
                formatted_lines.append(f"\nMEDICAL STAFF:")
                formatted_lines.append("-" * 13)
                current_section = "doctor"
            print(f"  {clean_text}")
            formatted_lines.append(f"  {clean_text}")
            
        else:
            # General information
            if current_section != "general":
                if current_section != "":  # Don't print if it's the first section
                    print(f"\nOTHER INFORMATION:")
                    print("-" * 18)
                    formatted_lines.append(f"\nOTHER INFORMATION:")
                    formatted_lines.append("-" * 18)
                current_section = "general"
            print(f"  {clean_text}")
            formatted_lines.append(f"  {clean_text}")
    
    print("=" * 80)
    return '\n'.join(formatted_lines)

def is_header(text):
    """Check if text is a header/title"""
    headers = ['labsmart', 'software', 'letterhead', 'haematology', 'complete blood count', 
               'cbc', 'pathology', 'laboratory', 'diagnostic', 'medical', 'report', 'healthcare']
    text_lower = text.lower()
    return any(header in text_lower for header in headers) or \
           (len(text) > 5 and text.isupper()) or \
           (len(text) > 10 and all(c.isupper() or c.isspace() for c in text))

def is_patient_info(text):
    """Check if text is patient information"""
    patient_keywords = ['patient', 'name', 'mr.', 'mrs.', 'ms.', 'dr.', 'age', 'sex', 'gender', 'dob', 
                       'date of birth', 'registered', 'referred', 'collected', 'received', 'reg. no', 
                       'id', 'identification', 'sample', 'specimen']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in patient_keywords) or \
           (('date' in text_lower or 'time' in text_lower) and 
            any(word in text_lower for word in ['collected', 'received', 'reported']))

def is_test_result(text):
    """Check if text is a test result"""
    test_keywords = ['hemoglobin', 'hgb', 'hb', 'count', 'neutrophils', 'lymphocyte', 'wbc', 'rbc', 
                    'platelet', 'plt', 'mcv', 'mch', 'mchc', 'hct', 'hematocrit', 'esr', 
                    'g/dl', 'mg/dl', '%', 'cumm', 'lakhs', 'million', 'fl', 'pg', 'test', 
                    'value', 'unit', 'reference', 'range', 'result', 'normal', 'abnormal']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in test_keywords) or \
           (any(unit in text_lower for unit in ['g/dl', 'mg/dl', '%', 'fl', 'pg', 'mm/hr'])) or \
           (re.search(r'\d+\.?\d*\s*(g/dl|mg/dl|%|fl|pg|mm/hr)', text_lower))

def is_clinical_note(text):
    """Check if text is clinical notes"""
    note_keywords = ['clinical', 'note', 'comment', 'interpretation', 'impression', 'conclusion', 
                    'complete blood count', 'evaluate', 'anemia', 'infection', 'inflammatory', 
                    'abnormal', 'high', 'low', 'elevated', 'reduced', 'causes', 'dehydration', 
                    'stress', 'recommend', 'suggest', 'advise']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in note_keywords) or \
           (len(text) > 40 and not any(char.isdigit() for char in text))  # Long text without numbers

def is_doctor_info(text):
    """Check if text is doctor/staff information"""
    doctor_keywords = ['dr.', 'doctor', 'md', 'mbbs', 'dmlt', 'pathologist', 'incharge', 
                      'kmc no', 'signature', 'approved', 'verified', 'lab director', 
                      'technologist', 'technician', 'specialist']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in doctor_keywords) or \
           (text_lower.startswith('dr.') or text_lower.startswith('dr '))

def save_to_file(formatted_text, filename):
    """Save the formatted text to a file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("MEDICAL REPORT - OCR EXTRACTION\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(formatted_text)
            f.write(f"\n\n--- End of Report ---")
        print(f"Report saved to: {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    print("Medical Report OCR Extractor")
    print("=" * 50)
    
    # You can change this path to your image path
    img_path = "C:\\Users\\Yash Bhardwaj\\Desktop\\Pathology-Lab-Software-2-638.jpg"
    
    # Check if file exists
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        print("Please check the file path and try again.")
        exit(1)
    
    # First try with preprocessing
    print("Trying with image preprocessing...")
    texts, formatted_text = extract_and_format_text(img_path)
    
    if not texts:
        print("Trying direct OCR without preprocessing...")
        # If preprocessing fails, try direct OCR
        try:
            ocr = PaddleOCR(use_textline_orientation=True, lang='en')
            result = ocr.predict(img_path)
            all_texts = parse_ocr_result(result)
            
            if all_texts:
                print(f"Successfully extracted {len(all_texts)} text items\n")
                sorted_texts = sorted(all_texts, key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))
                texts = [item['text'] for item in sorted_texts]
                confidences = [item['confidence'] for item in sorted_texts]
                formatted_text = format_extracted_text(texts, confidences, sorted_texts)
                save_to_file(formatted_text, "extracted_report.txt")
        except Exception as e:
            print(f"Direct OCR with use_textline_orientation failed: {e}")
            # Try with basic initialization
            try:
                print("Trying with basic OCR initialization...")
                ocr = PaddleOCR(lang='en')
                result = ocr.predict(img_path)
                all_texts = parse_ocr_result(result)
                
                if all_texts:
                    print(f"Successfully extracted {len(all_texts)} text items\n")
                    sorted_texts = sorted(all_texts, key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))
                    texts = [item['text'] for item in sorted_texts]
                    confidences = [item['confidence'] for item in sorted_texts]
                    formatted_text = format_extracted_text(texts, confidences, sorted_texts)
                    save_to_file(formatted_text, "extracted_report.txt")
            except Exception as e2:
                print(f"Basic OCR also failed: {e2}")
    
    if texts:
        print(f"\nExtraction completed successfully!")
        print(f"Total items extracted: {len(texts)}")
        print(f"Formatted report saved to: extracted_report.txt")

        print(f"\nQuick Summary:")
        print(f"  • Patient info, test results, and clinical notes extracted")
        print(f"  • Report organized into clear sections")
        print(f"  • Ready for further processing or analysis")
    else:
        print("Extraction failed. Please check your image path and try again.")
        print("Make sure the image exists and is readable.")