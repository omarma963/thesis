# pip install google-search-results
import os
import time
from PIL import Image
from PIL.ExifTags import TAGS
from serpapi import GoogleSearch
import requests

class ImageAuthenticityAnalyzer:
    def __init__(self):
        """Initialize EXIF scoring weights"""
        self.exif_weights = {
            'make_model': 0.6,
            'other_tags': 0.3,
            'no_exif': 0.0
        }

    def compute_exif_score(self, image_path: str) -> float:
        """Analyze EXIF metadata for authenticity clues."""
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif() or {}
                tags = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
                
                if 'Make' in tags or 'Model' in tags:
                    return self.exif_weights['make_model']
                elif tags:
                    return self.exif_weights['other_tags']
                return self.exif_weights['no_exif']
                
        except Exception as e:
            print(f"⚠️ EXIF analysis error: {str(e)}")
            return 0.0

def upload_image_to_temp_server(image_path):
    """Upload image to a temporary server for URL-based search"""
    # Note: In production, you'd want to use a proper file hosting service
    # This is just a placeholder concept
    try:
        with open(image_path, 'rb') as f:
            response = requests.post('https://tmpfiles.org/api/v1/upload', files={'file': f})
            response.raise_for_status()
            return response.json()['data']['url']
    except Exception as e:
        print(f"⚠️ Image upload failed: {e}")
        return None

def run_reverse_image_search(image_path, api_key, debug=False):
    """Perform reverse image search using SerpAPI & integrate EXIF scoring."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    analyzer = ImageAuthenticityAnalyzer()
    exif_score = analyzer.compute_exif_score(image_path)

    try:
        # First upload the image to get a URL (in a real scenario, you'd host it properly)
        image_url = upload_image_to_temp_server(image_path)
        if not image_url:
            raise Exception("Failed to get image URL for search")

        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": api_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if debug:
            print("🔍 API Response:", results)

        confidence = 0.0
        
        # Analyze the results structure
        if 'visual_matches' in results:
            for match in results['visual_matches']:
                if 'percentage' in match:  # Some APIs provide match percentages
                    confidence = max(confidence, float(match['percentage']) / 100)
                else:
                    # If no percentages, use presence as confidence indicator
                    confidence = max(confidence, 0.7)
        
        if 'knowledge_graph' in results or 'best_guess' in results:
            confidence = max(confidence, 0.5)

        return {
            'image_score': min(confidence, 0.9),  # Cap at 0.9 unless exact match found
            'exif_score': exif_score,
            'api_response': results if debug else None,
            'weights_used': analyzer.exif_weights
        }

    except Exception as e:
        print(f"⚠️ Reverse image search failed: {e}")
        return {
            'image_score': 0.3,
            'exif_score': exif_score,
            'error': str(e),
            'weights_used': analyzer.exif_weights
        }

# Example Usage
if __name__ == "__main__":
    # You need to get an API key from https://serpapi.com/
    api_key = "your_serpapi_api_key_here"
    image_path = "test_image.jpg"  # Update with your image path

    result = run_reverse_image_search(
        image_path=image_path,
        api_key=api_key,
        debug=True
    )
    
    print("\n🔍 Image Analysis Result:")
    print(f"Image Score: {result['image_score']}")
    print(f"EXIF Score: {result['exif_score']}")
    print(f"Combined Score: {0.7 * result['image_score'] + 0.3 * result['exif_score']}")