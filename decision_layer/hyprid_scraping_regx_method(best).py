import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Tuple, Optional, Dict, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instagram_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InstagramReferenceDetector:
    def __init__(self):
        """Initialize detector with configurations"""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        self.timeout = 15
        self.max_redirects = 2
        self.facebook_mobile_agent = "Mozilla/5.0 (Linux; Android 10; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"

    def _validate_url(self, url: str) -> bool:
        """Validate URL format and Facebook domain"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
            if 'facebook.com' in result.netloc and not url.startswith(('http://', 'https://')):
                return False
            return True
        except ValueError:
            return False

    def _fallback_regex_score(self, text: str) -> Tuple[float, Optional[str], List[str]]:
        """Enhanced regex detection with multilingual support"""
        text = text.lower().strip()
        patterns = [
            # URL patterns (highest confidence)
            (r'(https?://)?(www\.)?instagram\.com/([a-z0-9_.]+)/?', 1.0, 'direct_url'),
            (r'(https?://)?(www\.)?instagr\.am/([a-z0-9_.]+)/?', 1.0, 'direct_url'),
            
            # Username patterns
            (r'(?:^|\s)@([a-z0-9_.]{1,30})(?:$|\s)', 0.8, 'username'),
            
            # Platform references
            (r'(?:^|\s)insta(?:gram)?(?:$|\s|/)', 0.5, 'platform_ref'),
            (r'(?:^|\s)ig(?:$|\s)', 0.4, 'platform_ref'),
            
            # Multilingual support
            (r'(انستغرام|إنستا|ইনস্টা|インスタ|인스타)', 0.4, 'i18n_ref'),
            
            # Common phrases
            (r'(?:follow|check|see|visit)\s+(?:my|our)\s+insta', 0.3, 'hint_phrase'),
            (r'link\s+in\s+bio', 0.2, 'hint_phrase')
        ]

        max_score = 0.0
        best_match = None
        matches = []
        
        for pattern, score, match_type in patterns:
            found = re.finditer(pattern, text, re.IGNORECASE)
            for match in found:
                matches.append(match.group())
                if score > max_score:
                    max_score = score
                    best_match = match_type
                if match_type == 'direct_url':  # Highest confidence
                    return 1.0, match_type, matches

        return min(max_score, 1.0), best_match, matches

    def _scrape_website(self, url: str) -> Optional[BeautifulSoup]:
        """Robust website scraper with Facebook special handling"""
        if not self._validate_url(url):
            logger.warning(f"Invalid URL: {url}")
            return None

        session = requests.Session()
        session.max_redirects = self.max_redirects
        
        try:
            # Special handling for Facebook
            if 'facebook.com' in url:
                headers = {"User-Agent": self.facebook_mobile_agent}
                response = session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=False  # Critical for detecting login walls
                )
                
                # Check for Facebook login redirect
                if response.status_code in (301, 302):
                    location = response.headers.get('location', '')
                    if 'facebook.com/login' in location:
                        logger.warning("Facebook login wall detected")
                        return None
            else:
                response = session.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout
                )

            response.raise_for_status()
            
            if 'text/html' not in response.headers.get('Content-Type', ''):
                logger.warning("Non-HTML content received")
                return None
                
            return BeautifulSoup(response.text, 'html.parser')
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Scraping failed: {str(e)}")
            return None

    def _find_social_links(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Find all social media links with enhanced detection"""
        social_links = {
            'instagram': [],
            'facebook': [],
            'twitter': []
        }

        # Standard link tags
        for link in soup.find_all("a", href=True):
            href = link['href'].lower()
            if 'instagram.com' in href or 'instagr.am' in href:
                if re.match(r'(https?://)?(www\.)?(instagram\.com|instagr\.am)/[^/]+/?', href):
                    social_links['instagram'].append(link['href'])
            elif 'facebook.com' in href:
                social_links['facebook'].append(link['href'])
            elif 'twitter.com' in href:
                social_links['twitter'].append(link['href'])

        # Meta tags
        for meta in soup.find_all("meta"):
            if meta.get('property') in ['og:url', 'al:web:url']:
                content = meta.get('content', '').lower()
                if 'instagram.com' in content:
                    social_links['instagram'].append(content)

        # Deduplicate links
        for platform in social_links:
            social_links[platform] = list(set(social_links[platform]))
            
        return social_links

    def detect_references(
        self,
        url: Optional[str] = None,
        text: Optional[str] = None,
        html: Optional[str] = None
    ) -> Dict:
        """
        Comprehensive reference detection with multiple fallbacks
        
        Returns:
            {
                'instagram': {
                    'found': bool,
                    'score': float,
                    'method': str,
                    'matches': list,
                    'error': Optional[str],
                    'requires_login': bool (for Facebook)
                },
                'other_platforms': {
                    'facebook': list,
                    'twitter': list
                }
            }
        """
        result = {
            'instagram': {
                'found': False,
                'score': 0.0,
                'method': 'none',
                'matches': [],
                'error': None,
                'requires_login': False
            },
            'other_platforms': {
                'facebook': [],
                'twitter': []
            }
        }

        # HTML/URL scraping approach
        if url or html:
            try:
                soup = self._scrape_website(url) if url else BeautifulSoup(html, 'html.parser')
                if soup:
                    social_links = self._find_social_links(soup)
                    result['other_platforms'] = {
                        'facebook': social_links.get('facebook', []),
                        'twitter': social_links.get('twitter', [])
                    }
                    
                    if social_links.get('instagram'):
                        result['instagram'].update({
                            'found': True,
                            'score': 1.0,
                            'method': 'scrape',
                            'matches': social_links['instagram']
                        })
                        return result
                elif url and 'facebook.com' in url:
                    result['instagram']['requires_login'] = True
            except Exception as e:
                result['instagram']['error'] = str(e)
                logger.error(f"Detection error: {str(e)}")

        # Text analysis fallback
        if text and not result['instagram']['found']:
            try:
                score, match_type, matches = self._fallback_regex_score(text)
                result['instagram'].update({
                    'found': score >= 0.5,  # Threshold for considering it found
                    'score': score,
                    'method': 'regex',
                    'matches': matches
                })
            except Exception as e:
                result['instagram']['error'] = str(e)

        return result


# ==========================================
# TESTING IMPLEMENTATION
# ==========================================
if __name__ == "__main__":
    detector = InstagramReferenceDetector()

    test_cases = [
        {
            "name": "Facebook Profile with Instagram",
            "url": "https://facebook.com/example",
            "text": None
        },
        {
            "name": "Bio with Instagram URL",
            "url": None,
            "text": "Check my photos at https://instagram.com/me"
        },
        {
            "name": "Bio with @username",
            "url": None,
            "text": "Follow me @myprofile for updates"
        },
        {
            "name": "Facebook Login Wall",
            "url": "https://facebook.com/private-profile",
            "text": None
        },
        {
            "name": "Website with Multiple Links",
            "url": "https://example.com",
            "text": None
        }
    ]

    for case in test_cases:
        print(f"\n=== Testing: {case['name']} ===")
        result = detector.detect_references(
            url=case.get('url'),
            text=case.get('text')
        )
        
        insta = result['instagram']
        print(f"Instagram Found: {insta['found']}")
        print(f"Confidence Score: {insta['score']:.1f}")
        print(f"Detection Method: {insta['method']}")
        
        if insta['matches']:
            print(f"Matches: {', '.join(insta['matches'])}")
        if insta['error']:
            print(f"Error: {insta['error']}")
        if insta['requires_login']:
            print("NOTE: Facebook requires login to access this content")
        
        if result['other_platforms']['facebook']:
            print(f"Facebook Links Found: {len(result['other_platforms']['facebook'])}")
        if result['other_platforms']['twitter']:
            print(f"Twitter Links Found: {len(result['other_platforms']['twitter'])}")
------------------------------------------------
Usage:

# Initialize detector
detector = InstagramReferenceDetector()

# Check a Facebook profile
result = detector.detect_references(url="https://facebook.com/somepage")

# Check bio text
result = detector.detect_references(text="Follow me @insta_user")

# Get detailed results
if result['instagram']['found']:
    print(f"Found Instagram: {result['instagram']['matches']}")
if result['instagram']['requires_login']:
    print("Facebook content requires login")