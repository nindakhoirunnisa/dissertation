import requests
from geopy.geocoders import Nominatim
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class Geocoder:
    def __init__(self):
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.geolocator = Nominatim(user_agent='my_geocoder')

        # Set up a custom session with retries
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[ 500, 502, 503, 504 ])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
    
    def geocode(self, location_name):
        params = {
            "q": location_name,
            "format": "json"
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if data:
            latitude = float(data[0]["lat"])
            longitude = float(data[0]["lon"])
            return latitude, longitude
        else:
            return None, None
        
    def get_province_name(self, latitude, longitude):
        try: 
            location = self.geolocator.reverse((latitude, longitude), exactly_one=True)
            
            if location:
                address = location.raw.get('address', {})
                province_name = address.get('state', '')
                return province_name
        
        except requests.exceptions.Timeout:
            print("Timeout occurred. Retrying...")
            return None
        return None
