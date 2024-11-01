import json
from datetime import datetime
from utils import generate_hash


class LicensePlateSaver:

    def __init__(self, filename='../output/detected_plates.json', latitude=0, longitude=0):
        self.filename = filename
        self.latitude = latitude
        self.longitude = longitude

    
    def save(self, license_plate):
        timestamp = datetime.now().isoformat()
        hash = generate_hash(license_plate, timestamp, self.latitude, self.longitude) 

        plate_data = {
            'hash': hash, 
            'license_plate': license_plate,
            'timestamp': timestamp,
            'latitude': self.latitude,
            'longitude': self.longitude 
        }

        try:
            with open(self.filename, 'r') as json_file:
                data = json.load(json_file)
        
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(plate_data)

        with open(self.filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
