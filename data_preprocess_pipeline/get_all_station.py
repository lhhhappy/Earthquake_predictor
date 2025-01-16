import requests
from bs4 import BeautifulSoup
import pickle

def fetch_all_stations_with_coordinates(base_url):
    # Step 1: Fetch the main page
    response = requests.get(base_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {base_url}, status code: {response.status_code}")
    
    # Parse the main page to find all .tenv3 file links
    soup = BeautifulSoup(response.text, 'html.parser')
    station_links = [link.get('href') for link in soup.find_all('a') if link.get('href').endswith('.tenv3')]

    # Dictionary to store station names and their coordinates
    station_coordinates = {}

    # Step 2: Iterate through all station links and fetch their latitude and longitude
    for station_link in station_links:
        station_name = station_link.split('.')[0]  # Extract station name from link
        station_url = f"{base_url}{station_link}"

        try:
            # Fetch the .tenv3 file
            station_response = requests.get(station_url)
            if station_response.status_code != 200:
                print(f"Failed to fetch {station_url}, skipping...")
                continue
            
            # Parse the file line by line to find latitude and longitude
            lines = station_response.text.splitlines()
            latitude, longitude = None, None
            for line in lines:
                fields = line.split()
                if len(fields) > 20:  # Ensure the line has enough columns
                    try:
                        latitude = float(fields[20])  # _latitude(deg)
                        longitude = float(fields[21])  # _longitude(deg)
                    except (IndexError, ValueError):
                        continue
            
            # Save to dictionary if both latitude and longitude are found
            if latitude is not None and longitude is not None:
                print(f"Found coordinates for {station_name}: {latitude}, {longitude}")
                station_coordinates[station_name] = (latitude, longitude)

        except Exception as e:
            print(f"Error processing {station_url}: {e}")
            continue

    return station_coordinates

# Main function to run the scraper
if __name__ == "__main__":
    base_url = "http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/"
    station_coords = fetch_all_stations_with_coordinates(base_url)
    pickle.dump(station_coords, open("station_dict_all.pkl", "wb"))