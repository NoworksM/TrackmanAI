import time

from input import TM2020OpenPlanetClient

client = TM2020OpenPlanetClient()

while True:
    data = client.retrieve_data()
    print(data)
    time.sleep(1/20)
