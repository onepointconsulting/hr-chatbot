from dataclasses import dataclass
import requests

from log_init import logger


@dataclass
class GeoLocation:
    country_code: str
    country_name: str
    city: str
    postal: str
    latitude: float
    longitude: float


def extract_ip_address(environ: dict) -> str:
    asgi_scope = environ.get("asgi.scope")
    if asgi_scope:
        client = asgi_scope.get("client")
        return client[0]
    return None


def geolocate(ip_address: str) -> GeoLocation:
    response = requests.get(
        f"https://geolocation-db.com/json/{ip_address}&position=true"
    ).json()
    return GeoLocation(
        country_code=response["country_code"],
        country_name=response["country_name"],
        city=response["city"],
        postal=response["postal"],
        latitude=response["latitude"],
        longitude=response["longitude"],
    )


if __name__ == "__main__":
    response = geolocate("185.71.38.58")
    logger.info(response)
