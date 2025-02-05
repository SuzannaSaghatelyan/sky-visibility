import numpy as np
import healpy as hp
import json
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy import units as u
from datetime import datetime, timezone

def get_visible_healpix_cells(lat, lon, time_utc, nside=64, output_format='json'):
    """
    Compute the visible HEALPix cells for a given observer location and time.

    """
    observer_location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=0 * u.m)
    observation_time = Time(time_utc)
    altaz_frame = AltAz(obstime=observation_time, location=observer_location)

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = 90 - np.rad2deg(theta)

    sky_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    altaz_coords = sky_coords.transform_to(altaz_frame)

    visible_pixels = np.where(altaz_coords.alt > 0 * u.deg)[0]

    if output_format == 'json':
        return json.dumps({"visible_cells": visible_pixels.tolist()})
    elif output_format == 'fits':
        hp.write_map("visible_sky.fits", visible_pixels, overwrite=True)
        return "visible_sky.fits saved"
    else:
        raise ValueError("Unsupported output format. Choose 'json' or 'fits'.")


def visualize_visible_sky(visible_pixels, nside=64):
    """
    Visualize the visible HEALPix cells on a sky map.

    """
    sky_map = np.full(hp.nside2npix(nside), hp.UNSEEN)
    sky_map[visible_pixels] = 1

    hp.mollview(sky_map, title="Visible Sky", cmap="Blues", norm=None)

    plt.savefig("visible_sky.png")
    plt.close()


if __name__ == "__main__":
    latitude = 40.0
    longitude = 44.5

    current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    visible_cells_json = get_visible_healpix_cells(latitude, longitude, current_time_utc, output_format='json')
    visible_cells = json.loads(visible_cells_json)["visible_cells"]

    print(f"Visible HEALPix cells (JSON) at {current_time_utc}: {visible_cells_json}")

    visualize_visible_sky(visible_cells)
