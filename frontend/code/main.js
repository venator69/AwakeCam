let map;
let marker;
let geocoder;

function initMap() {
    // Default location (e.g., center of the world if geolocation fails)
    const defaultLocation = { lat: 0, lng: 0 };

    map = new google.maps.Map(document.getElementById("map"), {
        center: defaultLocation,
        zoom: 2, // A low zoom level to see the world initially
    });

    geocoder = new google.maps.Geocoder();

    // Try HTML5 geolocation.
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const userLocation = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude,
                };

                map.setCenter(userLocation);
                map.setZoom(15); // Zoom in on the user's location

                marker = new google.maps.Marker({
                    position: userLocation,
                    map: map,
                    title: "You are here!",
                });

                // Reverse geocode to get a readable address
                geocoder.geocode({ 'location': userLocation }, (results, status) => {
                    if (status === 'OK') {
                        if (results[0]) {
                            document.getElementById('location-info').innerText = `You are currently at: ${results[0].formatted_address}`;
                        } else {
                            document.getElementById('location-info').innerText = 'No address found.';
                        }
                    } else {
                        document.getElementById('location-info').innerText = 'Geocoder failed due to: ' + status;
                    }
                });
            },
            () => {
                handleLocationError(true, map.getCenter());
            }
        );
    } else {
        // Browser doesn't support Geolocation
        handleLocationError(false, map.getCenter());
    }
}

function handleLocationError(browserHasGeolocation, pos) {
    document.getElementById('location-info').innerText = browserHasGeolocation
        ? "Error: The Geolocation service failed."
        : "Error: Your browser doesn't support geolocation.";
    // Still set a marker at the default or provided position if geolocation fails
    marker = new google.maps.Marker({
        position: pos,
        map: map,
        title: "Could not find your location.",
    });
}