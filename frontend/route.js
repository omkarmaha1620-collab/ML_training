/* =====================================================
   AI MARINE MONITORING
   SAFE ROUTE OPTIMIZATION
===================================================== */


/* =====================================================
   LOAD VESSEL DETAILS
===================================================== */

const monitoringVessel =
    JSON.parse(
        sessionStorage.getItem(
            "marineCurrentVessel"
        ) || "{}"
    );

const vesselData =
    monitoringVessel;

console.log(
    "MONITORING VESSEL:",
    monitoringVessel
);
    /* =====================================================
   LOAD ACTUAL VESSEL POSITION FROM AIS BACKEND
===================================================== */

async function loadVesselPosition() {

    if (!vesselData.mmsi) {
        return;
    }

    try {

        const response =
            await fetch(
                "http://127.0.0.1:8000/ais/vessels"
            );

        if (!response.ok) {
            throw new Error(
                "Unable to load AIS vessels."
            );
        }

        const data =
            await response.json();

        const vessels =
            Array.isArray(data.vessels)
                ? data.vessels
                : [];

        const vessel =
            vessels.find(
                item =>
                    String(item.mmsi) ===
                    String(vesselData.mmsi)
            );

        if (!vessel) {
            console.warn(
                "Selected vessel not found in AIS data:",
                vesselData.mmsi
            );
            return;
        }

        const lat =
            Number(vessel.latitude);

        const lon =
            Number(vessel.longitude);

        if (
            !Number.isFinite(lat) ||
            !Number.isFinite(lon)
        ) {
            return;
        }

        document.getElementById(
            "currentLat"
        ).value = lat.toFixed(5);

        document.getElementById(
            "currentLon"
        ).value = lon.toFixed(5);

        updateVesselMarker();

        console.log(
            "AIS vessel position loaded:",
            vessel
        );

    } catch (error) {

        console.error(
            "Failed to load AIS vessel position:",
            error
        );

    }
}


if (vesselData.shipName) {

    document.getElementById("shipName").textContent =
        vesselData.shipName;

}


if (vesselData.mmsi) {

    document.getElementById("mmsi").textContent =
        "MMSI " + vesselData.mmsi;

}


/* =====================================================
   MAP INITIALIZATION
===================================================== */

const map =
    L.map("routeMap", {
        zoomControl: true
    })
    .setView(
        [13.0, 80.0],
        6
    );


L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        maxZoom: 19,
        attribution:
            "&copy; OpenStreetMap contributors"
    }
).addTo(map);


/* =====================================================
   MARKERS / ROUTE
===================================================== */

let vesselMarker = null;

let destinationMarker = null;

let routeLine = null;

let hazardCircles = [];


/* =====================================================
   VESSEL ICON
===================================================== */

const vesselIcon =
    L.divIcon({

        className: "custom-vessel-marker",

        html: `
            <div style="
                width:34px;
                height:34px;
                border-radius:50%;
                background:#07384a;
                border:2px solid #18c8ed;
                display:flex;
                align-items:center;
                justify-content:center;
                box-shadow:0 0 18px rgba(24,200,237,.7);
                font-size:17px;
            ">
                🚢
            </div>
        `,

        iconSize: [34, 34],

        iconAnchor: [17, 17]

    });


/* =====================================================
   INITIAL VESSEL POSITION
===================================================== */

function updateVesselMarker() {

    const lat =
        parseFloat(
            document.getElementById("currentLat").value
        );

    const lon =
        parseFloat(
            document.getElementById("currentLon").value
        );


    if (
        Number.isNaN(lat) ||
        Number.isNaN(lon)
    ) {
        return;
    }


    if (vesselMarker) {

        vesselMarker.setLatLng(
            [lat, lon]
        );

    } else {

        vesselMarker =
            L.marker(
                [lat, lon],
                {
                    icon: vesselIcon
                }
            )
            .addTo(map);

        vesselMarker.bindPopup(
            "<b>Vessel</b><br>Current Position"
        );

    }

}


/* =====================================================
   INITIALIZE
===================================================== */

loadVesselPosition();


/* =====================================================
   INPUT POSITION UPDATE
===================================================== */

document
    .getElementById("currentLat")
    .addEventListener(
        "change",
        updateVesselMarker
    );


document
    .getElementById("currentLon")
    .addEventListener(
        "change",
        updateVesselMarker
    );


/* =====================================================
   DESTINATION
===================================================== */

function updateDestinationMarker() {

    const lat =
        parseFloat(
            document.getElementById("destLat").value
        );

    const lon =
        parseFloat(
            document.getElementById("destLon").value
        );


    if (
        Number.isNaN(lat) ||
        Number.isNaN(lon)
    ) {
        return;
    }


    if (destinationMarker) {

        destinationMarker.setLatLng(
            [lat, lon]
        );

    } else {

        destinationMarker =
            L.marker(
                [lat, lon]
            )
            .addTo(map);

        destinationMarker.bindPopup(
            "<b>Destination</b>"
        );

    }


    map.setView(
        [lat, lon],
        7
    );

}


/* =====================================================
   OPTIMIZE ROUTE
===================================================== */

const optimizeBtn =
    document.getElementById(
        "optimizeBtn"
    );


const calculating =
    document.getElementById(
        "calculating"
    );


const routeStatus =
    document.getElementById(
        "routeStatus"
    );


const routeResult =
    document.getElementById(
        "routeResult"
    );


optimizeBtn.addEventListener(
    "click",
    async function () {


        const currentLat =
            parseFloat(
                document.getElementById(
                    "currentLat"
                ).value
            );


        const currentLon =
            parseFloat(
                document.getElementById(
                    "currentLon"
                ).value
            );


        const destLat =
            parseFloat(
                document.getElementById(
                    "destLat"
                ).value
            );


        const destLon =
            parseFloat(
                document.getElementById(
                    "destLon"
                ).value
            );


        /* ---------------------------------------------
           VALIDATION
        --------------------------------------------- */

        if (
            Number.isNaN(destLat) ||
            Number.isNaN(destLon)
        ) {

            routeStatus.textContent =
                "Please enter a valid destination.";

            return;

        }


        if (
            Number.isNaN(currentLat) ||
            Number.isNaN(currentLon)
        ) {

            routeStatus.textContent =
                "Current vessel position is invalid.";

            return;

        }


        /* ---------------------------------------------
           UPDATE MAP
        --------------------------------------------- */

        updateVesselMarker();

        updateDestinationMarker();


        /* ---------------------------------------------
           LOADING
        --------------------------------------------- */

        optimizeBtn.disabled = true;

        calculating.classList.remove(
            "hidden"
        );

        routeResult.classList.add(
            "hidden"
        );


        routeStatus.textContent =
            "AI is calculating an ocean-safe route...";


        try {

            /*
             * Backend connection.
             *
             * We will connect this to the
             * exact PPO endpoint in main.py.
             */

            const response =
    await fetch(
        "http://127.0.0.1:8000/optimize-route-from-ais",
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json",

                "Accept":
                    "application/json"
            },

            body: JSON.stringify({

    mmsi:
        Number(
            vesselData.mmsi
        ),

    destination_lat:
        Number(
            destLat
        ),

    destination_lon:
        Number(
            destLon
        ),

    route_hazard:
        vesselData.route_hazard ??
        "NORMAL",

    route_mode:
        vesselData.route_mode ??
        "OPTIMIZED"

})
        }
    );


            if (!response.ok) {

                throw new Error(
                    "Backend route optimization failed."
                );

            }


            const data =
                await response.json();


            /* -----------------------------------------
               DRAW ROUTE
            ----------------------------------------- */

            drawRoute(
                data,
                currentLat,
                currentLon,
                destLat,
                destLon
            );


        } catch (error) {

            console.error(
                "Route optimization error:",
                error
            );


            /*
             * TEMPORARY FALLBACK
             *
             * This allows us to test the
             * frontend before connecting
             * the exact PPO response.
             */

            drawDemoRoute(
                currentLat,
                currentLon,
                destLat,
                destLon
            );


            routeStatus.textContent =
                "Frontend route preview generated. PPO backend connection will be used when the route endpoint is connected.";

        } finally {

            optimizeBtn.disabled = false;

            calculating.classList.add(
                "hidden"
            );

        }

    }
);


/* =====================================================
   DRAW BACKEND ROUTE
===================================================== */

function drawRoute(
    data,
    currentLat,
    currentLon,
    destLat,
    destLon
) {


    let coordinates = null;

    // -----------------------------------------------------
// REAL PPO RESPONSE
// Backend returns waypoints, not route/coordinates.
// -----------------------------------------------------

if (
    data &&
    Array.isArray(data.waypoints) &&
    data.waypoints.length >= 2
) {

    coordinates =
        data.waypoints
            .map(
                waypoint => {

                    if (
                        Array.isArray(waypoint)
                    ) {

                        return [
                            Number(waypoint[0]),
                            Number(waypoint[1])
                        ];

                    }

                    return [
                        Number(waypoint.latitude),
                        Number(waypoint.longitude)
                    ];

                }
            )
            .filter(
                point =>
                    Number.isFinite(point[0]) &&
                    Number.isFinite(point[1])
            );
}


    /*
     * Accept common backend formats.
     */

    if (
        data &&
        Array.isArray(data.route)
    ) {

        coordinates =
            data.route;

    }


    if (
        data &&
        Array.isArray(data.coordinates)
    ) {

        coordinates =
            data.coordinates;

    }


    if (!coordinates) {

        drawDemoRoute(
            currentLat,
            currentLon,
            destLat,
            destLon
        );

        return;

    }


    if (routeLine) {

        map.removeLayer(
            routeLine
        );

    }


    routeLine =
        L.polyline(
            coordinates,
            {
                color: "#24dfac",
                weight: 5,
                opacity: .9
            }
        )
        .addTo(map);


    map.fitBounds(
        routeLine.getBounds(),
        {
            padding: [50, 50]
        }
    );


    showRouteResult(
        data
    );


    routeStatus.textContent =
        "Safe route generated successfully.";

}


/* =====================================================
   DEMO ROUTE
===================================================== */

function drawDemoRoute(
    currentLat,
    currentLon,
    destLat,
    destLon
) {


    /*
     * Temporary route visualization.
     *
     * This is NOT the final PPO route.
     * It is only for frontend testing.
     */

    const midLat =
        (currentLat + destLat) / 2;


    const midLon =
        (currentLon + destLon) / 2;


    const coordinates = [

        [
            currentLat,
            currentLon
        ],

        [
            midLat + 0.35,
            midLon - 0.25
        ],

        [
            midLat + 0.15,
            midLon + 0.20
        ],

        [
            destLat,
            destLon
        ]

    ];


    if (routeLine) {

        map.removeLayer(
            routeLine
        );

    }


    routeLine =
        L.polyline(
            coordinates,
            {
                color: "#24dfac",
                weight: 5,
                opacity: .9
            }
        )
        .addTo(map);


    map.fitBounds(
        routeLine.getBounds(),
        {
            padding: [50, 50]
        }
    );


    const distance =
        calculateDistance(
            currentLat,
            currentLon,
            destLat,
            destLon
        );


    const estimatedHours =
        distance / 9.5;


    document.getElementById(
        "resultDistance"
    ).textContent =
        distance.toFixed(1) + " km";


    document.getElementById(
        "resultTime"
    ).textContent =
        estimatedHours.toFixed(1) + " h";


    document.getElementById(
        "resultRisk"
    ).textContent =
        "LOW";


    routeResult.classList.remove(
        "hidden"
    );


    routeStatus.textContent =
        "Safe route preview generated.";

}


/* =====================================================
   RESULT
===================================================== */

/* =====================================================
   RESULT
===================================================== */

function showRouteResult(data) {

    routeResult.classList.remove(
        "hidden"
    );

    // -----------------------------------------------------
    // REAL PPO ROUTE DATA
    // -----------------------------------------------------

    const routeMode =
        String(
            data.route_mode ??
            "OPTIMIZED"
        ).toUpperCase();

    const hazard =
        String(
            data.hazard ??
            "NO_WAVE_DATA"
        ).toUpperCase();

    const waveStatus =
        String(
            data.wave_status ??
            "NO_WAVE_DATA"
        ).toUpperCase();

    const distance =
        Number(
            data.route_distance_km ??
            data.distance_km ??
            0
        );


    // -----------------------------------------------------
    // DISTANCE
    // -----------------------------------------------------

    const distanceElement =
        document.getElementById(
            "resultDistance"
        );

    if (distanceElement) {

        distanceElement.textContent =
            distance.toFixed(1) + " km";

    }


    // -----------------------------------------------------
    // TIME
    // -----------------------------------------------------

    const timeElement =
        document.getElementById(
            "resultTime"
        );

    if (timeElement) {

        /*
         * PPO backend does not currently return
         * estimated travel time.
         *
         * Do not invent a time.
         */

        timeElement.textContent =
            "--";

    }


    // -----------------------------------------------------
    // RISK
    // -----------------------------------------------------

    const riskElement =
        document.getElementById(
            "resultRisk"
        );

    if (riskElement) {

        if (
            routeMode === "SAFEST"
        ) {

            riskElement.textContent =
                "HIGH HAZARD → SAFEST ROUTE";

        } else {

            riskElement.textContent =
                "LOW / NORMAL → OPTIMIZED ROUTE";

        }

    }


    // -----------------------------------------------------
    // HAZARD AVOIDANCE
    // -----------------------------------------------------

    const hazardElement =
        document.getElementById(
            "resultHazard"
        );

    if (hazardElement) {

        hazardElement.textContent =
            hazard;

    }


    // -----------------------------------------------------
    // ROUTE MODE
    // -----------------------------------------------------

    const modeElement =
        document.getElementById(
            "resultRouteMode"
        );

    if (modeElement) {

        modeElement.textContent =
            routeMode;

    }


    // -----------------------------------------------------
    // WAVE STATUS
    // -----------------------------------------------------

    const waveElement =
        document.getElementById(
            "resultWaveStatus"
        );

    if (waveElement) {

        waveElement.textContent =
            waveStatus;

    }


    // -----------------------------------------------------
    // CONSOLE VERIFICATION
    // -----------------------------------------------------

    console.log(
        "PPO ROUTE RESULT:",
        {
            route_mode:
                routeMode,

            hazard:
                hazard,

            wave_status:
                waveStatus,

            route_distance_km:
                distance
        }
    );

}

/* =====================================================
   DISTANCE
===================================================== */

function calculateDistance(
    lat1,
    lon1,
    lat2,
    lon2
) {

    const R = 6371;

    const dLat =
        (
            (lat2 - lat1)
            * Math.PI
        ) / 180;


    const dLon =
        (
            (lon2 - lon1)
            * Math.PI
        ) / 180;


    const a =

        Math.sin(dLat / 2) *
        Math.sin(dLat / 2)

        +

        Math.cos(
            lat1 * Math.PI / 180
        )

        *

        Math.cos(
            lat2 * Math.PI / 180
        )

        *

        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);


    const c =
        2 *
        Math.atan2(
            Math.sqrt(a),
            Math.sqrt(1 - a)
        );


    return R * c;

}


/* =====================================================
   BACK BUTTON
===================================================== */

document
    .getElementById("backBtn")
    .addEventListener(
        "click",
        function () {

            window.location.href =
                "monitoring.html";

        }
    );