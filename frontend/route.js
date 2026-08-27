/* ============================================================
   AI MARINE MONITORING
   SAFE ROUTE OPTIMIZATION

   FLOW:

   VesselAPI
       “
   Selected live vessel
       “
   Current vessel position
       “
   XGBoost / hazard information
       “
   PPO route optimization
       “
   Safe / safest route
============================================================ */


/* ============================================================
   CONFIGURATION
============================================================ */

const API_BASE =
    "http://127.0.0.1:8000";


/* ============================================================
   LOAD SELECTED VESSEL
============================================================ */

let monitoringVessel = {};

try {

    monitoringVessel =
        JSON.parse(
            sessionStorage.getItem(
                "marineCurrentVessel"
            ) || "{}"
        );

} catch (error) {

    console.error(
        "Failed to read selected vessel:",
        error
    );

    monitoringVessel = {};
}


let vesselData =
    monitoringVessel;


console.log(
    "ROUTE PAGE SELECTED VESSEL:",
    vesselData
);


/* ============================================================
   DOM HELPERS
============================================================ */

function getElement(id) {

    return document.getElementById(id);

}


function setText(id, value) {

    const element =
        getElement(id);

    if (element) {

        element.textContent =
            value;

    }

}


/* ============================================================
   VESSEL STATE
============================================================ */

let latestVessel = null;

let latestRisk = null;

let latestStorm = null;

let latestLSTM = null;
let routeHazard = "NORMAL";


/* ============================================================
   LOAD VESSEL DETAILS INTO PAGE
============================================================ */

function displaySelectedVessel() {

    const shipName =
        vesselData.shipName ??
        vesselData.ship_name ??
        vesselData.name ??
        "UNKNOWN VESSEL";


    const mmsi =
        vesselData.mmsi ??
        "--";


    setText(
        "shipName",
        shipName
    );


    setText(
        "mmsi",
        "MMSI " + mmsi
    );


    console.log(
        "DISPLAYED VESSEL:",
        {
            shipName,
            mmsi
        }
    );

}


/* ============================================================
   LOAD CURRENT LIVE VESSEL FROM VESSELAPI BACKEND
============================================================ */

async function loadVesselPosition() {

    if (!vesselData.mmsi) {

        console.warn(
            "No MMSI available for route page."
        );

        return null;

    }


    try {

        const response =
                    await fetch(
                        `${API_BASE}/ais/vessels?t=${Date.now()}`,
                        {

                            method:
                                "GET",

                            headers:
                                {
                                "Accept":
                                    "application/json"
                                }
                        }
                    );


                if (!response.ok) {

            throw new Error(
                `AIS request failed: ${response.status}`
            );

        }


        const data =
            await response.json();


        const vessels =
            Array.isArray(
                data.vessels
            )
                ? data.vessels
                : [];


        console.log(
            "ROUTE AIS SOURCE:",
            data.source
        );


        /*
         * IMPORTANT:
         *
         * Do NOT simply use vessels[0].
         *
         * Find the exact MMSI selected
         * on the monitoring page.
         */

        const vessel =
            vessels.find(
                item =>
                    String(
                        item.mmsi
                    ) ===
                    String(
                        vesselData.mmsi
                    )
            );


        if (!vessel) {

            console.warn(
                "Selected vessel is not currently available:",
                vesselData.mmsi
            );

            return null;

        }


        latestVessel =
            vessel;


        /*
         * Update our local vessel object
         * with the newest VesselAPI values.
         */

        vesselData = {

            ...vesselData,

            ...vessel,

            mmsi:
                vessel.mmsi,

            shipName:
                vessel.ship_name ??
                vessel.shipName ??
                vesselData.shipName,

            latitude:
                vessel.latitude,

            longitude:
                vessel.longitude,

            speed:
                vessel.speed ??
                vessel.sog ??
                0,

            course:
                vessel.course ??
                vessel.cog ??
                0

        };


        /*
         * Keep sessionStorage synchronized.
         *
         * This prevents the route page from
         * continuing to use an old vessel position.
         */

        try {

            sessionStorage.setItem(
                "marineCurrentVessel",
                JSON.stringify(
                    vesselData
                )
            );

        } catch (storageError) {

            console.warn(
                "Unable to update sessionStorage:",
                storageError
            );

        }


        const lat =
            Number(
                vessel.latitude
            );


        const lon =
            Number(
                vessel.longitude
            );


        if (
            !Number.isFinite(lat) ||
            !Number.isFinite(lon)
        ) {

            console.warn(
                "Invalid live vessel coordinates:",
                vessel
            );

            return null;

        }


        const currentLat =
            getElement(
                "currentLat"
            );


        const currentLon =
            getElement(
                "currentLon"
            );


        if (currentLat) {

            currentLat.value =
                lat.toFixed(5);

        }


        if (currentLon) {

            currentLon.value =
                lon.toFixed(5);

        }


        displaySelectedVessel();

        updateVesselMarker();


        console.log(
            "LIVE VESSEL POSITION UPDATED:",
            {
                mmsi:
                    vessel.mmsi,

                name:
                    vessel.ship_name,

                latitude:
                    lat,

                longitude:
                    lon,

                speed:
                    vessel.speed,

                course:
                    vessel.course,

                timestamp:
                    vessel.timestamp,

                source:
                    vessel.source
            }
        );


        return vessel;


    } catch (error) {

        console.error(
            "Failed to load live vessel:",
            error
        );

        return null;

    }

}


/* ============================================================
   LOAD AI RISK FOR SELECTED VESSEL
============================================================ */

async function loadVesselRisk() {

    if (!vesselData.mmsi) {

        return null;

    }


    try {

        const response =
                    await fetch(
                        `${API_BASE}/ais/risk/${encodeURIComponent(vesselData.mmsi)}?t=${Date.now()}`,
                        {

                            method:
                                "GET",

                            headers:
                                {
                                "Accept":
                                    "application/json"
                                }
                        }
                    );


                if (!response.ok) {

            console.warn(
                "AI risk endpoint returned:",
                response.status
            );

            return null;

        }


        const data =
            await response.json();


        latestRisk =
            data;


        console.log(
            "ROUTE AI RISK:",
            data
        );


        /*
         * Store the complete AI result.
         */

        vesselData.routeRisk =
            data;


        /*
         * Random Forest vessel risk
         */

        vesselData.vessel_risk =
            data.random_forest ??
            data.vessel_risk ??
            data.risk ??
            vesselData.vessel_risk ??
            null;


        /*
         * XGBoost result
         */

        vesselData.xgboost_risk =
            data.xgboost ??
            data.xgboost_risk ??
            data.storm_prediction ??
            data.wave_risk ??
            null;


        /*
         * LSTM result
         */

        vesselData.lstm_prediction =
            data.lstm_prediction ??
            data.lstm ??
            data.lstm_result ??
            data.wave_prediction ??
            null;


        /*
         * Extract possible hazard values.
         */

        const xgb =
            vesselData.xgboost_risk;


        let hazard =
            data.route_hazard ??
            data.hazard ??
            data.wave_status ??
            data.risk_level ??
            null;


        if (
            xgb &&
            typeof xgb === "object"
        ) {

            hazard =
                xgb.hazard ??
                xgb.risk_level ??
                xgb.level ??
                xgb.prediction ??
                hazard;

        }


        /*
         * Normalize hazard.
         */

        if (hazard !== null) {

            hazard =
                String(
                    hazard
                ).toUpperCase();

        }


        if (
            hazard === "1" ||
            hazard === "TRUE" ||
            hazard === "HIGH_WAVE" ||
            hazard === "STORM" ||
            hazard === "HIGH"
        ) {

            vesselData.route_hazard =
                "HIGH_WAVE";

        } else {

            vesselData.route_hazard =
                hazard ??
                "NORMAL";

        }


        /*
         * Keep sessionStorage updated.
         */

        try {

            sessionStorage.setItem(
                "marineCurrentVessel",
                JSON.stringify(
                    vesselData
                )
            );

        } catch (error) {

            console.warn(
                "Could not save AI route state:",
                error
            );

        }


        console.log(
            "ROUTE HAZARD:",
            vesselData.route_hazard
        );


        return data;


    } catch (error) {

        console.warn(
            "Failed to load vessel AI risk:",
            error
        );

        return null;

    }

}


/* ============================================================
   MAP INITIALIZATION
============================================================ */

const map =
    L.map(
        "routeMap",
        {
            zoomControl:
                true
        }
    )
    .setView(
        [13.0, 80.0],
        6
    );


L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {

        maxZoom:
            19,

        attribution:
            "&copy; OpenStreetMap contributors"

    }
).addTo(
    map
);


/* ============================================================
   MAP OBJECTS
============================================================ */

let vesselMarker =
    null;

let destinationMarker =
    null;

let routeLine =
    null;

let hazardCircles =
    [];


/* ============================================================
   VESSEL ICON
============================================================ */

const vesselIcon =
    L.divIcon(
        {

            className:
                "custom-vessel-marker",

            html:
                `
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
                    
                &#128674;</div>
                `,

            iconSize:
                [34, 34],

            iconAnchor:
                [17, 17]

        }
    );


/* ============================================================
   UPDATE VESSEL MARKER
============================================================ */

function updateVesselMarker() {

    const latElement =
        getElement(
            "currentLat"
        );


    const lonElement =
        getElement(
            "currentLon"
        );


    if (
        !latElement ||
        !lonElement
    ) {

        return;

    }


    const lat =
        parseFloat(
            latElement.value
        );


    const lon =
        parseFloat(
            lonElement.value
        );


    if (
        !Number.isFinite(lat) ||
        !Number.isFinite(lon)
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
                    icon:
                        vesselIcon
                }
            )
            .addTo(
                map
            );


        vesselMarker.bindPopup(
            `
            <b>
                ${vesselData.shipName ?? "Vessel"}
            </b>
            <br>
            MMSI:
            ${vesselData.mmsi ?? "--"}
            <br>
            Live VesselAPI Position
            `
        );

    }

}


/* ============================================================
   DESTINATION MARKER
============================================================ */

function updateDestinationMarker() {

    const latElement =
        getElement(
            "destLat"
        );


    const lonElement =
        getElement(
            "destLon"
        );


    if (
        !latElement ||
        !lonElement
    ) {

        return;

    }


    const lat =
        parseFloat(
            latElement.value
        );


    const lon =
        parseFloat(
            lonElement.value
        );


    if (
        !Number.isFinite(lat) ||
        !Number.isFinite(lon)
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
            .addTo(
                map
            );


        destinationMarker.bindPopup(
            "<b>Destination</b>"
        );

    }

}


/* ============================================================
   INPUT POSITION EVENTS
============================================================ */

const currentLatElement =
    getElement(
        "currentLat"
    );


const currentLonElement =
    getElement(
        "currentLon"
    );


if (currentLatElement) {

    currentLatElement.addEventListener(
        "change",
        updateVesselMarker
    );

}


if (currentLonElement) {

    currentLonElement.addEventListener(
        "change",
        updateVesselMarker
    );

}


/* ============================================================
   DESTINATION EVENTS
============================================================ */

const destLatElement =
    getElement(
        "destLat"
    );


const destLonElement =
    getElement(
        "destLon"
    );


if (destLatElement) {

    destLatElement.addEventListener(
        "change",
        updateDestinationMarker
    );

}


if (destLonElement) {

    destLonElement.addEventListener(
        "change",
        updateDestinationMarker
    );

}


/* ============================================================
   GET CURRENT ROUTE HAZARD
============================================================ */

function getRouteHazard() {

    /*
     * Prefer the freshly fetched AI result.
     */

    if (latestRisk) {

        const possibleHazard =
            latestRisk.route_hazard ??
            latestRisk.hazard ??
            latestRisk.wave_status ??
            latestRisk.risk_level;


        if (
            possibleHazard !== undefined &&
            possibleHazard !== null
        ) {

            return String(
                possibleHazard
            ).toUpperCase();

        }

    }


    /*
     * Otherwise use stored vessel state.
     */

    if (
        vesselData.route_hazard
    ) {

        return String(
            vesselData.route_hazard
        ).toUpperCase();

    }


    return "NORMAL";

}


/* ============================================================
   GET NDBC OBSERVATIONS IF AVAILABLE
============================================================ */

async function loadNDBCData() {

    /*
     * The route page does not fabricate NDBC values.
     *
     * If the backend has a route hazard already,
     * PPO can use that.
     *
     * Otherwise this function returns null.
     */

    try {

        const response =
                    await fetch(
                        `${API_BASE}/health?t=${Date.now()}`,
                        {

                            method:
                                "GET",

                            headers:
                                {
                                "Accept":
                                    "application/json"
                                }
                        }
                    );


                if (!response.ok) {

            return null;

        }


        return null;

    } catch (error) {

        return null;

    }

}


/* ============================================================
   OPTIMIZE BUTTON
============================================================ */

const optimizeBtn =
    getElement(
        "optimizeBtn"
    );


const calculating =
    getElement(
        "calculating"
    );


const routeStatus =
    getElement(
        "routeStatus"
    );


const routeResult =
    getElement(
        "routeResult"
    );


if (optimizeBtn) {

    optimizeBtn.addEventListener(
        "click",
        async function () {

            /*
             * ALWAYS refresh the selected vessel
             * before running PPO.
             */

            if (routeStatus) {

                routeStatus.textContent =
                    "Refreshing live vessel and AI hazard...";

            }


            optimizeBtn.disabled =
                true;


            if (calculating) {

                calculating.classList.remove(
                    "hidden"
                );

            }


            try {

                /*
                 * ------------------------------------------------
                 * STEP 1
                 * Get newest VesselAPI position.
                 * ------------------------------------------------
                 */

                const liveVessel =
                    await loadVesselPosition();


                if (!liveVessel) {

                    throw new Error(
                        "Selected live vessel is not available from VesselAPI."
                    );

                }


                /*
                 * ------------------------------------------------
                 * STEP 2
                 * Get AI risk / hazard.
                 * ------------------------------------------------
                 */

                await loadVesselRisk();


                /*
                 * ------------------------------------------------
                 * STEP 3
                 * Read current position.
                 * ------------------------------------------------
                 */

                const currentLat =
                    parseFloat(
                        getElement(
                            "currentLat"
                        )?.value
                    );


                const currentLon =
                    parseFloat(
                        getElement(
                            "currentLon"
                        )?.value
                    );


                /*
                 * ------------------------------------------------
                 * STEP 4
                 * Read destination.
                 * ------------------------------------------------
                 */

                const destLat =
                    parseFloat(
                        getElement(
                            "destLat"
                        )?.value
                    );


                const destLon =
                    parseFloat(
                        getElement(
                            "destLon"
                        )?.value
                    );


                /*
                 * ------------------------------------------------
                 * VALIDATION
                 * ------------------------------------------------
                 */

                if (
                    !Number.isFinite(
                        currentLat
                    ) ||
                    !Number.isFinite(
                        currentLon
                    )
                ) {

                    throw new Error(
                        "Current live vessel position is invalid."
                    );

                }


                if (
                    !Number.isFinite(
                        destLat
                    ) ||
                    !Number.isFinite(
                        destLon
                    )
                ) {

                    throw new Error(
                        "Please enter a valid destination."
                    );

                }


                updateVesselMarker();

                updateDestinationMarker();


                /*
                 * ------------------------------------------------
                 * STEP 5
                 * Determine current hazard.
                 * ------------------------------------------------
                 */
                routeHazard =
                    getRouteHazard();


                console.log(
                    "PPO INPUT:",
                    {

                        mmsi:
                            vesselData.mmsi,

                        currentLat,

                        currentLon,

                        destinationLat:
                            destLat,

                        destinationLon:
                            destLon,

                        routeHazard

                    }
                );


                if (routeStatus) {

                    routeStatus.textContent =
                        `AI is optimizing route using ${routeHazard} hazard information...`;

                }


                /*
                 * ------------------------------------------------
                 * STEP 6
                 * Call PPO backend.
                 *
                 * This is the endpoint already used
                 * by your existing route.js.
                 * ------------------------------------------------
                 */

                const response =
                    await fetch(
                        `${API_BASE}/optimize-route-from-ais`,
                        {

                            method:
                                "POST",

                            headers:
                                {
                                    "Content-Type":
                                        "application/json",

                                    "Accept":
                                        "application/json"
                                },

                            body:
                                JSON.stringify(
                                    {

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
                                    }
                                )
                        }
                    );


                if (!response.ok) {

                    let errorMessage =
                        "PPO route optimization failed.";


                    try {

                        const errorData =
                            await response.json();


                        errorMessage =
                            errorData.detail ??
                            errorData.message ??
                            errorMessage;

                    } catch (error) {

                        // Keep default error.

                    }


                    throw new Error(
                        errorMessage
                    );

                }


                /*
                 * ------------------------------------------------
                 * STEP 7
                 * Read PPO response.
                 * ------------------------------------------------
                 */

                const data =
                    await response.json();


                console.log(
                    "PPO BACKEND RESPONSE:",
                    data
                );


                /*
                 * ------------------------------------------------
                 * STEP 8
                 * Draw actual PPO route.
                 * ------------------------------------------------
                 */

                drawRoute(
                    data,
                    currentLat,
                    currentLon,
                    destLat,
                    destLon
                );


            } catch (error) {

                console.error(
                    "ROUTE OPTIMIZATION ERROR:",
                    error
                );


                if (routeStatus) {

                    routeStatus.textContent =
                        "Route optimization failed: " +
                        error.message;

                }


                /*
                 * IMPORTANT:
                 *
                 * We DO NOT draw a fake route here.
                 *
                 * If PPO fails, show the error.
                 *
                 * This prevents a demo route from being
                 * mistaken for the actual PPO route.
                 */

                if (routeResult) {

                    routeResult.classList.add(
                        "hidden"
                    );

                }

            } finally {

                optimizeBtn.disabled =
                    false;


                if (calculating) {

                    calculating.classList.add(
                        "hidden"
                    );

                }

            }

        }
    );

}


/* ============================================================
   DRAW PPO ROUTE
============================================================ */

function drawRoute(
    data,
    currentLat,
    currentLon,
    destLat,
    destLon
) {

    if (!data) {

        throw new Error(
            "Empty PPO response."
        );

    }


    let coordinates =
        null;


    /*
     * ----------------------------------------------------------
     * FORMAT 1
     *
     * PPO backend:
     *
     * {
     *   waypoints: [
     *      [lat, lon],
     *      [lat, lon]
     *   ]
     * }
     * ----------------------------------------------------------
     */

    if (
        Array.isArray(
            data.waypoints
        ) &&
        data.waypoints.length >= 2
    ) {

        coordinates =
            data.waypoints
                .map(
                    waypoint => {

                        if (
                            Array.isArray(
                                waypoint
                            )
                        ) {

                            return [
                                Number(
                                    waypoint[0]
                                ),
                                Number(
                                    waypoint[1]
                                )
                            ];

                        }


                        if (
                            waypoint &&
                            typeof waypoint ===
                            "object"
                        ) {

                            return [
                                Number(
                                    waypoint.latitude ??
                                    waypoint.lat
                                ),

                                Number(
                                    waypoint.longitude ??
                                    waypoint.lon ??
                                    waypoint.lng
                                )
                            ];

                        }


                        return null;

                    }
                )
                .filter(
                    point =>
                        point &&
                        Number.isFinite(
                            point[0]
                        ) &&
                        Number.isFinite(
                            point[1]
                        )
                );

    }


    /*
     * ----------------------------------------------------------
     * FORMAT 2
     * ----------------------------------------------------------
     */

    if (
        !coordinates &&
        Array.isArray(
            data.route
        )
    ) {

        coordinates =
            data.route;

    }


    /*
     * ----------------------------------------------------------
     * FORMAT 3
     * ----------------------------------------------------------
     */

    if (
        !coordinates &&
        Array.isArray(
            data.coordinates
        )
    ) {

        coordinates =
            data.coordinates;

    }


    /*
     * ----------------------------------------------------------
     * PPO DID NOT RETURN A ROUTE
     * ----------------------------------------------------------
     */

    if (
        !coordinates ||
        coordinates.length < 2
    ) {

        throw new Error(
            "PPO returned no valid route waypoints."
        );

    }


    /*
     * Remove old route.
     */

    if (routeLine) {

        map.removeLayer(
            routeLine
        );

    }


    /*
     * Remove old hazards.
     */

    hazardCircles.forEach(
        circle => {

            try {

                map.removeLayer(
                    circle
                );

            } catch (error) {

                // Ignore.

            }

        }
    );


    hazardCircles =
        [];


    /*
     * Draw actual PPO route.
     */

    routeLine =
        L.polyline(
            coordinates,
            {

                color:
                    "#24dfac",

                weight:
                    5,

                opacity:
                    0.95

            }
        )
        .addTo(
            map
        );


    /*
     * Draw hazard points if backend provides them.
     */

    drawHazards(
        data
    );


    /*
     * Fit map around route.
     */

    map.fitBounds(
        routeLine.getBounds(),
        {
            padding:
                [50, 50]
        }
    );


    /*
     * Show route result.
     */

    showRouteResult(
        data
    );


    if (routeStatus) {

        routeStatus.textContent =
            "PPO safe route generated successfully.";

    }


    console.log(
        "ACTUAL PPO ROUTE DRAWN:",
        coordinates
    );

}


/* ============================================================
   DRAW HAZARDS
============================================================ */

function drawHazards(
    data
) {

    const hazards =
        Array.isArray(
            data.hazards
        )
            ? data.hazards
            : [];


    hazards.forEach(
        hazard => {

            const lat =
                Number(
                    hazard.latitude ??
                    hazard.lat
                );


            const lon =
                Number(
                    hazard.longitude ??
                    hazard.lon ??
                    hazard.lng
                );


            if (
                !Number.isFinite(lat) ||
                !Number.isFinite(lon)
            ) {

                return;

            }


            const radius =
                Number(
                    hazard.radius ??
                    10000
                );


            const circle =
                L.circle(
                    [lat, lon],
                    {

                        radius,

                        color:
                            "#ff4f81",

                        fillColor:
                            "#ff4f81",

                        fillOpacity:
                            0.15,

                        weight:
                            2

                    }
                )
                .addTo(
                    map
                );


            circle.bindPopup(
                `
                <b>Hazard Zone</b>
                <br>
                ${hazard.type ?? "Marine Hazard"}
                `
            );


            hazardCircles.push(
                circle
            );

        }
    );

}


/* ============================================================
   SHOW PPO RESULT
============================================================ */

function showRouteResult(
    data
) {

    if (!routeResult) {

        return;

    }


    routeResult.classList.remove(
        "hidden"
    );


    /*
     * ----------------------------------------------------------
     * ROUTE MODE
     * ----------------------------------------------------------
     */

    const routeMode =
        String(
            data.route_mode ??
            data.mode ??
            "OPTIMIZED"
        )
        .toUpperCase();


    /*
     * ----------------------------------------------------------
     * HAZARD
     * ----------------------------------------------------------
     */

    const hazard =
        String(
            data.hazard ??
            data.route_hazard ??
            vesselData.route_hazard ??
            "NORMAL"
        )
        .toUpperCase();


    /*
     * ----------------------------------------------------------
     * WAVE STATUS
     * ----------------------------------------------------------
     */

    const waveStatus =
        String(
            data.wave_status ??
            data.waveStatus ??
            "NO_WAVE_DATA"
        )
        .toUpperCase();


    /*
     * ----------------------------------------------------------
     * DISTANCE
     * ----------------------------------------------------------
     */

    let distance =
        Number(
            data.route_distance_km ??
            data.distance_km ??
            data.distance ??
            0
        );


    /*
     * If backend doesn't return distance,
     * calculate it from start/end.
     */

    if (
        !Number.isFinite(distance) ||
        distance <= 0
    ) {

        const currentLat =
            Number(
                getElement(
                    "currentLat"
                )?.value
            );


        const currentLon =
            Number(
                getElement(
                    "currentLon"
                )?.value
            );


        const destLat =
            Number(
                getElement(
                    "destLat"
                )?.value
            );


        const destLon =
            Number(
                getElement(
                    "destLon"
                )?.value
            );


        if (
            Number.isFinite(
                currentLat
            ) &&
            Number.isFinite(
                currentLon
            ) &&
            Number.isFinite(
                destLat
            ) &&
            Number.isFinite(
                destLon
            )
        ) {

            distance =
                calculateDistance(
                    currentLat,
                    currentLon,
                    destLat,
                    destLon
                );

        }

    }


    /*
     * ----------------------------------------------------------
     * UPDATE DISTANCE
     * ----------------------------------------------------------
     */

    const distanceElement =
        getElement(
            "resultDistance"
        );


    if (distanceElement) {

        distanceElement.textContent =
            Number.isFinite(distance)
                ? distance.toFixed(1) + " km"
                : "--";

    }


    /*
     * ----------------------------------------------------------
     * TIME
     *
     * Prefer backend value.
     *
     * We don't invent travel time if backend
     * doesn't provide it.
     * ----------------------------------------------------------
     */

    const timeElement =
        getElement(
            "resultTime"
        );


    if (timeElement) {

        const travelTime =
            data.estimated_time_hours ??
            data.travel_time_hours ??
            data.estimated_hours ??
            null;


        if (
            travelTime !== null &&
            Number.isFinite(
                Number(
                    travelTime
                )
            )
        ) {

            timeElement.textContent =
                Number(
                    travelTime
                ).toFixed(1) +
                " h";

        } else {

            timeElement.textContent =
                "--";

        }

    }


    /*
     * ----------------------------------------------------------
     * RISK
     * ----------------------------------------------------------
     */

    const riskElement =
        getElement(
            "resultRisk"
        );


    if (riskElement) {

        if (
            routeMode ===
            "SAFEST"
        ) {

            riskElement.textContent =
                "HIGH HAZARD  SAFEST ROUTE";

        } else if (
            hazard.includes(
                "HIGH"
            ) ||
            hazard.includes(
                "STORM"
            ) ||
            hazard.includes(
                "WAVE"
            )
        ) {

            riskElement.textContent =
                "HAZARD DETECTED  SAFE ROUTE";

        } else {

            riskElement.textContent =
                "LOW / NORMAL  OPTIMIZED ROUTE";

        }

    }


    /*
     * ----------------------------------------------------------
     * HAZARD
     * ----------------------------------------------------------
     */

    const hazardElement =
        getElement(
            "resultHazard"
        );


    if (hazardElement) {

        hazardElement.textContent =
            hazard;

    }


    /*
     * ----------------------------------------------------------
     * ROUTE MODE
     * ----------------------------------------------------------
     */

    const modeElement =
        getElement(
            "resultRouteMode"
        );


    if (modeElement) {

        modeElement.textContent =
            routeMode;

    }


    /*
     * ----------------------------------------------------------
     * WAVE STATUS
     * ----------------------------------------------------------
     */

    const waveElement =
        getElement(
            "resultWaveStatus"
        );


    if (waveElement) {

        waveElement.textContent =
            waveStatus;

    }


    /*
     * ----------------------------------------------------------
     * OPTIONAL EXTRA RESULT FIELDS
     * ----------------------------------------------------------
     */

    const probabilityElement =
        getElement(
            "resultProbability"
        );


    if (probabilityElement) {

        let probability =
            data.probability_percent ??
            data.hazard_probability_percent ??
            null;


        if (
            probability !== null &&
            Number.isFinite(
                Number(
                    probability
                )
            )
        ) {

            probabilityElement.textContent =
                Number(
                    probability
                ).toFixed(1) +
                "%";

        } else {

            probabilityElement.textContent =
                "--";

        }

    }


    /*
     * ----------------------------------------------------------
     * CONSOLE VERIFICATION
     * ----------------------------------------------------------
     */

    console.log(
        "========== PPO ROUTE RESULT ==========",
        {

            route_mode:
                routeMode,

            hazard:
                hazard,

            wave_status:
                waveStatus,

            distance_km:
                distance,

            mmsi:
                vesselData.mmsi,

            vessel:
                vesselData.shipName,

            backend:
                data

        }
    );

}


/* ============================================================
   DISTANCE CALCULATION
============================================================ */

function calculateDistance(
    lat1,
    lon1,
    lat2,
    lon2
) {

    const R =
        6371;


    const dLat =
        (
            lat2 -
            lat1
        )
        *
        Math.PI /
        180;


    const dLon =
        (
            lon2 -
            lon1
        )
        *
        Math.PI /
        180;


    const a =
        Math.sin(
            dLat / 2
        ) *
        Math.sin(
            dLat / 2
        )
        +

        Math.cos(
            lat1 *
            Math.PI /
            180
        )
        *

        Math.cos(
            lat2 *
            Math.PI /
            180
        )
        *

        Math.sin(
            dLon / 2
        )
        *

        Math.sin(
            dLon / 2
        );


    const c =
        2 *
        Math.atan2(
            Math.sqrt(a),
            Math.sqrt(
                1 - a
            )
        );


    return R * c;

}


/* ============================================================
   INITIAL PAGE LOAD
============================================================ */

async function initializeRoutePage() {

    console.log(
        "======================================"
    );

    console.log(
        "INITIALIZING SAFE ROUTE PAGE"
    );

    console.log(
        "======================================"
    );


    /*
     * First display the selected vessel
     * from monitoring page.
     */

    displaySelectedVessel();


    /*
     * Then immediately replace its position
     * with the newest VesselAPI position.
     */

    await loadVesselPosition();


    /*
     * Then load AI risk/hazard.
     */

    await loadVesselRisk();


    /*
     * Update marker after everything.
     */

    updateVesselMarker();


    /*
     * Update destination marker.
     */

    updateDestinationMarker();


    /*
     * Center map on vessel if coordinates exist.
     */

    const lat =
        Number(
            getElement(
                "currentLat"
            )?.value
        );


    const lon =
        Number(
            getElement(
                "currentLon"
            )?.value
        );


    if (
        Number.isFinite(lat) &&
        Number.isFinite(lon)
    ) {

        map.setView(
            [lat, lon],
            7
        );

    }


    console.log(
        "ROUTE PAGE READY"
    );

}


/* ============================================================
   REFRESH LIVE VESSEL
============================================================ */

let routeRefreshTimer =
    null;


function startLiveRefresh() {

    /*
     * Refresh VesselAPI position every 30 seconds.
     *
     * This does NOT run PPO automatically.
     * It only keeps the selected vessel current.
     */

    routeRefreshTimer =
        setInterval(
            async function () {

                console.log(
                    "Refreshing live route vessel..."
                );


                await loadVesselPosition();


                await loadVesselRisk();


                updateVesselMarker();

            },
            30000
        );

}


/* ============================================================
   BACK BUTTON
============================================================ */

const backBtn =
    getElement(
        "backBtn"
    );


if (backBtn) {

    backBtn.addEventListener(
        "click",
        function () {

            /*
             * Return to monitoring page
             * without destroying the selected vessel.
             */

            window.location.href =
                "monitoring.html";

        }
    );

}


/* ============================================================
   START
============================================================ */

initializeRoutePage()
    .then(
        function () {

            startLiveRefresh();

        }
    )
    .catch(
        function (error) {

            console.error(
                "Route page initialization failed:",
                error
            );

        }
    );









