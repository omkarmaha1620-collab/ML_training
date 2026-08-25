// ============================================================
// SHIP DETAILS â†’ MONITORING
// ============================================================

let shipProfile = {
    shipName: "",
    mmsi: "",
    shipType: ""
};


// ============================================================
// START MONITORING
// ============================================================

document.addEventListener("DOMContentLoaded", () => {

    const startButton =
        document.getElementById(
            "startMonitoringBtn"
        );

    const shipDetailsScreen =
        document.getElementById(
            "shipDetailsScreen"
        );

    const monitoringScreen =
        document.getElementById(
            "monitoringScreen"
        );

    const shipDetailsError =
        document.getElementById(
            "shipDetailsError"
        );


    if (!startButton) {

        console.error(
            "START MONITORING button not found."
        );

        return;
    }


    startButton.addEventListener(
        "click",
        async () => {

            const shipName =
                document.getElementById(
                    "inputShipName"
                ).value.trim();


            const mmsi =
                document.getElementById(
                    "inputMmsi"
                ).value.trim();


            const shipType =
                document.getElementById(
                    "inputShipType"
                ).value;


            // ------------------------------------------------
            // VALIDATION
            // ------------------------------------------------

            if (
                !shipName ||
                !mmsi ||
                !shipType
            ) {

                shipDetailsError.textContent =
                    "Please enter Ship Name, MMSI and Ship Type.";

                shipDetailsError.classList.remove(
                    "hidden"
                );

                return;
            }


            // ------------------------------------------------
            // MMSI VALIDATION
            // ------------------------------------------------

            if (
                !/^\d{9}$/.test(mmsi)
            ) {

                shipDetailsError.textContent =
                    "MMSI must contain exactly 9 digits.";

                shipDetailsError.classList.remove(
                    "hidden"
                );

                return;
            }


            // ------------------------------------------------
            // SAVE SHIP PROFILE
            // ------------------------------------------------

            shipProfile = {

                shipName:
                    shipName,

                mmsi:
                    mmsi,

                shipType:
                    shipType

            };


            // Save locally so a page refresh
            // doesn't immediately lose the details.

            localStorage.setItem(
                "marineShipProfile",
                JSON.stringify(
                    shipProfile
                )
            );

            // ------------------------------------------------
            // SAVE SELECTED MANUAL VESSEL FOR MONITORING PAGE
            // ------------------------------------------------

            sessionStorage.setItem(
                "marineShipProfile",
                JSON.stringify(
                    shipProfile
                )
            );

            sessionStorage.setItem(
                "marineCurrentVessel",
                JSON.stringify({
                    mmsi: shipProfile.mmsi,
                    name: shipProfile.shipName,
                    ship_name: shipProfile.shipName,
                    ship_type: shipProfile.shipType,
                    latitude: null,
                    longitude: null
                })
            );

            console.log(
                "MANUAL AIS VESSEL SELECTED:",
                shipProfile.mmsi
            );


            // ------------------------------------------------
            // UPDATE MONITORING SCREEN
            // ------------------------------------------------

            const selectedShipName =
                document.getElementById(
                    "selectedShipName"
                );

            const selectedShipMmsi =
                document.getElementById(
                    "selectedShipMmsi"
                );

            const selectedShipType =
                document.getElementById(
                    "selectedShipType"
                );


            if (selectedShipName) {

                selectedShipName.textContent =
                    shipProfile.shipName;

            }


            if (selectedShipMmsi) {

                selectedShipMmsi.textContent =
                    shipProfile.mmsi;

            }


            if (selectedShipType) {

                selectedShipType.textContent =
                    shipProfile.shipType;

            }


            // ------------------------------------------------
            // START BACKEND AIS -> NDBC PIPELINE
            // ------------------------------------------------

            try {

                console.log(
                    "SELECTING LOGIN VESSEL:",
                    shipProfile
                );

                const response =
                    await fetch(
                        `${API_BASE}/select-vessel`,
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify({
                                    mmsi:
                                        shipProfile.mmsi,

                                    ship_name:
                                        shipProfile.shipName,

                                    ship_type:
                                        shipProfile.shipType
                                })
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        "Backend could not find this vessel."
                    );
                }


                console.log(
                    "AIS -> NDBC RESULT:",
                    data
                );


                // Save backend-selected AIS vessel
                if (
                    data.vessel
                ) {

                    sessionStorage.setItem(
                        "marineCurrentVessel",
                        JSON.stringify(
                            data.vessel
                        )
                    );
                }


                // Save NDBC result
                sessionStorage.setItem(
                    "marineNDBCResult",
                    JSON.stringify(
                        data.ndbc ||
                        {}
                    )
                );


                console.log(
                    "AIS -> NDBC PIPELINE COMPLETE"
                );


            } catch (error) {

                console.error(
                    "AIS -> NDBC SELECTION FAILED:",
                    error
                );

                shipDetailsError.textContent =
                    error.message ||
                    "Unable to find this vessel.";

                shipDetailsError.classList.remove(
                    "hidden"
                );

                return;
            }

            // ------------------------------------------------
            // HIDE ERROR
            // ------------------------------------------------

            shipDetailsError.classList.add(
                "hidden"
            );


            // ------------------------------------------------
            // SWITCH SCREEN
            // ------------------------------------------------

            shipDetailsScreen.classList.add(
                "hidden"
            );

            monitoringScreen.classList.remove(
                "hidden"
            );


            console.log(
                "Ship profile:",
                shipProfile
            );


            // ------------------------------------------------
            // START EXISTING AIS SYSTEM
            // ------------------------------------------------

            if (
                typeof loadVessels ===
                "function"
            ) {

                loadVessels();

            }

        }
    );


    // ========================================================
    // RESTORE SAVED PROFILE
    // ========================================================

    try {

        const savedProfile =
            localStorage.getItem(
                "marineShipProfile"
            );


        if (savedProfile) {

            const parsed =
                JSON.parse(
                    savedProfile
                );


            if (
                parsed &&
                parsed.shipName &&
                parsed.mmsi &&
                parsed.shipType
            ) {

                document.getElementById(
                    "inputShipName"
                ).value =
                    parsed.shipName;


                document.getElementById(
                    "inputMmsi"
                ).value =
                    parsed.mmsi;


                document.getElementById(
                    "inputShipType"
                ).value =
                    parsed.shipType;

            }

        }

    } catch (error) {

        console.warn(
            "Could not restore ship profile:",
            error
        );

    }

});

// ============================================================
// AI MARINE MONITORING FRONTEND
// ============================================================

const API_BASE = "http://127.0.0.1:8000";

// ============================================================
// APPLICATION STATE
// ============================================================

const state = {
    map: null,

    vesselLayer: L.layerGroup(),

    routeLayer: L.layerGroup(),

    hazardLayer: L.layerGroup(),

    vessels: [],

    selected: null,

    selectedMarker: null,

    route: null,

    routeAnimationTimer: null,

    refreshTimer: null
};

// ============================================================
// HELPER
// ============================================================

const $ = (id) =>
    document.getElementById(id);

// ============================================================
// MAP
// ============================================================

function initMap() {

    state.map = L.map(
        "map",
        {
            zoomControl: true,
            preferCanvas: true
        }
    ).setView(
        [14.0, 79.0],
        5
    );

    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            maxZoom: 18,

            attribution:
                "&copy; OpenStreetMap contributors"
        }
    ).addTo(
        state.map
    );

    state.vesselLayer.addTo(
        state.map
    );

    state.routeLayer.addTo(
        state.map
    );

    state.hazardLayer.addTo(
        state.map
    );

    state.map.on(
        "mousemove",
        (event) => {

            if ($("cursorCoords")) {

                $("cursorCoords").textContent =
                    `LAT ${
                        event.latlng.lat.toFixed(4)
                    } / LON ${
                        event.latlng.lng.toFixed(4)
                    }`;
            }
        }
    );

    state.map.on(
        "click",
        () => {

            state.selected = null;

            state.selectedMarker = null;

            if ($("emptySelection")) {

                $("emptySelection")
                    .classList
                    .remove("hidden");
            }

            if ($("vesselDetails")) {

                $("vesselDetails")
                    .classList
                    .add("hidden");
            }
        }
    );
}

// ============================================================
// VESSEL NORMALIZATION
// ============================================================

function normalizeVessel(raw) {

    const risk =
        raw.vessel_risk ||
        raw.risk ||
        {};

    const features =
        risk.features ||
        {};

    const latitude =
        Number(
            raw.latitude ??
            raw.lat ??
            features.latitude
        );

    const longitude =
        Number(
            raw.longitude ??
            raw.lon ??
            features.longitude
        );

    const speed =
        Number(
            raw.speed ??
            features.average_speed_knots ??
            0
        );

    const course =
        Number(
            raw.course ??
            raw.heading ??
            0
        );

    const probability =
        Number(
            risk.probability ??
            risk.risk_probability ??
            0
        );

    let riskLevel =
        String(
            risk.risk_level ??
            "LOW"
        ).toUpperCase();

    if (
        riskLevel === "MODERATE"
    ) {

        riskLevel = "MEDIUM";
    }

    return {

        mmsi:
            raw.mmsi ??
            raw.MMSI ??
            "UNKNOWN",

        name:
            (
                raw.ship_name ??
                raw.name ??
                "UNKNOWN VESSEL"
            ).trim(),

        lat:
            latitude,

        lon:
            longitude,

        speed:
            Number.isFinite(speed)
                ? speed
                : 0,

        course:
            Number.isFinite(course)
                ? course
                : 0,

        riskPrediction:
            Number(
                risk.prediction ??
                0
            ),

        riskProbability:
            Number.isFinite(
                probability
            )
                ? probability
                : 0,

        riskLevel,

        features
    };
}

// ============================================================
// AIS LOADING
// ============================================================

async function loadVessels() {

    try {

        console.log(
            "Connecting to AIS backend..."
        );

        const response =
            await fetch(
                `${API_BASE}/ais/vessels`,
                {
                    method: "GET",

                    cache: "no-store",

                    headers: {
                        "Accept":
                            "application/json"
                    }
                }
            );

        if (!response.ok) {

            throw new Error(
                `AIS endpoint returned ${response.status}`
            );
        }

        const data =
            await response.json();

        console.log(
            "AIS data:",
            data
        );

        const rawVessels =
            Array.isArray(
                data.vessels
            )
                ? data.vessels
                : [];

        state.vessels =
            rawVessels
                .map(
                    normalizeVessel
                )
                .filter(
                    vessel =>
                        Number.isFinite(
                            vessel.lat
                        ) &&
                        Number.isFinite(
                            vessel.lon
                        )
                );

        const source =
            String(
                data.source ||
                "unknown"
            ).toUpperCase();

        console.log(
            `${source} vessels received: ${state.vessels.length}`
        );

        if ($("connectionText")) {

            $("connectionText").textContent =
                source === "LIVE"
                    ? "AIS CONNECTED"
                    : source === "DEMO"
                        ? "DEMO AIS"
                        : "AIS CONNECTED";
        }

        setConnection(true);

        renderVessels();

        updateStatistics();

        // ----------------------------------------------------
        // Preserve selected vessel
        // ----------------------------------------------------

        if (state.selected) {

            const fresh =
                state.vessels.find(
                    vessel =>
                        String(
                            vessel.mmsi
                        ) ===
                        String(
                            state.selected.mmsi
                        )
                );

            if (fresh) {

                state.selected =
                    fresh;

                renderDetails(
                    fresh
                );

            } else {

                state.selected =
                    null;

                state.selectedMarker =
                    null;
            }
        }

        // ----------------------------------------------------
        // SELECT THE EXACT MMSI ENTERED ON LOGIN
        // ----------------------------------------------------

        const manualMmsi =
            String(
                shipProfile?.mmsi ||
                ""
            ).trim();

        if (
            manualMmsi &&
            state.vessels.length > 0
        ) {

            const matchedVessel =
                state.vessels.find(
                    vessel =>
                        String(
                            vessel.mmsi
                        ).trim() ===
                        manualMmsi
                );

            if (matchedVessel) {

                let matchedMarker =
                    null;

                state.vesselLayer.eachLayer(
                    layer => {

                        if (
                            !matchedMarker &&
                            typeof layer.getLatLng ===
                                "function"
                        ) {

                            const latLng =
                                layer.getLatLng();

                            if (
                                Math.abs(
                                    latLng.lat -
                                    matchedVessel.lat
                                ) < 0.0001 &&
                                Math.abs(
                                    latLng.lng -
                                    matchedVessel.lon
                                ) < 0.0001
                            ) {

                                matchedMarker =
                                    layer;
                            }
                        }
                    }
                );

                state.selected =
                    matchedVessel;

                state.selectedMarker =
                    matchedMarker;

                // ------------------------------------------------
                // SAVE LIVE AIS DETAILS FOR MONITORING PAGE
                // ------------------------------------------------

                const monitoringVessel = {

                    ...matchedVessel,

                    shipName:
                        shipProfile.shipName,

                    ship_name:
                        shipProfile.shipName,

                    shipType:
                        shipProfile.shipType,

                    ship_type:
                        shipProfile.shipType
                };

                sessionStorage.setItem(
                    "marineCurrentVessel",
                    JSON.stringify(
                        monitoringVessel
                    )
                );

                sessionStorage.setItem(
                    "marineVesselDetails",
                    JSON.stringify(
                        shipProfile
                    )
                );

                renderDetails(
                    matchedVessel
                );

                console.log(
                    "EXACT LOGIN MMSI MATCHED:",
                    matchedVessel.mmsi
                );

                console.log(
                    "LIVE AIS POSITION:",
                    matchedVessel.lat,
                    matchedVessel.lon
                );

                console.log(
                    "MONITORING VESSEL SAVED:",
                    monitoringVessel
                );

            }
            else {

                console.warn(
                    "LOGIN MMSI NOT CURRENTLY FOUND IN LIVE AIS:",
                    manualMmsi
                );

                state.selected =
                    null;

                state.selectedMarker =
                    null;
            }

        }
        else {

            console.warn(
                "No manual MMSI available for AIS matching."
            );

            state.selected =
                null;

            state.selectedMarker =
                null;
        }

    } catch (error) {

        console.error(
            "AIS refresh failed:",
            error
        );

        setConnection(false);

        updateStatistics();
    }
}

// ============================================================
// CONNECTION STATUS
// ============================================================

function setConnection(
    online
) {

    if ($("connectionDot")) {

        $("connectionDot").className =
            `status-dot ${
                online
                    ? "online"
                    : "offline"
            }`;
    }

    if ($("connectionText")) {

        $("connectionText").textContent =
            online
                ? "AIS CONNECTED"
                : "AIS OFFLINE";
    }
}

// ============================================================
// VESSEL RISK COLOR
// ============================================================

function riskClassName(
    level
) {

    const value =
        String(level)
            .toLowerCase();

    if (
        value.includes("high")
    ) {

        return "high";
    }

    if (
        value.includes("medium") ||
        value.includes("moderate")
    ) {

        return "medium";
    }

    return "low";
}

// ============================================================
// RENDER VESSELS
// ============================================================

function renderVessels() {

    state.vesselLayer
        .clearLayers();

    if (
        $("toggleVessels") &&
        !$("toggleVessels").checked
    ) {

        return;
    }

    for (
        const vessel
        of state.vessels
    ) {

        const riskClass =
            riskClassName(
                vessel.riskLevel
            );

        const icon =
            L.divIcon(
                {
                    className:
                        `vessel-marker ${riskClass}`,

                    html:
                        `
                        <div
                            class="vessel-icon"
                            title="${escapeHtml(
                                vessel.name
                            )}"
                        >
                            ðŸš¢
                        </div>
                        `,

                    iconSize:
                        [30, 30],

                    iconAnchor:
                        [15, 15]
                }
            );

        const marker =
            L.marker(
                [
                    vessel.lat,
                    vessel.lon
                ],
                {
                    icon,

                    keyboard: false
                }
            );

        marker.bindTooltip(
            `
            <b>${escapeHtml(
                vessel.name
            )}</b>

            <br>

            Risk:
            ${escapeHtml(
                vessel.riskLevel
            )}

            <br>

            Speed:
            ${vessel.speed.toFixed(1)}
            kn
            `,
            {
                direction: "top",

                offset:
                    [0, -10]
            }
        );

        marker.on(
            "click",
            event => {

                L.DomEvent
                    .stopPropagation(
                        event
                    );

                selectVessel(
                    vessel,
                    marker
                );
            }
        );

        state.vesselLayer
            .addLayer(
                marker
            );
    }
}

// ============================================================
// SELECT VESSEL
// ============================================================

function selectVessel(
    vessel,
    marker
) {

    state.selected =
        vessel;

    state.selectedMarker =
        marker;

    if ($("emptySelection")) {

        $("emptySelection")
            .classList
            .add("hidden");
    }

    if ($("vesselDetails")) {

        $("vesselDetails")
            .classList
            .remove("hidden");
    }

    renderDetails(
        vessel
    );

    state.map.flyTo(
        [
            vessel.lat,
            vessel.lon
        ],
        Math.max(
            state.map.getZoom(),
            7
        ),
        {
            duration: 0.8
        }
    );
}

// ============================================================
// VESSEL DETAILS
// ============================================================

function renderDetails(
    vessel
) {

    const riskClass =
        riskClassName(
            vessel.riskLevel
        );

    if ($("shipName")) {

        $("shipName").textContent =
            vessel.name;
    }

    if ($("mmsi")) {

        $("mmsi").textContent =
            `MMSI ${vessel.mmsi}`;
    }

    if ($("riskLevel")) {

        $("riskLevel").textContent =
            vessel.riskLevel;

        $("riskLevel").className =
            `risk-value ${riskClass}`;
    }

    if ($("selectedRiskPill")) {

        $("selectedRiskPill").textContent =
            vessel.riskLevel;

        $("selectedRiskPill").className =
            `risk-pill ${riskClass}`;
    }

    if ($("riskProbability")) {

        $("riskProbability").textContent =
            `${(
                vessel.riskProbability *
                100
            ).toFixed(1)}% probability`;
    }

    if ($("detailLat")) {

        $("detailLat").textContent =
            vessel.lat.toFixed(4);
    }

    if ($("detailLon")) {

        $("detailLon").textContent =
            vessel.lon.toFixed(4);
    }

    if ($("detailSpeed")) {

        $("detailSpeed").textContent =
            `${vessel.speed.toFixed(1)} kn`;
    }

    if ($("detailCourse")) {

        $("detailCourse").textContent =
            `${vessel.course.toFixed(0)}Â°`;
    }

    if ($("shoreDistance")) {

        $("shoreDistance").textContent =
            formatKm(
                vessel.features
                    ?.end_distance_from_shore_km ??
                vessel.features
                    ?.start_distance_from_shore_km
            );
    }

    if ($("portDistance")) {

        $("portDistance").textContent =
            formatKm(
                vessel.features
                    ?.end_distance_from_port_km ??
                vessel.features
                    ?.start_distance_from_port_km
            );
    }

    if ($("averageSpeed")) {

        $("averageSpeed").textContent =
            formatKnots(
                vessel.features
                    ?.average_speed_knots
            );
    }

    if ($("totalDistance")) {

        $("totalDistance").textContent =
            formatKm(
                vessel.features
                    ?.total_distance_km
            );
    }

    if ($("routePanelStatus")) {

        $("routePanelStatus").textContent =
            state.route
                ? "Optimized route displayed"
                : "No route calculated. Enter a destination and optimize.";
    }
}

// ============================================================
// STATISTICS
// ============================================================

function updateStatistics() {

    if ($("vesselCount")) {

        $("vesselCount").textContent =
            state.vessels.length;
    }

    const highRisk =
        state.vessels.filter(
            vessel =>
                riskClassName(
                    vessel.riskLevel
                ) === "high"
        ).length;

    if ($("highRiskCount")) {

        $("highRiskCount").textContent =
            highRisk;
    }

    if ($("waveCount")) {

        $("waveCount").textContent =
            "--";
    }
}

// ============================================================
// FORMAT
// ============================================================

function formatKm(
    value
) {

    const number =
        Number(value);

    return Number.isFinite(
        number
    )
        ? `${number.toFixed(2)} km`
        : "-- km";
}

function formatKnots(
    value
) {

    const number =
        Number(value);

    return Number.isFinite(
        number
    )
        ? `${number.toFixed(2)} kn`
        : "-- kn";
}

// ============================================================
// CLEAR ROUTE
// ============================================================

function clearRoute() {

    state.routeLayer
        .clearLayers();

    state.route =
        null;

    if (
        state.routeAnimationTimer
    ) {

        clearInterval(
            state.routeAnimationTimer
        );

        state.routeAnimationTimer =
            null;
    }

    if ($("routeStatus")) {

        $("routeStatus").textContent =
            "Route cleared. Select a vessel and enter a destination.";
    }

    if ($("routePanelStatus")) {

        $("routePanelStatus").textContent =
            "No route calculated";
    }
}

// ============================================================
// OPTIMIZE ROUTE
// ============================================================

async function optimizeRoute() {

    if (!state.selected) {

        if ($("routeStatus")) {

            $("routeStatus").textContent =
                "Select a vessel on the map first.";
        }

        return;
    }

    const destLat =
        Number(
            $("destLat")?.value
        );

    const destLon =
        Number(
            $("destLon")?.value
        );

    if (
        !Number.isFinite(destLat) ||
        !Number.isFinite(destLon)
    ) {

        if ($("routeStatus")) {

            $("routeStatus").textContent =
                "Enter valid destination latitude and longitude.";
        }

        return;
    }

    showRouteAnimation(
        true,
        "AI ANALYZING SAFE OCEAN ROUTE..."
    );

    try {

        const route =
            await getOptimizedRoute(
                state.selected,
                destLat,
                destLon
            );

        drawAnimatedRoute(
            route
        );

        const ppo =
            route.ppo ||
            {};

        if ($("routeStatus")) {

            $("routeStatus").textContent =
                `PPO optimized ocean route displayed â€¢ ` +
                `Action ${
                    ppo.action ?? "--"
                } â€¢ Turn ${
                    Number(
                        ppo.turnAngle ?? 0
                    ).toFixed(0)
                }Â°`;
        }

        if ($("routePanelStatus")) {

            $("routePanelStatus").textContent =
                `Ocean route active â†’ ${
                    destLat.toFixed(4)
                }, ${
                    destLon.toFixed(4)
                }`;
        }

    } catch (error) {

        console.error(
            "Route optimization failed:",
            error
        );

        if ($("routeStatus")) {

            $("routeStatus").textContent =
                `Route calculation failed: ${
                    error.message
                }`;
        }

    } finally {

        setTimeout(
            () => {

                showRouteAnimation(
                    false
                );

            },
            650
        );
    }
}

// ============================================================
// REAL PPO ROUTE
// ============================================================

async function getOptimizedRoute(
    vessel,
    destLat,
    destLon
) {

    console.log(
        "Calling REAL PPO ocean-route backend..."
    );

    const response =
        await fetch(
            `${API_BASE}/optimize-route-from-ais`,
            {
                method: "POST",

                headers: {
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
                                    vessel.mmsi
                                ),

                            destination_lat:
                                Number(
                                    destLat
                                ),

                            destination_lon:
                                Number(
                                    destLon
                                )
                        }
                    )
            }
        );

    if (!response.ok) {

        let message =
            `PPO backend returned ${response.status}`;

        try {

            const errorData =
                await response.json();

            if (errorData.detail) {

                message =
                    String(
                        errorData.detail
                    );
            }

        } catch (_) {

            try {

                const text =
                    await response.text();

                if (text) {
                    message = text;
                }

            } catch (_) {}
        }

        throw new Error(
            message
        );
    }

    const ppo =
        await response.json();

    console.log(
        "REAL PPO RESPONSE:",
        ppo
    );

    if (
        !ppo.current_position ||
        !ppo.destination
    ) {

        throw new Error(
            "PPO response does not contain current position or destination."
        );
    }

    // --------------------------------------------------------
    // Backend current position
    // --------------------------------------------------------

    const start = [

        Number(
            ppo.current_position.latitude
        ),

        Number(
            ppo.current_position.longitude
        )

    ];

    // --------------------------------------------------------
    // Backend destination
    // --------------------------------------------------------

    const end = [

        Number(
            ppo.destination.latitude
        ),

        Number(
            ppo.destination.longitude
        )

    ];

    // --------------------------------------------------------
    // IMPORTANT:
    // Use ONLY backend-generated waypoints.
    //
    // The backend performs land collision checks.
    // The frontend must never invent a replacement
    // curved route across land.
    // --------------------------------------------------------

    let points = [];

    if (
        Array.isArray(
            ppo.waypoints
        ) &&
        ppo.waypoints.length >= 2
    ) {

        points =
            ppo.waypoints
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

                        return [

                            Number(
                                waypoint.latitude
                            ),

                            Number(
                                waypoint.longitude
                            )

                        ];
                    }
                )
                .filter(
                    point =>
                        Number.isFinite(
                            point[0]
                        ) &&
                        Number.isFinite(
                            point[1]
                        )
                );
    }

    if (
        points.length < 2
    ) {

        throw new Error(
            "Backend did not return a valid ocean-only route."
        );
    }

    // --------------------------------------------------------
    // Force exact backend start/end
    // --------------------------------------------------------

    points[0] =
        start;

    points[
        points.length - 1
    ] =
        end;

    return {

        points,

        distanceKm:
            Number(
                ppo.route_distance_km ??
                approximateDistanceKm(
                    points
                )
            ),

                ppo: {

            action:
                Number(
                    ppo.action ??
                    2
                ),

            turnAngle:
                Number(
                    ppo.turn_angle ??
                    0
                ),

            desiredBearing:
                Number(
                    ppo.desired_bearing ??
                    0
                ),

            headingError:
                Number(
                    ppo.heading_error ??
                    0
                ),

            originalDistanceKm:
                Number(
                    ppo.initial_distance_km ??
                    ppo.distance_km ??
                    0
                ),

            // ------------------------------------------------
            // ROUTE MODE / HAZARD FROM BACKEND
            // ------------------------------------------------

            routeMode:
                ppo.route_mode ??
                "OPTIMIZED",

            hazard:
                ppo.hazard ??
                "NO_WAVE_DATA",

            waveStatus:
                ppo.wave_status ??
                "NO_WAVE_DATA",

            wavePredictionM:
                ppo.wave_prediction_m ?? null
        },

        oceanOnly:
            ppo.ocean_only === true,

        landCheck:
            ppo.land_check === true
    };
}

// ============================================================
// DRAW ROUTE
// ============================================================

function drawAnimatedRoute(
    route
) {

    state.routeLayer
        .clearLayers();

    if (
        $("toggleRoute") &&
        !$("toggleRoute").checked
    ) {

        return;
    }

    if (
        !route ||
        !Array.isArray(
            route.points
        ) ||
        route.points.length < 2
    ) {

        throw new Error(
            "Invalid ocean route received."
        );
    }

    state.route =
        route;

    // --------------------------------------------------------
    // Route corridor
    // --------------------------------------------------------

    const corridor =
        L.polyline(
            route.points,
            {

                color:
                    "#31d7ff",

                weight:
                    14,

                opacity:
                    0.08,

                lineCap:
                    "round",

                lineJoin:
                    "round"
            }
        );

    state.routeLayer
        .addLayer(
            corridor
        );

    // --------------------------------------------------------
    // Main route
    // --------------------------------------------------------

    const line =
        L.polyline(
            route.points,
            {

                className:
                    "route-line route-moving",

                color:
                    "#31d7ff",

                weight:
                    5,

                opacity:
                    0.95,

                dashArray:
                    "12 14",

                lineCap:
                    "round",

                lineJoin:
                    "round"
            }
        );

    state.routeLayer
        .addLayer(
            line
        );

    // --------------------------------------------------------
    // Start marker
    // --------------------------------------------------------

    const start =
        route.points[0];

    const end =
        route.points[
            route.points.length - 1
        ];

    const startMarker =
        L.circleMarker(
            start,
            {

                radius:
                    7,

                color:
                    "#39e58c",

                fillColor:
                    "#39e58c",

                fillOpacity:
                    1,

                weight:
                    2
            }
        );

    startMarker.bindTooltip(
        "CURRENT VESSEL",
        {
            direction:
                "top"
        }
    );

    state.routeLayer
        .addLayer(
            startMarker
        );

    // --------------------------------------------------------
    // Destination
    // --------------------------------------------------------

    const destinationMarker =
        L.circleMarker(
            end,
            {

                radius:
                    8,

                color:
                    "#31d7ff",

                fillColor:
                    "#06111d",

                fillOpacity:
                    1,

                weight:
                    3
            }
        );

    destinationMarker.bindTooltip(
        "DESTINATION",
        {
            direction:
                "top"
        }
    );

    state.routeLayer
        .addLayer(
            destinationMarker
        );

    // --------------------------------------------------------
    // Animated AI navigation marker
    // --------------------------------------------------------

    const moving =
        L.circleMarker(
            start,
            {

                radius:
                    6,

                color:
                    "#ffffff",

                fillColor:
                    "#ffffff",

                fillOpacity:
                    1,

                weight:
                    2
            }
        );

    moving.bindTooltip(
        "PPO AI NAVIGATION",
        {
            direction:
                "top"
        }
    );

    state.routeLayer
        .addLayer(
            moving
        );

    // --------------------------------------------------------
    // Animation
    // --------------------------------------------------------

    if (
        state.routeAnimationTimer
    ) {

        clearInterval(
            state.routeAnimationTimer
        );
    }

    let index =
        0;

    state.routeAnimationTimer =
        setInterval(
            () => {

                index++;

                if (
                    index >=
                    route.points.length
                ) {

                    index = 0;
                }

                moving.setLatLng(
                    route.points[index]
                );

            },
            90
        );

    // --------------------------------------------------------
    // Status
    // --------------------------------------------------------

    if ($("routePanelStatus")) {

        const ppo =
            route.ppo ||
            {};

        const oceanText =
            route.oceanOnly
                ? "OCEAN SAFE"
                : "ROUTE";

                const routeMode =
            String(
                ppo.routeMode ??
                "OPTIMIZED"
            );

        const hazard =
            String(
                ppo.hazard ??
                "NO_WAVE_DATA"
            );

        const waveStatus =
            String(
                ppo.waveStatus ??
                "NO_WAVE_DATA"
            );

        $("routePanelStatus").textContent =
            `${oceanText} â€¢ PPO route active â€¢ ${
                route.distanceKm.toFixed(1)
            } km â€¢ MODE: ${
                routeMode
            } â€¢ HAZARD: ${
                hazard
            } â€¢ WAVE: ${
                waveStatus
            }`;
    }

    // --------------------------------------------------------
    // Zoom
    // --------------------------------------------------------

    state.map.fitBounds(
        line.getBounds(),
        {
            padding:
                [55, 55],

            maxZoom:
                9
        }
    );
}

// ============================================================
// DISTANCE
// ============================================================

function approximateDistanceKm(
    points
) {

    let total =
        0;

    for (
        let i = 1;
        i < points.length;
        i++
    ) {

        total +=
            haversineKm(
                points[i - 1][0],
                points[i - 1][1],

                points[i][0],
                points[i][1]
            );
    }

    return total;
}

function haversineKm(
    lat1,
    lon1,
    lat2,
    lon2
) {

    const R =
        6371;

    const dLat =
        toRad(
            lat2 - lat1
        );

    const dLon =
        toRad(
            lon2 - lon1
        );

    const a =
        Math.sin(
            dLat / 2
        ) ** 2 +

        Math.cos(
            toRad(lat1)
        ) *

        Math.cos(
            toRad(lat2)
        ) *

        Math.sin(
            dLon / 2
        ) ** 2;

    return (
        R *
        2 *
        Math.atan2(
            Math.sqrt(a),
            Math.sqrt(
                1 - a
            )
        )
    );
}

function toRad(
    value
) {

    return (
        value *
        Math.PI /
        180
    );
}

// ============================================================
// ROUTE LOADING ANIMATION
// ============================================================

function showRouteAnimation(
    show,
    text =
        "CALCULATING SAFE OCEAN ROUTE..."
) {

    if (
        !$("routeAnimation")
    ) {

        return;
    }

    $("routeAnimation")
        .classList
        .toggle(
            "hidden",
            !show
        );

    if (
        $("routeAnimationText")
    ) {

        $("routeAnimationText").textContent =
            text;
    }
}

// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHtml(
    value
) {

    return String(
        value
    )

        .replaceAll(
            "&",
            "&amp;"
        )

        .replaceAll(
            "<",
            "&lt;"
        )

        .replaceAll(
            ">",
            "&gt;"
        )

        .replaceAll(
            '"',
            "&quot;"
        )

        .replaceAll(
            "'",
            "&#039;"
        );
}

// ============================================================
// CONTROLS
// ============================================================

function setupControls() {

    if ($("optimizeBtn")) {

        $("optimizeBtn")
            .addEventListener(
                "click",
                optimizeRoute
            );
    }

    if ($("clearRouteBtn")) {

        $("clearRouteBtn")
            .addEventListener(
                "click",
                clearRoute
            );
    }

    if ($("toggleVessels")) {

        $("toggleVessels")
            .addEventListener(
                "change",
                renderVessels
            );
    }

    if ($("toggleRoute")) {

        $("toggleRoute")
            .addEventListener(
                "change",
                () => {

                    if (
                        $("toggleRoute")
                            .checked
                    ) {

                        if (
                            state.route
                        ) {

                            drawAnimatedRoute(
                                state.route
                            );
                        }

                    } else {

                        state.routeLayer
                            .clearLayers();
                    }
                }
            );
    }

    if ($("toggleHazard")) {

        $("toggleHazard")
            .addEventListener(
                "change",
                () => {

                    if (
                        $("toggleHazard")
                            .checked
                    ) {

                        state.hazardLayer
                            .addTo(
                                state.map
                            );

                    } else {

                        state.hazardLayer
                            .clearLayers();
                    }
                }
            );
    }
}

// ============================================================
// CLOCK
// ============================================================

function updateClock() {

    const now =
        new Date();

    if ($("clock")) {

        $("clock").textContent =
            `${now
                .toISOString()
                .slice(
                    11,
                    19
                )} UTC`;
    }
}

// ============================================================
// APPLICATION START
// ============================================================

async function start() {

    console.log(
        "AI Marine Monitoring starting..."
    );

    initMap();

    setupControls();

    updateClock();

    setInterval(
        updateClock,
        1000
    );

    // --------------------------------------------------------
    // Initial AIS load
    // --------------------------------------------------------

    await loadVessels();

    // --------------------------------------------------------
    // Refresh every 5 seconds
    // --------------------------------------------------------

    state.refreshTimer =
        setInterval(
            loadVessels,
            5000
        );

    console.log(
        "AI Marine Monitoring ready."
    );
}

// ============================================================
// START
// ============================================================

start();
