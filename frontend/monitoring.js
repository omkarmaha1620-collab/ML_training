// ============================================================
// AI MARINE MONITORING SYSTEM
// FRESH COMPLETE MONITORING.JS
// ============================================================

"use strict";


// ============================================================
// CONFIGURATION
// ============================================================

const API_BASE =
    "http://127.0.0.1:8000";

const AIS_REFRESH_MS =
    5000;

const HEALTH_REFRESH_MS =
    10000;

const LSTM_REFRESH_MS =
    5000;


// ============================================================
// GLOBAL STATE
// ============================================================

let currentVessel = null;

let allVessels = [];

let monitoringTimer = null;

let healthTimer = null;

let lstmTimer = null;

let requestInProgress = false;

let lstmRequestInProgress = false;


// ============================================================
// DEMO FALLBACK
//
// Set to false when you want ONLY real AIS vessels.
// ============================================================

const DEMO_FALLBACK_ENABLED = true;


// ============================================================
// SHIP DETAILS FROM LOCAL STORAGE
// ============================================================

let shipDetails = {};

try {

    shipDetails =
        JSON.parse(
            localStorage.getItem(
                "marineVessel"
            ) || "{}"
        );

} catch (error) {

    console.warn(
        "Could not read marineVessel from localStorage:",
        error
    );
}


const enteredShipName =
    shipDetails.shipName || "";


const enteredMmsi =
    shipDetails.mmsi || "";


const enteredShipType =
    shipDetails.shipType || "--";


// ============================================================
// BASIC DOM HELPER
// ============================================================

function setText(
    id,
    value
) {

    const element =
        document.getElementById(id);

    if (!element) {
        return;
    }

    element.textContent =
        value;
}


// ============================================================
// RISK CSS HELPER
// ============================================================

function setRiskClass(
    id,
    level
) {

    const element =
        document.getElementById(id);

    if (!element) {
        return;
    }


    element.classList.remove(
        "low",
        "medium",
        "high",
        "safe",
        "warning",
        "danger",
        "moderate"
    );


    const normalized =
        String(level || "")
            .toLowerCase()
            .replace(
                /\s+/g,
                "_"
            );


    if (
        normalized === "high" ||
        normalized === "high_wave" ||
        normalized === "danger"
    ) {

        element.classList.add(
            "high",
            "danger"
        );

    } else if (
        normalized === "medium" ||
        normalized === "moderate" ||
        normalized === "warning"
    ) {

        element.classList.add(
            "medium",
            "moderate",
            "warning"
        );

    } else if (
        normalized === "low" ||
        normalized === "safe" ||
        normalized === "normal"
    ) {

        element.classList.add(
            "low",
            "safe"
        );
    }
}


// ============================================================
// PROGRESS BAR
// ============================================================

function updateProgress(
    id,
    value
) {

    const progress =
        document.getElementById(id);

    if (!progress) {
        return;
    }


    let numeric =
        Number(value);


    if (
        !Number.isFinite(
            numeric
        )
    ) {

        numeric = 0;
    }


    numeric =
        Math.max(
            0,
            Math.min(
                100,
                numeric
            )
        );


    progress.style.width =
        `${numeric}%`;
}


// ============================================================
// CLOCK
// ============================================================

function updateClock() {

    const clock =
        document.getElementById(
            "monitorClock"
        );


    if (!clock) {
        return;
    }


    clock.textContent =
        new Date()
            .toISOString()
            .substring(
                11,
                19
            ) +
        " UTC";
}


updateClock();


setInterval(
    updateClock,
    1000
);


// ============================================================
// INITIAL SHIP DISPLAY
// ============================================================

function displayInitialShipDetails() {

    setText(
        "monitorShipName",
        enteredShipName ||
        "WAITING FOR LIVE AIS"
    );


    setText(
        "panelShipName",
        enteredShipName ||
        "WAITING FOR LIVE AIS"
    );


    setText(
        "monitorMmsi",
        enteredMmsi
            ? `MMSI ${enteredMmsi}`
            : "MMSI ----------"
    );


    setText(
        "panelMmsi",
        enteredMmsi
            ? `MMSI ${enteredMmsi}`
            : "MMSI ----------"
    );


    setText(
        "panelShipType",
        enteredShipType
    );
}


// ============================================================
// CONNECTION STATUS
// ============================================================

function updateConnectionStatus(
    connected,
    vesselCount
) {

    const status =
        document.getElementById(
            "connectionStatus"
        );


    if (!status) {
        return;
    }


    if (connected) {

        status.textContent =
            `AIS LIVE • ${vesselCount} VESSELS`;


        status.classList.remove(
            "offline"
        );


        status.classList.add(
            "online"
        );

    } else {

        status.textContent =
            "AIS OFFLINE";


        status.classList.remove(
            "online"
        );


        status.classList.add(
            "offline"
        );
    }
}


// ============================================================
// VESSEL COUNT
// ============================================================

function updateVesselCount(
    count
) {

    const safeCount =
        Number(count) || 0;


    const ids = [

        "vesselCount",

        "liveVesselCount",

        "monitorVesselCount",

        "aisVesselCount",

        "totalVessels"

    ];


    ids.forEach(
        id => {

            setText(
                id,
                String(
                    safeCount
                )
            );

        }
    );
}


// ============================================================
// NORMALIZE AIS DATA
// ============================================================

function normalizeAisVessel(
    raw
) {

    if (
        !raw ||
        typeof raw !== "object"
    ) {

        return null;
    }


    const latitude =
        Number(
            raw.latitude ??
            raw.lat
        );


    const longitude =
        Number(
            raw.longitude ??
            raw.lng ??
            raw.lon
        );


    const speed =
        Number(
            raw.speed ??
            raw.sog ??
            0
        );


    const course =
        Number(
            raw.course ??
            raw.cog ??
            0
        );


    const heading =
        Number(
            raw.heading ??
            raw.hdg ??
            raw.true_heading ??
            course
        );


    const mmsi =
        raw.mmsi != null
            ? Number(
                raw.mmsi
            )
            : null;


    const shipName =
        raw.ship_name ??
        raw.shipName ??
        raw.name ??
        "UNKNOWN VESSEL";


    return {

        ...raw,


        mmsi:
            Number.isFinite(
                mmsi
            )
                ? mmsi
                : null,


        ship_name:
            String(
                shipName
            ),


        latitude:
            Number.isFinite(
                latitude
            )
                ? latitude
                : null,


        longitude:
            Number.isFinite(
                longitude
            )
                ? longitude
                : null,


        speed:
            Number.isFinite(
                speed
            )
                ? speed
                : 0,


        course:
            Number.isFinite(
                course
            )
                ? course
                : 0,


        heading:
            Number.isFinite(
                heading
            )
                ? heading
                : course,


        lat:
            Number.isFinite(
                latitude
            )
                ? latitude
                : null,


        lng:
            Number.isFinite(
                longitude
            )
                ? longitude
                : null,


        sog:
            Number.isFinite(
                speed
            )
                ? speed
                : 0,


        cog:
            Number.isFinite(
                course
            )
                ? course
                : 0,


        hdg:
            Number.isFinite(
                heading
            )
                ? heading
                : course
    };
}


// ============================================================
// WAITING STATE
// ============================================================

function setMonitoringWaiting() {

    setText(
        "monitorShipName",
        "WAITING FOR LIVE AIS"
    );


    setText(
        "panelShipName",
        "WAITING FOR LIVE AIS"
    );


    setText(
        "monitorMmsi",
        enteredMmsi
            ? `MMSI ${enteredMmsi}`
            : "MMSI ----------"
    );


    setText(
        "panelMmsi",
        enteredMmsi
            ? `MMSI ${enteredMmsi}`
            : "MMSI ----------"
    );


    setText(
        "liveSpeed",
        "-- km/h"
    );


    setText(
        "liveCourse",
        "--°"
    );


    setText(
        "liveHeading",
        "--°"
    );


    setText(
        "liveLat",
        "--"
    );


    setText(
        "liveLon",
        "--"
    );


    setText(
        "monitorLatitude",
        "--"
    );


    setText(
        "monitorLongitude",
        "--"
    );


    setText(
        "predictionStatus",
        "WAITING"
    );


    setText(
        "overallRisk",
        "WAITING"
    );


    setText(
        "overallProbability",
        "--"
    );


    setText(
        "waveRisk",
        "WAITING"
    );


    setText(
        "waveProbability",
        "--"
    );


    setText(
        "xgbStatus",
        "WAITING FOR NDBC"
    );
}


// ============================================================
// BACKEND UNAVAILABLE
// ============================================================

function setMonitoringUnavailable() {

    setText(
        "monitorShipName",
        "AIS CONNECTION ERROR"
    );


    setText(
        "panelShipName",
        "AIS CONNECTION ERROR"
    );


    setText(
        "monitorMmsi",
        enteredMmsi
            ? `MMSI ${enteredMmsi}`
            : "MMSI ----------"
    );


    setText(
        "liveSpeed",
        "-- km/h"
    );


    setText(
        "liveCourse",
        "--°"
    );


    setText(
        "liveHeading",
        "--°"
    );


    setText(
        "liveLat",
        "--"
    );


    setText(
        "liveLon",
        "--"
    );


    setText(
        "monitorLatitude",
        "--"
    );


    setText(
        "monitorLongitude",
        "--"
    );


    setText(
        "predictionStatus",
        "OFFLINE"
    );


    setText(
        "overallRisk",
        "UNKNOWN"
    );


    setText(
        "overallProbability",
        "0.0%"
    );


    setText(
        "waveRisk",
        "OFFLINE"
    );


    setText(
        "waveProbability",
        "--"
    );


    setText(
        "xgbStatus",
        "OFFLINE"
    );


    setText(
        "lstmPrediction",
        "OFFLINE"
    );


    setText(
        "lstmStatus",
        "OFFLINE"
    );


    updateProgress(
        "riskProgress",
        0
    );


    updateProgress(
        "waveProgress",
        0
    );


    updateConnectionStatus(
        false,
        0
    );


    const message =
        document.getElementById(
            "safetyMessage"
        );


    if (message) {

        message.textContent =
            "Unable to connect to the AI marine backend.";
    }
}


// ============================================================
// SHIP ANIMATION
// ============================================================

function animateShip(
    course
) {

    const candidates = [

        document.getElementById(
            "vesselIcon"
        ),

        document.getElementById(
            "shipScene"
        )

    ];


    const ship =
        candidates.find(
            element =>
                element
        );


    if (!ship) {
        return;
    }


    const numericCourse =
        Number(course);


    if (
        !Number.isFinite(
            numericCourse
        )
    ) {

        return;
    }


    ship.style.transform =
        `rotate(${numericCourse}deg)`;


    ship.style.setProperty(
        "--ship-heading",
        `${numericCourse}deg`
    );


    ship.classList.add(
        "ship-moving"
    );
}


// ============================================================
// VESSEL DISPLAY
// ============================================================

function updateVesselDisplay(
    vessel
) {

    if (!vessel) {
        return;
    }


    const latitude =
        Number(
            vessel.latitude ??
            vessel.lat
        );


    const longitude =
        Number(
            vessel.longitude ??
            vessel.lng ??
            vessel.lon
        );


    // AIS SOG is in knots.
    const speedKnots =
        Number(
            vessel.speed ??
            vessel.sog ??
            0
        );


    const course =
        Number(
            vessel.course ??
            vessel.cog ??
            0
        );


    const heading =
        Number(
            vessel.heading ??
            vessel.hdg ??
            vessel.true_heading ??
            course
        );


    const speedKmh =
        speedKnots * 1.852;


    setText(
        "liveSpeed",
        Number.isFinite(
            speedKnots
        )
            ? `${speedKmh.toFixed(1)} km/h`
            : "-- km/h"
    );


    setText(
        "liveCourse",
        Number.isFinite(
            course
        )
            ? `${course.toFixed(1)}°`
            : "--°"
    );


    setText(
        "liveHeading",
        Number.isFinite(
            heading
        )
            ? `${heading.toFixed(1)}°`
            : "--°"
    );


    setText(
        "liveLat",
        Number.isFinite(
            latitude
        )
            ? latitude.toFixed(4)
            : "--"
    );


    setText(
        "monitorLatitude",
        Number.isFinite(
            latitude
        )
            ? latitude.toFixed(4)
            : "--"
    );


    setText(
        "liveLon",
        Number.isFinite(
            longitude
        )
            ? longitude.toFixed(4)
            : "--"
    );


    setText(
        "monitorLongitude",
        Number.isFinite(
            longitude
        )
            ? longitude.toFixed(4)
            : "--"
    );


    const shipName =
        vessel.ship_name ||
        vessel.shipName ||
        vessel.name ||
        "UNKNOWN VESSEL";


    setText(
        "monitorShipName",
        String(
            shipName
        ).trim()
    );


    setText(
        "panelShipName",
        String(
            shipName
        ).trim()
    );


    if (vessel.mmsi) {

        setText(
            "monitorMmsi",
            `MMSI ${vessel.mmsi}`
        );


        setText(
            "panelMmsi",
            `MMSI ${vessel.mmsi}`
        );
    }


    const shipType =
        vessel.ship_type ??
        vessel.shipType ??
        vessel.vessel_type ??
        vessel.vesselType ??
        enteredShipType;


    setText(
        "panelShipType",
        shipType || "--"
    );


    animateShip(
        course
    );
}


// ============================================================
// OVERALL VESSEL RISK
// ============================================================

function updateOverallRisk(
    level,
    probability
) {

    let normalized =
        String(
            level ||
            "MONITORING"
        )
            .toUpperCase()
            .replace(
                /\s+/g,
                "_"
            );


    if (
        normalized === "1" ||
        normalized === "TRUE"
    ) {

        normalized =
            "HIGH";
    }


    if (
        normalized === "0" ||
        normalized === "FALSE"
    ) {

        normalized =
            "LOW";
    }


    let displayLevel =
        normalized;


    if (
        normalized ===
        "HIGH_WAVE"
    ) {

        displayLevel =
            "HIGH WAVE";

    } else if (
        normalized ===
        "MODERATE"
    ) {

        displayLevel =
            "MODERATE";

    } else if (
        normalized ===
        "LOW"
    ) {

        displayLevel =
            "LOW";

    } else if (
        normalized ===
        "HIGH"
    ) {

        displayLevel =
            "HIGH";

    } else if (
        normalized ===
        "NORMAL"
    ) {

        displayLevel =
            "NORMAL";
    }


    setText(
        "predictionStatus",
        displayLevel
    );


    setText(
        "overallRisk",
        displayLevel
    );


    setText(
        "riskLevel",
        displayLevel
    );


    const numericProbability =
        Number(
            probability
        );


    if (
        Number.isFinite(
            numericProbability
        )
    ) {

        setText(
            "overallProbability",
            `${numericProbability.toFixed(1)}%`
        );


        setText(
            "riskProbability",
            `${numericProbability.toFixed(1)}%`
        );


        updateProgress(
            "riskProgress",
            numericProbability
        );

    } else {

        setText(
            "overallProbability",
            "--"
        );


        setText(
            "riskProbability",
            "--"
        );


        updateProgress(
            "riskProgress",
            0
        );
    }


    setRiskClass(
        "predictionStatus",
        normalized
    );


    setRiskClass(
        "overallRisk",
        normalized
    );
}


// ============================================================
// NDBC STORM DISPLAY
//
// IMPORTANT:
// This is ONLY for NDBC storm/high-wave detection.
//
// Do NOT use:
// vessel.xgboost_risk
//
// vessel.xgboost_risk is the normal vessel-risk model.
// ============================================================

function updateStormDisplay(
    vessel
) {

    if (!vessel) {
        return;
    }


    const storm =
        vessel.ndbc_storm_prediction ??
        vessel.high_wave_prediction ??
        vessel.storm_prediction ??
        null;


    if (!storm) {

        setText(
            "waveRisk",
            "WAITING"
        );


        setText(
            "waveProbability",
            "--"
        );


        setText(
            "xgbStatus",
            "WAITING FOR NDBC"
        );


        updateProgress(
            "waveProgress",
            0
        );


        return;
    }


    let level =
        String(
            storm.hazard ??
            storm.risk_level ??
            storm.level ??
            storm.prediction ??
            "NORMAL"
        )
            .toUpperCase()
            .replace(
                /\s+/g,
                "_"
            );


    if (
        level === "1" ||
        level === "TRUE"
    ) {

        level =
            "HIGH_WAVE";
    }


    if (
        level === "0" ||
        level === "FALSE"
    ) {

        level =
            "NORMAL";
    }


    let probability =
        Number(
            storm.probability_percent
        );


    if (
        !Number.isFinite(
            probability
        )
    ) {

        probability =
            Number(
                storm.probability
            ) * 100;
    }


    if (
        !Number.isFinite(
            probability
        )
    ) {

        probability = 0;
    }


    probability =
        Math.max(
            0,
            Math.min(
                100,
                probability
            )
        );


    const displayLevel =
        level ===
        "HIGH_WAVE"
            ? "HIGH WAVE"
            : level;


    setText(
        "waveRisk",
        displayLevel
    );


    setText(
        "waveProbability",
        `${probability.toFixed(1)}%`
    );


    setText(
        "xgbStatus",
        "NDBC XGBOOST"
    );


    setRiskClass(
        "waveRisk",
        level
    );


    updateProgress(
        "waveProgress",
        probability
    );


    // --------------------------------------------------------
    // ROUTE DECISION
    //
    // HIGH WAVE -> SAFEST
    // NORMAL    -> OPTIMIZED
    // --------------------------------------------------------

    if (
        level ===
        "HIGH_WAVE"
    ) {

        vessel.route_hazard =
            "HIGH_WAVE";


        vessel.route_mode =
            "SAFEST";


        // Environmental hazard overrides
        // normal vessel risk on the display.

        updateOverallRisk(
            "HIGH_WAVE",
            probability
        );

    } else {

        vessel.route_hazard =
            "NORMAL";


        vessel.route_mode =
            "OPTIMIZED";
    }


    console.log(
        "NDBC STORM DECISION:",
        {
            hazard:
                vessel.route_hazard,

            route_mode:
                vessel.route_mode,

            probability:
                probability
        }
    );
}


// ============================================================
// NDBC XGBOOST STORM PREDICTION
//
// BACKEND FLOW:
//
// GET /ndbc/observations
//          |
//          v
// 3 observations
//          |
//          v
// POST /predict/high-wave
//          |
//          v
// NDBC XGBoost
//          |
//          +---- HIGH_WAVE
//          |
//          +---- NORMAL
// ============================================================

async function getNDBCStormPrediction() {

    const vessel =
        currentVessel;


    if (!vessel) {

        console.warn(
            "NDBC: No current vessel selected."
        );

        return null;
    }


    try {

        console.log(
            "================================================"
        );


        console.log(
            "FETCHING LIVE NDBC OBSERVATIONS..."
        );


        // ----------------------------------------------------
        // STEP 1
        // GET NDBC OBSERVATIONS
        // ----------------------------------------------------

        const ndbcResponse =
            await fetch(
                `${API_BASE}/ndbc/observations?t=${Date.now()}`,
                {
                    method:
                        "GET",

                    cache:
                        "no-store",

                    headers: {
                        "Accept":
                            "application/json"
                    }
                }
            );


        if (
            !ndbcResponse.ok
        ) {

            throw new Error(
                `NDBC observations returned ` +
                `${ndbcResponse.status}`
            );
        }


        const ndbcData =
            await ndbcResponse.json();


        console.log(
            "NDBC DATA:",
            ndbcData
        );


        // ----------------------------------------------------
        // STEP 2
        // EXTRACT OBSERVATIONS
        // ----------------------------------------------------

        const observations =
            Array.isArray(
                ndbcData.observations
            )
                ? ndbcData.observations
                : Array.isArray(
                    ndbcData.data
                )
                    ? ndbcData.data
                    : [];


        if (
            observations.length <
            3
        ) {

            throw new Error(
                "NDBC returned fewer than " +
                "3 observations."
            );
        }


        // ----------------------------------------------------
        // STEP 3
        // USE EXACTLY 3 OBSERVATIONS
        // ----------------------------------------------------

        const latestThree =
            observations.slice(
                -3
            );


        if (
            latestThree.length !==
            3
        ) {

            throw new Error(
                "NDBC XGBoost requires " +
                "exactly 3 observations."
            );
        }


        // ----------------------------------------------------
        // STEP 4
        // REQUIRED MODEL FEATURES
        // ----------------------------------------------------

        const requiredFeatures = [

            "WVHT",

            "WSPD",

            "GST",

            "DPD",

            "APD",

            "PRES",

            "ATMP",

            "WTMP"

        ];


        const valid =
            latestThree.every(
                observation => {

                    return requiredFeatures.every(
                        feature => {

                            return Number.isFinite(
                                Number(
                                    observation[
                                        feature
                                    ]
                                )
                            );

                        }
                    );

                }
            );


        if (!valid) {

            throw new Error(
                "NDBC observations contain " +
                "missing or invalid features."
            );
        }


        console.log(
            "NDBC 3 OBSERVATIONS:",
            latestThree
        );


        // ----------------------------------------------------
        // STEP 5
        // CALL NDBC XGBOOST
        // ----------------------------------------------------

        console.log(
            "CALLING NDBC XGBOOST..."
        );


        const stormResponse =
            await fetch(
                `${API_BASE}/predict/high-wave`,
                {
                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"

                    },

                    cache:
                        "no-store",

                    body:
                        JSON.stringify({
                            observations:
                                latestThree
                        })
                }
            );


        if (
            !stormResponse.ok
        ) {

            const errorText =
                await stormResponse.text();


            throw new Error(
                `NDBC XGBoost returned ` +
                `${stormResponse.status}: ` +
                errorText
            );
        }


        const storm =
            await stormResponse.json();


        console.log(
            "NDBC XGBOOST RESULT:",
            storm
        );


        // ----------------------------------------------------
        // STEP 6
        // SAVE RESULT
        // ----------------------------------------------------

        currentVessel
            .ndbc_storm_prediction =
                storm;


        currentVessel
            .ndbc_observations =
                latestThree;


        // ----------------------------------------------------
        // STEP 7
        // UPDATE DISPLAY
        // ----------------------------------------------------

        updateStormDisplay(
            currentVessel
        );


        console.log(
            "================================================"
        );


        return storm;


    } catch (error) {

        console.error(
            "NDBC STORM PREDICTION ERROR:",
            error
        );


        setText(
            "waveRisk",
            "NDBC ERROR"
        );


        setText(
            "waveProbability",
            "--"
        );


        setText(
            "xgbStatus",
            "NDBC ERROR"
        );


        updateProgress(
            "waveProgress",
            0
        );


        return null;
    }
}


// ============================================================
// LSTM DISPLAY
// ============================================================

function updateLSTMDisplay(
    vessel
) {

    const statusElement =
        document.getElementById(
            "lstmStatus"
        );


    const predictionElement =
        document.getElementById(
            "lstmPrediction"
        );


    if (
        !statusElement ||
        !predictionElement
    ) {

        return;
    }


    if (!vessel) {

        setText(
            "lstmPrediction",
            "-- m"
        );


        setText(
            "lstmStatus",
            "WAITING FOR LSTM"
        );


        return;
    }


    const lstm =
        vessel.lstm_prediction ??
        vessel.lstmPrediction ??
        vessel.lstm ??
        vessel.lstm_result ??
        vessel.wave_prediction ??
        null;


    if (!lstm) {

        setText(
            "lstmPrediction",
            "-- m"
        );


        setText(
            "lstmStatus",
            "WAITING FOR LSTM"
        );


        return;
    }


    const predicted =
        Number(
            lstm.predicted_vhm0_m ??
            lstm.prediction ??
            lstm.predicted_wave_height ??
            lstm.wave_height ??
            lstm.vhm0 ??
            lstm.VHM0
        );


    if (
        !Number.isFinite(
            predicted
        )
    ) {

        setText(
            "lstmPrediction",
            "-- m"
        );


        setText(
            "lstmStatus",
            "WAITING FOR LSTM"
        );


        return;
    }


    let waveStatus =
        String(
            lstm.wave_status ??
            lstm.status ??
            ""
        )
            .toUpperCase()
            .replace(
                /\s+/g,
                "_"
            );


    if (!waveStatus) {

        if (
            predicted >=
            3.0
        ) {

            waveStatus =
                "HIGH_WAVE";

        } else if (
            predicted >=
            2.0
        ) {

            waveStatus =
                "MODERATE";

        } else {

            waveStatus =
                "LOW";
        }
    }


    setText(
        "lstmPrediction",
        `${predicted.toFixed(2)} m`
    );


    setText(
        "lstmStatus",
        waveStatus.replace(
            /_/g,
            " "
        )
    );


    setRiskClass(
        "lstmStatus",
        waveStatus
    );


    console.log(
        "LSTM DISPLAY UPDATED:",
        {
            prediction:
                predicted,

            status:
                waveStatus
        }
    );
}


// ============================================================
// LSTM PREDICTION
//
// IMPORTANT:
// This does NOT convert AIS speed/course into wave values.
//
// The backend must provide the actual LSTM result.
// ============================================================

async function getLSTMPrediction() {

    if (
        lstmRequestInProgress
    ) {

        return null;
    }


    const vessel =
        currentVessel ||
        allVessels[0];


    if (!vessel) {

        updateLSTMDisplay(
            null
        );

        return null;
    }


    if (!vessel.mmsi) {

        updateLSTMDisplay(
            null
        );

        return null;
    }


    lstmRequestInProgress =
        true;


    try {

        console.log(
            "CALLING LIVE LSTM:",
            vessel.mmsi
        );


        const response =
            await fetch(
                `${API_BASE}/predict/lstm/live?t=${Date.now()}`,
                {
                    method:
                        "GET",

                    cache:
                        "no-store",

                    headers: {
                        "Accept":
                            "application/json"
                    }
                }
            );


        if (
            !response.ok
        ) {

            throw new Error(
                `LSTM returned ` +
                `${response.status}`
            );
        }


        const result =
            await response.json();


        console.log(
            "LSTM RESULT:",
            result
        );


        currentVessel
            .lstm_prediction =
                result;


        updateLSTMDisplay(
            currentVessel
        );


        return result;


    } catch (error) {

        console.warn(
            "LSTM prediction unavailable:",
            error
        );


        setText(
            "lstmStatus",
            "LSTM WAITING"
        );


        setText(
            "lstmPrediction",
            "-- m"
        );


        return null;


    } finally {

        lstmRequestInProgress =
            false;
    }
}


// ============================================================
// UPDATE ALL AI RESULTS
// ============================================================

function updateAIResults(
    vessel
) {

    if (!vessel) {
        return;
    }


    // --------------------------------------------------------
    // RANDOM FOREST VESSEL RISK
    // --------------------------------------------------------

    const risk =
        vessel.vessel_risk ||
        vessel.risk ||
        null;


    if (risk) {

        let probability =
            Number(
                risk.probability_percent ??
                risk.probability ??
                risk.risk_probability ??
                0
            );


        if (
            !Number.isFinite(
                probability
            )
        ) {

            probability = 0;
        }


        // If probability is 0-1,
        // convert it to percentage.

        if (
            probability > 0 &&
            probability <= 1
        ) {

            probability *=
                100;
        }


        let level =
            String(
                risk.risk_level ??
                risk.level ??
                risk.prediction ??
                "LOW"
            )
                .toUpperCase()
                .replace(
                    /\s+/g,
                    "_"
                );


        if (
            level === "1" ||
            level === "TRUE"
        ) {

            level =
                "HIGH";
        }


        if (
            level === "0" ||
            level === "FALSE"
        ) {

            level =
                "LOW";
        }


        updateOverallRisk(
            level,
            probability
        );

    } else {

        updateOverallRisk(
            "MONITORING",
            0
        );
    }


    // --------------------------------------------------------
    // NDBC XGBOOST
    // --------------------------------------------------------

    updateStormDisplay(
        vessel
    );


    // --------------------------------------------------------
    // LSTM
    // --------------------------------------------------------

    updateLSTMDisplay(
        vessel
    );
}


// ============================================================
// FETCH LIVE AIS
// ============================================================

async function loadMonitoringData() {

    if (
        requestInProgress
    ) {

        return;
    }


    requestInProgress =
        true;


    try {

        console.log(
            "FETCHING LIVE AIS..."
        );


        // ----------------------------------------------------
        // GET AIS
        // ----------------------------------------------------

        const response =
            await fetch(
                `${API_BASE}/ais/vessels?t=${Date.now()}`,
                {
                    method:
                        "GET",

                    cache:
                        "no-store",

                    headers: {
                        "Accept":
                            "application/json"
                    }
                }
            );


        if (
            !response.ok
        ) {

            throw new Error(
                `AIS backend returned ` +
                `${response.status}`
            );
        }


        const data =
            await response.json();


        console.log(
            "AIS BACKEND RESPONSE:",
            data
        );


        // ----------------------------------------------------
        // SUPPORT BOTH:
        //
        // { vessels: [...] }
        //
        // AND:
        //
        // { data: [...] }
        // ----------------------------------------------------

        const rawVessels =
            Array.isArray(
                data.vessels
            )
                ? data.vessels
                : Array.isArray(
                    data.data
                )
                    ? data.data
                    : [];


        let vessels =
            rawVessels
                .map(
                    normalizeAisVessel
                )
                .filter(
                    vessel =>
                        vessel &&
                        vessel.mmsi &&
                        Number.isFinite(
                            vessel.latitude
                        ) &&
                        Number.isFinite(
                            vessel.longitude
                        )
                );


        // ----------------------------------------------------
        // OPTIONAL DEMO FALLBACK
        // ----------------------------------------------------

        if (
            DEMO_FALLBACK_ENABLED
        ) {

            const demoVessel =
                normalizeAisVessel({

                    mmsi:
                        235090959,

                    ship_name:
                        "AIS DEMO VESSEL 4",

                    latitude:
                        50.70678,

                    longitude:
                        -1.98339,

                    speed:
                        6.9,

                    course:
                        133.7,

                    heading:
                        133.7,

                    vessel_type:
                        "other"

                });


            if (
                demoVessel &&
                !vessels.some(
                    vessel =>
                        String(
                            vessel.mmsi
                        ) ===
                        String(
                            demoVessel.mmsi
                        )
                )
            ) {

                vessels.push(
                    demoVessel
                );
            }
        }


        // ----------------------------------------------------
        // SAVE ALL VESSELS
        // ----------------------------------------------------

        allVessels =
            vessels;


        updateVesselCount(
            vessels.length
        );


        updateConnectionStatus(
            vessels.length >
            0,
            vessels.length
        );


        console.log(
            "AIS VESSEL COUNT:",
            vessels.length
        );


        // ----------------------------------------------------
        // NO VESSELS
        // ----------------------------------------------------

        if (
            vessels.length ===
            0
        ) {

            setMonitoringWaiting();

            return;
        }


        // ----------------------------------------------------
        // SELECT VESSEL
        // ----------------------------------------------------

        let vessel =
            null;


        // First:
        // MMSI from previous page.

        if (
            enteredMmsi
        ) {

            vessel =
                vessels.find(
                    item =>
                        String(
                            item.mmsi
                        ) ===
                        String(
                            enteredMmsi
                        )
                ) ||
                null;
        }


        // Second:
        // Previously selected vessel.

        if (
            !vessel &&
            currentVessel?.mmsi
        ) {

            vessel =
                vessels.find(
                    item =>
                        String(
                            item.mmsi
                        ) ===
                        String(
                            currentVessel.mmsi
                        )
                ) ||
                null;
        }


        // Third:
        // First available vessel.

        if (!vessel) {

            vessel =
                vessels[0];
        }


        if (!vessel) {

            setMonitoringWaiting();

            return;
        }


        console.log(
            "SELECTED VESSEL:",
            vessel
        );


        // ----------------------------------------------------
        // SAVE CURRENT VESSEL FIRST
        // ----------------------------------------------------

        currentVessel =
            vessel;


        // ----------------------------------------------------
        // GET VESSEL AI RISK
        // ----------------------------------------------------

        if (
            vessel.mmsi
        ) {

            try {

                const riskResponse =
                    await fetch(
                        `${API_BASE}/ais/risk/${encodeURIComponent(vessel.mmsi)}?t=${Date.now()}`,
                        {
                            method:
                                "GET",

                            cache:
                                "no-store",

                            headers: {
                                "Accept":
                                    "application/json"
                            }
                        }
                    );


                if (
                    riskResponse.ok
                ) {

                    const riskData =
                        await riskResponse.json();


                    console.log(
                        "LIVE RISK DATA:",
                        riskData
                    );


                    // ------------------------------------------------
                    // RANDOM FOREST
                    // ------------------------------------------------

                    if (
                        riskData.random_forest
                    ) {

                        vessel.vessel_risk =
                            riskData.random_forest;
                    }


                    // ------------------------------------------------
                    // NORMAL XGBOOST
                    //
                    // This is vessel risk.
                    // It is NOT storm detection.
                    // ------------------------------------------------

                    vessel.xgboost_risk =
                        riskData.xgboost ??
                        riskData.xgboost_risk ??
                        null;


                    // ------------------------------------------------
                    // LSTM
                    // ------------------------------------------------

                    vessel.lstm_prediction =
                        riskData.lstm_prediction ??
                        riskData.lstm ??
                        riskData.lstm_result ??
                        riskData.wave_prediction ??
                        null;
                }


            } catch (
                riskError
            ) {

                console.warn(
                    "LIVE RISK REQUEST FAILED:",
                    riskError
                );
            }
        }


        // ----------------------------------------------------
        // NDBC XGBOOST
        //
        // THIS IS THE STORM DETECTION MODEL.
        // ----------------------------------------------------

        await getNDBCStormPrediction();


        // ----------------------------------------------------
        // UPDATE VESSEL UI
        // ----------------------------------------------------

        updateVesselDisplay(
            vessel
        );


        // ----------------------------------------------------
        // UPDATE AI UI
        // ----------------------------------------------------

        updateAIResults(
            vessel
        );


        // ----------------------------------------------------
        // SAFETY MESSAGE
        // ----------------------------------------------------

        const safetyMessage =
            document.getElementById(
                "safetyMessage"
            );


        if (
            safetyMessage
        ) {

            safetyMessage.textContent =
                `LIVE AIS TRACKING ACTIVE — ${vessel.ship_name}`;
        }


    } catch (
        error
    ) {

        console.error(
            "MONITORING ERROR:",
            error
        );


        updateConnectionStatus(
            false,
            0
        );


        setMonitoringUnavailable();


    } finally {

        requestInProgress =
            false;
    }
}


// ============================================================
// MANUAL VESSEL SELECTION
// ============================================================

async function selectVessel(
    mmsi
) {

    const vessel =
        allVessels.find(
            item =>
                String(
                    item.mmsi
                ) ===
                String(
                    mmsi
                )
        );


    if (!vessel) {

        console.warn(
            "VESSEL NOT FOUND:",
            mmsi
        );

        return;
    }


    currentVessel =
        vessel;


    updateVesselDisplay(
        vessel
    );


    updateAIResults(
        vessel
    );


    // Refresh NDBC for
    // newly selected vessel.

    await getNDBCStormPrediction();


    console.log(
        "MANUALLY SELECTED VESSEL:",
        vessel
    );
}


// ============================================================
// BACKEND HEALTH
// ============================================================

async function checkBackendHealth() {

    try {

        const response =
            await fetch(
                `${API_BASE}/health?t=${Date.now()}`,
                {
                    method:
                        "GET",

                    cache:
                        "no-store",

                    headers: {
                        "Accept":
                            "application/json"
                    }
                }
            );


        if (
            !response.ok
        ) {

            throw new Error(
                `Health HTTP ` +
                `${response.status}`
            );
        }


        const data =
            await response.json();


        console.log(
            "BACKEND HEALTH:",
            data
        );


        const backendStatus =
            document.getElementById(
                "backendStatus"
            );


        if (
            backendStatus
        ) {

            backendStatus.textContent =
                "BACKEND ONLINE";


            backendStatus.classList.remove(
                "offline"
            );


            backendStatus.classList.add(
                "online"
            );
        }


        if (
            data.ais &&
            Number.isFinite(
                Number(
                    data.ais.vessel_count
                )
            )
        ) {

            updateVesselCount(
                Number(
                    data.ais.vessel_count
                )
            );
        }


    } catch (
        error
    ) {

        console.warn(
            "BACKEND HEALTH CHECK FAILED:",
            error
        );


        const backendStatus =
            document.getElementById(
                "backendStatus"
            );


        if (
            backendStatus
        ) {

            backendStatus.textContent =
                "BACKEND OFFLINE";


            backendStatus.classList.remove(
                "online"
            );


            backendStatus.classList.add(
                "offline"
            );
        }
    }
}


// ============================================================
// ROUTE BUTTON
// ============================================================

function setupRouteButton() {

    const routeBtn =
        document.getElementById(
            "routeButton"
        );


    if (!routeBtn) {
        return;
    }


    routeBtn.addEventListener(
        "click",
        function() {

            if (
                !currentVessel
            ) {

                alert(
                    "No live AIS vessel is currently selected."
                );

                return;
            }


            // Ensure the latest NDBC
            // decision is included.

            if (
                !currentVessel.route_hazard
            ) {

                currentVessel.route_hazard =
                    "NORMAL";
            }


            if (
                !currentVessel.route_mode
            ) {

                currentVessel.route_mode =
                    currentVessel.route_hazard ===
                    "HIGH_WAVE"
                        ? "SAFEST"
                        : "OPTIMIZED";
            }


            sessionStorage.setItem(
                "marineCurrentVessel",
                JSON.stringify(
                    currentVessel
                )
            );


            sessionStorage.setItem(
                "marineVesselDetails",
                JSON.stringify(
                    shipDetails
                )
            );


            window.location.href =
                "route.html";
        }
    );
}


// ============================================================
// BACK BUTTON
// ============================================================

function setupBackButton() {

    const backBtn =
        document.getElementById(
            "backBtn"
        );


    if (!backBtn) {
        return;
    }


    backBtn.addEventListener(
        "click",
        function() {

            window.location.href =
                "index.html";
        }
    );
}


// ============================================================
// PUBLIC DEBUG API
// ============================================================

window.MarineMonitoring = {

    getCurrentVessel:
        function() {

            return currentVessel;
        },


    getAllVessels:
        function() {

            return allVessels;
        },


    selectVessel:
        function(mmsi) {

            return selectVessel(
                mmsi
            );
        },


    refresh:
        function() {

            return loadMonitoringData();
        },


    refreshNDBC:
        function() {

            return getNDBCStormPrediction();
        },


    refreshLSTM:
        function() {

            return getLSTMPrediction();
        },


    health:
        function() {

            return checkBackendHealth();
        }
};


// ============================================================
// LOWERCASE COMPATIBILITY API
// ============================================================

window.marineMonitoring = {

    getCurrentVessel:
        function() {

            return currentVessel;
        },


    getAllVessels:
        function() {

            return allVessels;
        },


    selectVessel:
        function(mmsi) {

            return selectVessel(
                mmsi
            );
        },


    refresh:
        function() {

            return loadMonitoringData();
        },


    getLSTM:
        function() {

            if (
                !currentVessel
            ) {

                return null;
            }


            return (
                currentVessel.lstm_prediction ||
                currentVessel.lstm ||
                null
            );
        },


    getNDBC:
        function() {

            if (
                !currentVessel
            ) {

                return null;
            }


            return (
                currentVessel.ndbc_storm_prediction ||
                null
            );
        }
};


// ============================================================
// START MONITORING PAGE
// ============================================================

function startMonitoringPage() {

    console.log(
        "================================================"
    );


    console.log(
        "AI MARINE MONITORING STARTED"
    );


    console.log(
        "API:",
        API_BASE
    );


    console.log(
        "ENTERED SHIP:",
        enteredShipName
    );


    console.log(
        "ENTERED MMSI:",
        enteredMmsi
    );


    console.log(
        "================================================"
    );


    // --------------------------------------------------------
    // INITIAL DISPLAY
    // --------------------------------------------------------

    displayInitialShipDetails();


    // --------------------------------------------------------
    // BUTTONS
    // --------------------------------------------------------

    setupRouteButton();

    setupBackButton();


    // --------------------------------------------------------
    // FIRST AIS REQUEST
    // --------------------------------------------------------

    loadMonitoringData();


    // --------------------------------------------------------
    // AIS REFRESH
    // --------------------------------------------------------

    if (
        !monitoringTimer
    ) {

        monitoringTimer =
            setInterval(
                loadMonitoringData,
                AIS_REFRESH_MS
            );
    }


    // --------------------------------------------------------
    // BACKEND HEALTH
    // --------------------------------------------------------

    checkBackendHealth();


    if (
        !healthTimer
    ) {

        healthTimer =
            setInterval(
                checkBackendHealth,
                HEALTH_REFRESH_MS
            );
    }


    // --------------------------------------------------------
    // LSTM
    // --------------------------------------------------------

    getLSTMPrediction();


    if (
        !lstmTimer
    ) {

        lstmTimer =
            setInterval(
                getLSTMPrediction,
                LSTM_REFRESH_MS
            );
    }
}


// ============================================================
// CLEANUP
// ============================================================

window.addEventListener(
    "beforeunload",
    function() {

        if (
            monitoringTimer
        ) {

            clearInterval(
                monitoringTimer
            );


            monitoringTimer =
                null;
        }


        if (
            healthTimer
        ) {

            clearInterval(
                healthTimer
            );


            healthTimer =
                null;
        }


        if (
            lstmTimer
        ) {

            clearInterval(
                lstmTimer
            );


            lstmTimer =
                null;
        }
    }
);


// ============================================================
// DOM READY
// ============================================================

if (
    document.readyState ===
    "loading"
) {

    document.addEventListener(
        "DOMContentLoaded",
        startMonitoringPage,
        {
            once:
                true
        }
    );

} else {

    startMonitoringPage();
}