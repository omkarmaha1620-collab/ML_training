# AI Marine Monitoring Frontend

This frontend is intentionally designed as a marine command-center UI:

- Large interactive marine map
- Live AIS vessels
- Random Forest V2 vessel-risk display
- Animated risk-aware vessel markers
- Route planning controls
- Animated route visualization
- AI engine status panel
- Vessel intelligence panel
- Responsive layout

## Current backend integration

The frontend calls:

GET http://127.0.0.1:8000/ais/vessels

It expects the response structure already demonstrated by your FastAPI backend:

{
  "status": "success",
  "count": 1,
  "vessels": [
    {
      "mmsi": 419001812,
      "latitude": 13.092,
      "longitude": 80.2973,
      "speed": 0,
      "course": 152,
      "ship_name": "SSL KAVERI",
      "vessel_risk": {
        "prediction": 0,
        "probability": 0,
        "risk_level": "LOW",
        "features": {}
      }
    }
  ]
}

## Run

The easiest option is VS Code Live Server:

1. Open the `frontend` folder in VS Code.
2. Open `index.html`.
3. Use "Open with Live Server".
4. Keep FastAPI running on:
   http://127.0.0.1:8000

If your browser blocks the API because of CORS, add FastAPI CORS middleware to the backend.

## Important PPO note

The exact request schema for your PPO route endpoint was not supplied in the conversation.

Therefore `app.js` does NOT invent a PPO request body.

The button currently creates an animated marine route between the selected vessel and destination so the frontend can be developed and tested.

Once the exact `/optimize-route` or `/optimize-route-from-ais` schema is known, only `getOptimizedRoute()` needs to be replaced with the real API call.

Do not change the AIS or vessel-risk integration.
