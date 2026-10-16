# AgroShield — AgroDrone AI

A React + TypeScript prototype for exploring agricultural-image indicators and disaster-risk visualization. The public implementation is currently a dashboard prototype, not a validated remote-sensing or emergency-warning system.

## What the current code does

- Accepts an uploaded image and draws it to an HTML canvas.
- In agriculture mode, applies hand-written pixel-colour heuristics to estimate broad colour groups such as green vegetation, yellow/brown stress-like regions, water-like pixels, and dark pixels.
- In risk mode, attempts a Gemini image call for the limited labels `flood`, `lava`, and `earthquake`; pixel-based heuristics are also used as a fallback.
- Displays derived counts, charts, alert areas, risk summaries, and suggested actions in the React UI.
- Includes sample image choices for demonstration.

## Important limitations

- The agriculture pipeline is pixel-colour rule logic, not a trained segmentation model.
- The returned areas, alert zones, confidence-style values, and recommendations are not calibrated geospatial measurements or ground-truth-validated hazard predictions.
- The Gemini prompt supports only a narrow set of hazard labels; it should not be used for operational disaster detection or evacuation decisions.
- The sample images are linked from external image URLs and require network access.
- This repository does not currently demonstrate a validated building-footprint detector, photogrammetry pipeline, or calibrated area estimator.

## Security warning — fix before public model-backed deployment

`vite.config.ts` injects `GEMINI_API_KEY` into `process.env.GEMINI_API_KEY` at build time. Because the Gemini SDK is called from the frontend code, a real key can be exposed in the JavaScript bundle sent to browsers. **Do not deploy this prototype with an unrestricted production API key.** Move Gemini requests to a server-side endpoint, apply rate limits and input validation, and rotate any key that may already have been exposed.

## Tech stack

- React 19 and TypeScript
- Vite
- Tailwind CSS (Vite plugin)
- Recharts for charts
- Motion and Lucide React for UI effects/icons
- Google GenAI SDK for the experimental hazard-image call

## Run locally

Prerequisites: a current Node.js LTS release and npm.

```bash
git clone https://github.com/abhijay3003/drone_image_processing.git
cd drone_image_processing
npm install
```

For the Gemini-backed risk mode, create a local `.env.local` only for controlled development and put your local test key there. Do not commit this file. The current frontend key-injection design is unsafe for production; use a server-side proxy instead.

Start the Vite dev server:

```bash
npm run dev
```

Create a production bundle:

```bash
npm run build
```

Run the TypeScript check:

```bash
npm run lint
```

## Configuration

The repository includes an `.env.example` placeholder for host-specific configuration. Never replace placeholders with real credentials in a commit. The existing Vite configuration expects `GEMINI_API_KEY`, but that value is currently compiled into client-side code; follow the security warning above before using model-backed features on a public host.

## Recommended next steps

1. Move the Gemini call into a protected server endpoint and remove secret injection from the Vite client bundle.
2. Add tests for image decoding, pixel classification, error handling, and the server-side hazard response schema.
3. Evaluate against labelled, held-out images and report precision/recall for each class before claiming model performance.
4. Distinguish rule-derived visual indicators from ML predictions in the UI.
5. If area measurement is added, require ground sampling distance or map/geospatial calibration and test against reference measurements.

## License and attribution

Retain the Apache-2.0 attribution included in source files. Verify licenses for dependencies and the sample image sources before redistributing or deploying.
